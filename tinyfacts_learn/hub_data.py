"""Reading the tinyfacts dataset from the Hugging Face Hub.

The texts used to sit in a `tinyfacts-gen/` submodule, as one `.txt` file per
text. They now live in a dataset repository on the Hub, as `.jsonl` chunks of
one row per text.

Only the fetch is done here. The rows themselves are read by `DatasetStore`,
and described by `DatasetRecord`, both of which come from the `tinyfacts`
package that writes the chunks in the first place, so the schema followed here
cannot fall out of step with the one on the Hub.
"""

import os
import re
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from tinyfacts.dataset import DatasetRecord, DatasetStore, RecordFilter, resolve_token

#: The dataset repository the rows are read from.
DEFAULT_REPO_ID = "Stur86/tinyfacts"

#: Set this to read the rows from somewhere other than `DEFAULT_REPO_ID`.
REPO_ENV_VAR = "TINYFACTS_HF_REPO"

#: Any of these is used as the Hugging Face token when none is given. The same
#: names the `tinyfacts` CLI uses, so one token serves both.
TOKEN_ENV_VARS = ("TINYFACTS_HF_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN")

#: Where the chunks sit inside the repository, and what each one is called.
#: These match the `hub` block of the upstream `dataset.yaml`.
DATA_DIR = "data"
CHUNK_PREFIX = "tinyfacts"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class HubDataError(Exception):
    """The dataset could not be read from the Hub."""


def resolve_repo_id(repo_id: str | None = None) -> str:
    """The repository to read: the one given, the environment's, or the default."""
    return repo_id or os.environ.get(REPO_ENV_VAR) or DEFAULT_REPO_ID


def _load_dotenv() -> None:
    """Read a `.env` file, so a token kept there is found like the CLI finds it."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        return
    load_dotenv()


def download_snapshot(
    repo_id: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Fetch the dataset chunks and give back the folder they landed in.

    Only the `.jsonl` chunks are fetched, not the dataset card. Nothing is
    downloaded again if the local cache already holds it.

    Args:
        repo_id: The dataset repository. Defaults to `resolve_repo_id()`.
        revision: A branch, tag or commit sha. Defaults to the default branch.
        token: A Hugging Face token. Defaults to one from `TOKEN_ENV_VARS`.
    """
    repo_id = resolve_repo_id(repo_id)
    if token is None:
        _load_dotenv()
        token = resolve_token()

    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            token=token,
            allow_patterns=[f"{DATA_DIR}/*.jsonl"],
        )
    except RevisionNotFoundError as exc:
        raise HubDataError(
            f"No revision {revision!r} in the dataset repo '{repo_id}'."
        ) from exc
    except (RepositoryNotFoundError, GatedRepoError) as exc:
        raise HubDataError(
            f"Could not read the dataset repo '{repo_id}'. It is private, gated or "
            "does not exist, and no usable token was found. Set one of "
            f"{', '.join(TOKEN_ENV_VARS)} in the environment or in a .env file, or "
            f"set {REPO_ENV_VAR} to point at another repo."
        ) from exc
    except HfHubHTTPError as exc:
        raise HubDataError(
            f"Could not reach the Hugging Face Hub for '{repo_id}': {exc}"
        ) from exc

    return Path(path)


def snapshot_revision(snapshot_path: Path) -> str | None:
    """The commit sha a snapshot folder holds, when the cache layout gives it.

    The Hub cache keeps a snapshot at `.../snapshots/<sha>/`, so the folder
    names the revision without asking the Hub again. Runs record this, so a
    training run says which version of a dataset that keeps changing it saw.
    """
    name = Path(snapshot_path).name
    return name if _SHA_RE.match(name) else None


def load_records(
    repo_id: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    record_filter: RecordFilter | None = None,
) -> tuple[list[DatasetRecord], str | None]:
    """Read the dataset rows from the Hub.

    Rows are sorted by id, so the same filters always give the same rows in the
    same order however the chunks came down.

    Returns:
        The rows that passed the filter, and the commit sha they were read from
        (`None` when the cache layout did not give one).
    """
    snapshot_path = download_snapshot(repo_id=repo_id, revision=revision, token=token)
    store = DatasetStore.open(
        snapshot_path,
        data_dir=DATA_DIR,
        chunk_prefix=CHUNK_PREFIX,
    )

    records = list(store)
    if not records:
        raise HubDataError(
            f"No rows found in '{resolve_repo_id(repo_id)}'. Looked for "
            f"{DATA_DIR}/{CHUNK_PREFIX}-*.jsonl in {snapshot_path}."
        )

    if record_filter is not None:
        records = record_filter.apply(records)

    records.sort(key=lambda record: record.id)
    return records, snapshot_revision(snapshot_path)


def available_sources(records: list[DatasetRecord]) -> list[str]:
    """Every distinct `source` in the given rows, in name order."""
    return sorted({record.source for record in records if record.source})
