# dataset.py
"""The training set: rows from the Hub, tokenized into sliding windows."""

import hashlib
import warnings

import torch
from torch.utils.data import Dataset
from tinyfacts.dataset import DatasetRecord, RecordFilter

from .hub_data import available_sources, load_records
from .tokenizers import WordTokenizer


def _record_split(record_id: str, val_fraction: float, split_seed: int) -> str:
    """Deterministically assign a row to the train or val split.

    Splitting at the *row* level (rather than slicing the concatenated token
    stream) keeps both splits representative of every source and guarantees
    no sliding window ever straddles the boundary, so val windows share no
    tokens with train windows.
    """
    digest = hashlib.sha256(f"{split_seed}:{record_id}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if fraction < val_fraction else "train"


class TinyfactsDataset(Dataset):
    """Rows of the tinyfacts dataset, tokenized into `(input_ids, target_ids)` pairs.

    The texts are read from the dataset repository on the Hugging Face Hub. Rows
    are taken in id order, so the same filters always give the same token stream.

    Rows that use words outside the Thing Explainer list are already left out
    when the dataset is built, so no checking is done here. Pass `validate=True`
    to check anyway, at the cost of a pass over every row.

    Args:
        sources: Which runs to train on, by `source` name. All of them if None.
        context_size: Number of tokens per training window.
        tokenizer: Optional WordTokenizer instance; a default one is made if None.
        repo_id: The dataset repository. Defaults to the one `hub_data` names.
        revision: A branch, tag or commit sha to pin the run to.
        token: A Hugging Face token. Read from the environment or `.env` if None.
        filters: Extra row filters, passed to `RecordFilter.build` — `min_words`,
            `max_words`, `model`, `tag`, `has_instruction`, and the `id`, `title`,
            `text` and `instruction` regular expressions.
        validate: Check every row against the word list before using it.
        stride: Step between consecutive window start positions. The default of 1
            yields maximally overlapping windows, which means one "epoch" over this
            dataset is really ``context_size`` passes over every token. Set
            ``stride=context_size`` for non-overlapping windows and an epoch count
            that means one pass over the corpus.
        split: "all" (default), "train", or "val" — which side of the row-level
            split to load.
        val_fraction: Fraction of *rows* assigned to the val split. Ignored when
            ``split="all"``.
        split_seed: Seed for the deterministic row-level split assignment.
    """

    def __init__(
        self,
        sources: list[str] | None = None,
        context_size: int = 128,
        tokenizer: WordTokenizer | None = None,
        repo_id: str | None = None,
        revision: str | None = None,
        token: str | None = None,
        filters: dict | None = None,
        validate: bool = False,
        stride: int = 1,
        split: str = "all",
        val_fraction: float = 0.05,
        split_seed: int = 0,
    ):
        if split not in ("all", "train", "val"):
            raise ValueError(f"split must be 'all', 'train' or 'val', got {split!r}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self._context_size = context_size
        self._stride = stride
        self._split = split
        self._tokenizer = tokenizer or WordTokenizer(ignore_case=True, digits=True)

        filter_args = dict(filters or {})
        if sources:
            filter_args["source"] = list(sources)
        record_filter = RecordFilter.build(**filter_args) if filter_args else None

        records, resolved_revision = load_records(
            repo_id=repo_id,
            revision=revision,
            token=token,
            record_filter=record_filter,
        )
        self._revision = resolved_revision

        if not records:
            self._raise_no_rows(sources, repo_id, revision, token)

        if validate:
            records = self._drop_invalid(records)

        if split != "all":
            records = [
                record
                for record in records
                if _record_split(record.id, val_fraction, split_seed) == split
            ]

        all_tokens: list[int] = []
        for record in records:
            all_tokens.extend(self._tokenizer.tokenize(record.text))

        if len(all_tokens) <= context_size:
            raise ValueError(
                f"Not enough tokens ({len(all_tokens)}) in split {split!r} for "
                f"context_size={context_size}. Load more rows or reduce context_size."
            )

        self._records = records
        self._sources = available_sources(records)
        self._tokens = torch.tensor(all_tokens, dtype=torch.long)

    @staticmethod
    def _raise_no_rows(
        sources: list[str] | None,
        repo_id: str | None,
        revision: str | None,
        token: str | None,
    ) -> None:
        """Say which sources there are, since a wrong name is the usual reason."""
        every, _ = load_records(repo_id=repo_id, revision=revision, token=token)
        raise ValueError(
            f"No rows matched sources={sources}. "
            f"The dataset has: {', '.join(available_sources(every))}"
        )

    @staticmethod
    def _drop_invalid(records: list[DatasetRecord]) -> list[DatasetRecord]:
        from tinyfacts.check_words import check_words_with_context

        kept: list[DatasetRecord] = []
        for record in records:
            invalid = {
                item.word for item in check_words_with_context(record.text).invalid_words
            }
            if invalid:
                warnings.warn(
                    f"Row '{record.id}' uses {len(invalid)} word(s) outside the list: "
                    f"{sorted(invalid)[:5]}"
                )
                continue
            kept.append(record)
        return kept

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def revision(self) -> str | None:
        """The dataset commit the rows were read from, when it is known."""
        return self._revision

    @property
    def sources(self) -> list[str]:
        """The `source` names the rows actually came from."""
        return list(self._sources)

    @property
    def tokens(self) -> torch.Tensor:
        """The flat 1-D token stream backing this dataset."""
        return self._tokens

    @property
    def n_records(self) -> int:
        return len(self._records)

    @property
    def n_tokens(self) -> int:
        """Number of tokens in the corpus — the honest denominator for an epoch."""
        return len(self._tokens)

    def __len__(self) -> int:
        return (len(self._tokens) - self._context_size + self._stride - 1) // self._stride

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self._stride
        chunk = self._tokens[start : start + self._context_size + 1]
        return chunk[:-1], chunk[1:]
