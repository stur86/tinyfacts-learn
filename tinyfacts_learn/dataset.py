# dataset.py
"""The training set: rows from the Hub, tokenized into sliding windows."""

import warnings

import torch
from torch.utils.data import Dataset
from tinyfacts.dataset import DatasetRecord, RecordFilter

from .hub_data import available_sources, load_records
from .tokenizers import WordTokenizer


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
    ):
        self._context_size = context_size
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

        all_tokens: list[int] = []
        for record in records:
            all_tokens.extend(self._tokenizer.tokenize(record.text))

        if len(all_tokens) <= context_size:
            raise ValueError(
                f"Not enough tokens ({len(all_tokens)}) for context_size={context_size}. "
                "Load more rows or reduce context_size."
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
    def n_records(self) -> int:
        return len(self._records)

    @property
    def n_tokens(self) -> int:
        return len(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens) - self._context_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self._tokens[idx : idx + self._context_size + 1]
        return chunk[:-1], chunk[1:]
