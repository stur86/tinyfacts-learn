# dataset.py
import hashlib
import warnings
from pathlib import Path

import torch
from torch.utils.data import Dataset

from tinyfacts.check_words import check_words_with_context
from .tokenizers import WordTokenizer

# Root of the tinyfacts-gen submodule, relative to this file
TINYFACTS_GEN_DIR = Path(__file__).parent.parent / "tinyfacts-gen"


def _file_split(rel_path: str, val_fraction: float, split_seed: int) -> str:
    """Deterministically assign a file to the train or val split.

    Splitting at the *file* level (rather than slicing the concatenated token
    stream) keeps both splits representative of every subfolder and guarantees
    no sliding window ever straddles the boundary, so val windows share no
    tokens with train windows.
    """
    digest = hashlib.sha256(f"{split_seed}:{rel_path}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if fraction < val_fraction else "train"


class TinyfactsDataset(Dataset):
    """Dataset of tokenized tinyfacts .txt files from named subfolders.

    Args:
        subfolders: List of subfolder names inside tinyfacts-gen/ to load .txt files from.
        context_size: Number of tokens per training window.
        tokenizer: Optional WordTokenizer instance; a default one is created if None.
        skip_invalid: If True, invalid files are skipped with a warning. If False (default),
                      raises ValueError on the first invalid file found.
        stride: Step between consecutive window start positions. The default of 1 yields
                maximally overlapping windows, which means one "epoch" over this dataset
                is really ``context_size`` passes over every token. Set
                ``stride=context_size`` for non-overlapping windows and an epoch count
                that means one pass over the corpus.
        split: "all" (default), "train", or "val" — which side of the file-level split
               to load.
        val_fraction: Fraction of *files* assigned to the val split. Ignored when
                      ``split="all"``.
        split_seed: Seed for the deterministic file-level split assignment.
    """

    def __init__(
        self,
        subfolders: list[str],
        context_size: int = 128,
        tokenizer: WordTokenizer | None = None,
        skip_invalid: bool = False,
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

        all_tokens: list[int] = []
        self._n_files = 0

        for subfolder_name in subfolders:
            subfolder = TINYFACTS_GEN_DIR / subfolder_name
            if not subfolder.exists():
                raise ValueError(
                    f"Subfolder not found: {subfolder} "
                    f"(looked inside {TINYFACTS_GEN_DIR})"
                )

            txt_files = sorted(subfolder.glob("*.txt"))
            for txt_file in txt_files:
                if split != "all":
                    rel = txt_file.relative_to(TINYFACTS_GEN_DIR).as_posix()
                    if _file_split(rel, val_fraction, split_seed) != split:
                        continue

                text = txt_file.read_text(encoding="utf-8")
                result = check_words_with_context(text)
                invalid = {item.word for item in result.invalid_words}
                if invalid:
                    msg = (
                        f"File '{txt_file}' contains {len(invalid)} invalid word(s): "
                        f"{list(invalid)[:5]}{'...' if len(invalid) > 5 else ''}"
                    )
                    if skip_invalid:
                        warnings.warn(msg)
                        continue
                    else:
                        raise ValueError(msg)

                tokens = self._tokenizer.tokenize(text)
                all_tokens.extend(tokens)
                self._n_files += 1

        if len(all_tokens) <= context_size:
            raise ValueError(
                f"Not enough tokens ({len(all_tokens)}) in split {split!r} for "
                f"context_size={context_size}. Load more data or reduce context_size."
            )

        self._tokens = torch.tensor(all_tokens, dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def tokens(self) -> torch.Tensor:
        """The flat 1-D token stream backing this dataset."""
        return self._tokens

    @property
    def n_tokens(self) -> int:
        """Number of tokens in the corpus — the honest denominator for an epoch."""
        return len(self._tokens)

    @property
    def n_files(self) -> int:
        """Number of .txt files that made it into this split."""
        return self._n_files

    def __len__(self) -> int:
        return (len(self._tokens) - self._context_size + self._stride - 1) // self._stride

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self._stride
        chunk = self._tokens[start : start + self._context_size + 1]
        return chunk[:-1], chunk[1:]
