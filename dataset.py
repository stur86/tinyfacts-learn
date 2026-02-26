# dataset.py
import warnings
from pathlib import Path

import torch
from torch.utils.data import Dataset

from tinyfacts.check_words import check_words, split_words
from tokenizers import WordTokenizer

# Root of the tinyfacts-gen submodule, relative to this file
TINYFACTS_GEN_DIR = Path(__file__).parent / "tinyfacts-gen"


class TinyfactsDataset(Dataset):
    """Dataset of tokenized tinyfacts .txt files from named subfolders.

    Args:
        subfolders: List of subfolder names inside tinyfacts-gen/ to load .txt files from.
        context_size: Number of tokens per training window.
        tokenizer: Optional WordTokenizer instance; a default one is created if None.
        skip_invalid: If True, invalid files are skipped with a warning. If False (default),
                      raises ValueError on the first invalid file found.
    """

    def __init__(
        self,
        subfolders: list[str],
        context_size: int = 128,
        tokenizer: WordTokenizer | None = None,
        skip_invalid: bool = True,
    ):
        self._context_size = context_size
        self._tokenizer = tokenizer or WordTokenizer(ignore_case=True, digits=True)

        all_tokens: list[int] = []

        for subfolder_name in subfolders:
            subfolder = TINYFACTS_GEN_DIR / subfolder_name
            if not subfolder.exists():
                raise ValueError(
                    f"Subfolder not found: {subfolder} "
                    f"(looked inside {TINYFACTS_GEN_DIR})"
                )

            txt_files = sorted(subfolder.glob("*.txt"))
            for txt_file in txt_files:
                text = txt_file.read_text(encoding="utf-8")
                invalid = check_words(split_words(text))
                if invalid:
                    msg = (
                        f"File '{txt_file}' contains {len(invalid)} invalid word(s): "
                        f"{list(invalid.keys())[:5]}{'...' if len(invalid) > 5 else ''}"
                    )
                    if skip_invalid:
                        warnings.warn(msg)
                        continue
                    else:
                        raise ValueError(msg)

                tokens = self._tokenizer.tokenize(text)
                all_tokens.extend(tokens)

        if len(all_tokens) <= context_size:
            raise ValueError(
                f"Not enough tokens ({len(all_tokens)}) for context_size={context_size}. "
                "Load more data or reduce context_size."
            )

        self._tokens = torch.tensor(all_tokens, dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer._tokens)

    def __len__(self) -> int:
        return len(self._tokens) - self._context_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self._tokens[idx : idx + self._context_size + 1]
        return chunk[:-1], chunk[1:]
