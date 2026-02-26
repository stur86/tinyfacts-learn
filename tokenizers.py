import re
from pathlib import Path
from tinyfacts.word_forms import WordFormsDictionary


class LetterTokenizer:
    def __init__(
        self,
        digits: bool = True,
        allowed_special: str = ".,:;!?'-\"",
        ignore_case: bool = True,
    ):
        self._digits = digits
        self._allowed_special = allowed_special
        self._ignore_case = ignore_case
        # Build a full list of tokens:
        # - all lowercase letters
        # - digits if specified
        # - allowed special characters
        # - space
        # - special tokens <UPC> (if case is not ignored) and <UNK>
        self._tokens = [chr(i) for i in range(ord("a"), ord("z") + 1)]
        if self._digits:
            self._tokens.extend([str(i) for i in range(10)])
        self._tokens.extend(list(self._allowed_special) + [" "])
        if not ignore_case:
            self._tokens.append("<UPC>")
        self._tokens.append("<UNK>")
        self._token_to_id = {tok: idx for idx, tok in enumerate(self._tokens)}

    def tokenize(self, text: str) -> list[int]:
        tokens = []
        for char in text:
            # Check if it's any space character
            if char.isspace():
                tokens.append(self._token_to_id[" "])
                continue
            # Check if it's an uppercase letter
            if re.match(r"[A-Z]", char):
                if not self._ignore_case:
                    tokens.append(self._token_to_id["<UPC>"])
                char = char.lower()
            if char in self._token_to_id:
                tokens.append(self._token_to_id[char])
            else:
                tokens.append(self._token_to_id["<UNK>"])
        return tokens

    def detokenize(self, token_ids: list[int]) -> str:
        id_to_token = {idx: tok for tok, idx in self._token_to_id.items()}
        chars = []
        skip_next_upper = False
        for tid in token_ids:
            token = id_to_token.get(tid, "<UNK>")
            if token == "<UPC>":
                skip_next_upper = True
                continue
            if skip_next_upper:
                chars.append(token.upper())
                skip_next_upper = False
            else:
                chars.append(token)
        return "".join(chars)


class WordTokenizer:
    
    _digits: list[str]
    _allowed_special: str
    _ignore_case: bool
    _word_forms_dict: WordFormsDictionary
    _words: list[str]
    _special: list[str]

    def __init__(
        self,
        digits: bool = False,
        allowed_special: str = ".,:;!?'-\"",
        ignore_case: bool = True,
    ):
        self._allowed_special = allowed_special
        self._ignore_case = ignore_case
        self._word_forms_dict = WordFormsDictionary()

        # Build a list of tokens. First, gather all words and all tags from the word forms dictionary
        words = set([])
        tags = set([])
        for tagged_word in self._word_forms_dict._word_map.values():
            words.add(tagged_word.base)
            if tagged_word.tag:
                tags.add(f"<{tagged_word.tag}>")

        self._words = list(sorted(list(words)))
        self._digits = list([] if not digits else [str(i) for i in range(10)])
        self._special = list(self._allowed_special)
        self._tags = list(sorted(list(tags)))
        self._extras = ["<UNK>"]
        if not ignore_case:
            self._extras.insert(0, "<UPC>")
        self._tokens: list[str] = (
            self._words + self._digits + self._special + self._tags + self._extras
        )
        self._token_to_id = {tok: idx for idx, tok in enumerate(self._tokens)}

        # Build a reverse map from base+tag to token id for quick lookup
        self._base_tag_to_id = {}
        for word, tagged_word in self._word_forms_dict._word_map.items():
            self._base_tag_to_id[(tagged_word.base, tagged_word.tag)] = word

    @property
    def vocab_size(self) -> int:
        return len(self._tokens)

    def tokenize(self, text: str) -> list[int]:
        # Split the text into words based on whitespace
        token_ids = []
        word_re = re.compile(r"^[a-zA-Z][a-z]*")
        space_re = re.compile(r"^\s+")
        while text:
            # Is there a word?
            match = word_re.search(text)
            space_match = space_re.search(text)
            if match:
                raw_word = match.group(0)
                text = text[match.end():]
                print(raw_word, " | ", text)
                if re.match(r"[A-Z]", raw_word[0]):
                    if not self._ignore_case:
                        token_ids.append(self._tokens.index("<UPC>"))
                    raw_word = raw_word.lower()
                if raw_word in self._word_forms_dict._word_map:
                    print("Known word:", raw_word)
                    tokens = self._word_forms_dict.get_tokens(raw_word)
                    print("Tokens:", tokens)
                    for token in tokens:
                        token_ids.append(self._tokens.index(token))
                else:
                    token_ids.append(self._token_to_id["<UNK>"])
            elif space_match:
                # Skip whitespace
                text = text[space_match.end():]
                continue
            else:
                # Digit or special character?
                char = text[0]
                text = text[1:]
                if re.match(r"\s", char):
                    # Spaces are ignored
                    continue
                if char in self._token_to_id:
                    token_ids.append(self._token_to_id[char])
                else:
                    token_ids.append(self._token_to_id["<UNK>"])
        return token_ids

    def detokenize(self, token_ids: list[int]) -> str:
        text = ""
        queued_tag: str | None = None
        queued_upper = False
        for tid in token_ids:
            token = self._tokens[tid]
            if token in self._tags:
                queued_tag = token
                continue
            if token == "<UPC>":
                queued_upper = True
                continue

            if queued_tag:
                base = token
                tag = queued_tag[1:-1]  # Remove angle brackets
                word = self._base_tag_to_id.get((base, tag), "<UNK>")
                token = word
                queued_tag = None

            if queued_upper:
                token = token.upper()
                queued_upper = False

            if token in self._allowed_special:
                text = text.strip()  # Remove space before punctuation
            
            if token in self._digits and len(text) > 1 and text[-2] in self._digits:
                # No space between digits
                text = text.strip()
            
            text += token + " "
        return text.strip()


if __name__ == "__main__":
    import sys

    content = Path(sys.argv[1]).read_text().strip()

    # tokenizer = LetterTokenizer(ignore_case=False)
    # token_ids = tokenizer.tokenize(content)
    # print("Token IDs:", token_ids)
    # print("Tokens:", [tokenizer._tokens[tid] for tid in token_ids])
    # print("Detokenized:", tokenizer.detokenize(token_ids))

    wtk = WordTokenizer(ignore_case=True, digits=True)

    print("\nWord Tokenizer:")
    word_token_ids = wtk.tokenize(content)
    print("Token IDs:", word_token_ids)
    print("Tokens:", [wtk._tokens[tid] for tid in word_token_ids])
    print("Detokenized:", wtk.detokenize(word_token_ids))
