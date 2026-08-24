# tests/test_webapp_fixtures.py
"""Keep the web app's tokenizer fixtures in step with the Python tokenizer.

The browser re-implements ``WordTokenizer`` in TypeScript. To make sure the two
cannot drift apart, the vocabulary dump and a set of tokenize/detokenize cases
are committed under ``webapp/test/fixtures`` and checked by the vitest suite.
This module regenerates them and fails if the committed copies are stale.

Run ``python tests/test_webapp_fixtures.py`` to rewrite the fixtures after an
intentional tokenizer change.
"""
import json
from pathlib import Path

import pytest

from tinyfacts_learn.export_onnx import tokenizer_payload
from tinyfacts_learn.tokenizers import WordTokenizer

FIXTURES_DIR = Path(__file__).parent.parent / "webapp" / "test" / "fixtures"
TOKENIZER_FIXTURE = FIXTURES_DIR / "tokenizer.json"
CASES_FIXTURE = FIXTURES_DIR / "cases.json"

# Texts chosen to exercise every branch of WordTokenizer.tokenize: base words,
# inflected forms (which emit a tag token before the base), capitals, digits,
# punctuation, apostrophes, repeated whitespace and out-of-vocabulary words.
CASE_TEXTS = [
    "the sun is a big ball of hot air",
    "The Sun Gives Light",
    "cats and dogs run faster than birds",
    "there are 42 things here",
    "water, air, and fire.",
    "a man's hat",
    "quantum entanglement of xylophones",
    "the  sun\n  is\thot",
    "no words at all: ;!?",
    "",
]


def _cases() -> list[dict]:
    tokenizer = WordTokenizer(ignore_case=True, digits=True)
    cases = []
    for text in CASE_TEXTS:
        ids = tokenizer.tokenize(text)
        cases.append(
            {
                "text": text,
                "ids": ids,
                "tokens": [tokenizer._tokens[i] for i in ids],
                "detokenized": tokenizer.detokenize(ids),
            }
        )
    return cases


def _payload() -> dict:
    return tokenizer_payload(WordTokenizer(ignore_case=True, digits=True))


def write_fixtures() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_FIXTURE.write_text(json.dumps(_payload()))
    CASES_FIXTURE.write_text(json.dumps(_cases(), indent=2) + "\n")


def test_tokenizer_fixture_is_current():
    assert TOKENIZER_FIXTURE.exists(), "run: python tests/test_webapp_fixtures.py"
    assert json.loads(TOKENIZER_FIXTURE.read_text()) == _payload(), (
        "webapp/test/fixtures/tokenizer.json is stale; "
        "run: python tests/test_webapp_fixtures.py"
    )


def test_cases_fixture_is_current():
    assert CASES_FIXTURE.exists(), "run: python tests/test_webapp_fixtures.py"
    assert json.loads(CASES_FIXTURE.read_text()) == _cases(), (
        "webapp/test/fixtures/cases.json is stale; "
        "run: python tests/test_webapp_fixtures.py"
    )


def test_payload_covers_the_whole_vocabulary():
    tokenizer = WordTokenizer(ignore_case=True, digits=True)
    payload = _payload()

    assert payload["vocabSize"] == tokenizer.vocab_size
    assert len(payload["tokens"]) == tokenizer.vocab_size
    assert payload["tokens"][payload["unkId"]] == "<UNK>"
    # ignore_case=True means there is no <UPC> token to emit.
    assert payload["upcId"] is None

    # Every id the word table can produce must be a real vocabulary index, and
    # looking a form up must give what tokenizing that form gives. (Forms with
    # an apostrophe, e.g. "aren't", are unreachable through tokenize(), which
    # splits on the apostrophe — the table still carries them.)
    for form, ids in payload["wordTokens"].items():
        assert ids, form
        assert all(0 <= i < tokenizer.vocab_size for i in ids), form
        if form.isalpha():
            assert ids == tokenizer.tokenize(form), form


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["text"][:24] or "empty")
def test_payload_round_trips_each_case(case):
    """The exported tables alone are enough to reproduce the Python output."""
    payload = _payload()
    tokens = payload["tokens"]

    text = ""
    queued_tag = None
    for tid in case["ids"]:
        token = tokens[tid]
        if token in payload["tags"]:
            queued_tag = token
            continue
        if queued_tag:
            token = payload["formLookup"].get(f"{queued_tag[1:-1]}|{token}", "<UNK>")
            queued_tag = None
        if token in payload["specials"]:
            text = text.strip()
        if token in payload["digits"] and len(text) > 1 and text[-2] in payload["digits"]:
            text = text.strip()
        text += token + " "

    assert text.strip() == case["detokenized"]


if __name__ == "__main__":
    write_fixtures()
    print(f"wrote {TOKENIZER_FIXTURE}")
    print(f"wrote {CASES_FIXTURE}")
