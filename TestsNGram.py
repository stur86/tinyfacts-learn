import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    import marimo as mo
    import numpy as np
    from numpy.typing import NDArray
    return NDArray, np, re


@app.cell
def _(re):
    from tinyfacts_learn.hub_data import load_records

    # The texts come from the dataset on the Hugging Face Hub. Name sources in
    # the list below to narrow it down; leave it empty to use every row.
    sources = ["gpt-5_1", "manually", "claude_sonnet_4_5"]

    records, revision = load_records()
    if sources:
        records = [r for r in records if r.source in sources]
    data = [r.text for r in records]
    all_text = '\n\n'.join(data)

    print(f"Loaded {len(records)} rows at revision {revision}")
    print(f"{len(all_text)} characters")
    print(f"Approximately {len(all_text.split())} words")

    # Normalization methods
    def only_literals(text: str):
        text = re.sub("[\-\s‑—]", " ", text.lower())
        text = re.sub("[^a-z ]", "", text)
        return text

    only_literals_text = only_literals(all_text)
    return (only_literals_text,)


@app.cell
def _(NDArray, np, re):
    class Bigram:

        _UNK_TOKEN = "<|UNK|>"
    
        def __init__(self, text: str, as_words: bool = False):
            self._define_tokens(text, as_words)
            self._compute_frequencies(self.tokenize(text))

        def _split_az_words(self, text: str) -> str:
            return re.findall(r"([a-z]+|^\s)", text)

        def _define_tokens(self, text: str, words: bool = False):
            self.has_word_tokens = words
            if not words:
                # Each character is a token
                self.tokens = sorted(list(set(text)))
            else:
                # Each consecutive set of (lowercase) letters is a token
                self.tokens = sorted(list(set(self._split_az_words(text))))
            self.tokens += [self._UNK_TOKEN]
            self.tokens = np.array(self.tokens)
            self.tokens.sort()
            self.n = len(self.tokens)

        def tokenize(self, text: str) -> NDArray[np.int32]:
            tokens: list[int] = []
            if self.has_word_tokens:
                split_text = self._split_az_words(text)
            else:
                split_text = list(text)
            tokens = np.searchsorted(self.tokens, split_text, side='left')
            # Now double check
            unk_i = list(self.tokens).index(self._UNK_TOKEN)
            tokens = np.where(self.tokens[tokens] == split_text, tokens, unk_i)
            return tokens

        def decode(self, tokens: list[int]) -> str:
            joiner = ' ' if self.has_word_tokens else ''
            return joiner.join(self.tokens[tokens])

        def _compute_frequencies(self, tokens: NDArray[np.int32]):
            n = self.n
            M = np.ones(n*n)
            pair_idx = tokens[:-1]*n + tokens[1:]
            cres_M = np.unique_counts(pair_idx)
            M[cres_M.values] += cres_M.counts
            M = M.reshape((n,n))
            M /= np.sum(M, axis=1, keepdims=True)
            self.logM = np.log(M)
            self.logM -= np.amax(self.logM, axis=1, keepdims=True)

            p = np.ones(n)
            cres_p = np.unique_counts(tokens)
            p[cres_p.values] += cres_p.counts
            p /= np.sum(p)
            self.logp = np.log(p)
                
        def generate(self, input: str = "", n_gen: int = 100, T: float = 1.0) -> str:
            M = np.exp(self.logM/T)
            M /= np.sum(M, axis=1, keepdims=True)

            output_tokens = list(self.tokenize(input))

            tkrange = np.arange(self.n)
            if input == "":
                # Pick a starting token at random
                p = np.exp(self.logp/T)
                p /= np.sum(p)
                tk = np.random.choice(tkrange, p=p)
                output_tokens += [tk]

            for i in range(n_gen):
                tk = output_tokens[-1]
                tk_new = np.random.choice(tkrange, p=M[tk])
                output_tokens += [tk_new]

            return self.decode(output_tokens)
    return (Bigram,)


@app.cell
def _(Bigram, only_literals_text):
    bg = Bigram(only_literals_text, True)
    return (bg,)


@app.cell
def _(bg):
    bg.generate('a', T=0.4)
    return


@app.cell
def _(only_literals_text):
    pairs = zip(only_literals_text[:-1], only_literals_text[1:])
    return


@app.cell
def _(bg):
    bg.tokens
    return


@app.cell
def _(bg, np):
    np.searchsorted(bg.tokens, ['a', 'b', 'z', '&'], side='left')
    return


@app.cell
def _(bg):
    bg.tokens
    return


@app.cell
def _(bg, np):
    embed_dim = 256

    _svd = np.linalg.svd(bg.logM)
    input_embeds = (_svd.U*_svd.S[None,:])[:,:embed_dim]
    output_embeds = _svd.Vh[:embed_dim].T
    print(f"Truncated dimension: {embed_dim} - Norm of difference: {np.sum((input_embeds@output_embeds.T - bg.logM)**2)}")
    return embed_dim, input_embeds, output_embeds


@app.cell
def _(bg, embed_dim, input_embeds, np, only_literals_text, output_embeds):
    from scipy.signal import lfilter
    # Now fit
    _tokens = bg.tokenize(only_literals_text)
    alphas = np.array(np.logspace(-2, 0, 4))

    _X_base = input_embeds[_tokens[:-1]]
    _y = output_embeds[_tokens[1:]]
    _X = []
    for _alpha in alphas:
        # The recursive EMA formula parameters:
        _b = [_alpha]            # Numerator coefficients
        _a = [1, -(1 - _alpha)]  # Denominator coefficients

        # Apply EMA
        _X_ema = lfilter(_b, _a, _X_base, axis=0)
        _X.append(_X_ema)

    _X = np.concatenate(_X, axis=1)

    _A = np.linalg.inv(_X.T@_X)@_X.T
    LinFitM = np.zeros((_X.shape[1], embed_dim))
    for _i in range(embed_dim):
        LinFitM[:,_i] = _A@_y[:,_i]
    return LinFitM, alphas


@app.cell
def _(LinFitM, alphas, bg, embed_dim, input_embeds, np, output_embeds):
    # Start with any token
    _gen_T = 1e-2
    _x = np.zeros(LinFitM.shape[0])
    _prompt = "a long time ago "
    tk_seq = list(bg.tokenize(_prompt))
    _tkrange = np.arange(bg.n)
    _n_gen = 40
    _alphas_grid = np.repeat(alphas, repeats=embed_dim)
    for _ in range(_n_gen):
        _tk = tk_seq[-1]
        _x = _alphas_grid*np.tile(input_embeds[_tk], reps=len(alphas))+(1-_alphas_grid)*_x
        # New token?
        _y = _x@LinFitM
        _logits = output_embeds@_y
        _logits -= np.max(_logits)
        _p = np.exp(_logits/_gen_T)
        _p = np.where(np.isnan(_p), 0.0, _p)
        _p /= np.sum(_p)
        _tk_new = np.random.choice(_tkrange, p=_p)
        tk_seq.append(_tk_new)
    
    print(f"Bigram generation:\n{bg.generate(_prompt, n_gen=_n_gen, T=_gen_T)}\n***")
    print(f"LinFit generation:\n{bg.decode(tk_seq)}\n***")
    return


@app.cell
def _(LinFitM, bg, input_embeds, output_embeds):
    print(bg.logM.size)
    print(input_embeds.size+output_embeds.size+LinFitM.size)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
