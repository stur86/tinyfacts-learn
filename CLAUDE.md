The goal of this repository is to test training some small scale language models (~1M parameters) on a single user GPU using a very limited synthetic dataset and token vocabulary. The vocabulary uses xkcd's "Thing Explainer" 1000 most common words dataset, plus punctuation and special tokens.

A custom tokenizer can be found in `tokenizers.py`. This is what we should use.

The dataset files are .txt files inside the tinyfacts-gen submodule directory, specifically within some of the subfolders. We should allow defining a list of subfolders to grab files from. Tinyfacts-gen also has some utilities to check the validity of a dataset file (meaning that it only contains tokens from the allowed vocabulary). We should use those utilities to check the validity of the dataset files we use for training.

Things I want:

* a custom dataset loader that easily performs the necessary checks and prepares tokens ready for training
* a training framework that allows supporting different models, each defined in its own named folder (e.g. `gpt-small`). Each folder should contain standard files. Once trained, they should be saved as checkpoints with a timestamp.
* a utility to load and try a model by providing a prompt and running a generation for N tokens.
* a first starting model that's a very small GPT-like transformer, with a small context window (e.g. 128 tokens).