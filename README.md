# chunky

a lightweight Python package for document splitting automation into symantically meaningful chunks using Hidden Markov Models (HMMs).

It detects shifts in subject or category across a document and groups sentences into coherent segments.

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/michavardy/chunky.git
```

For OpenAI-powered embeddings and eval tooling:

```bash
pip install "chunky[openai,eval] @ git+https://github.com/michavardy/chunky.git"
```

`chunky` depends on `hmmx`, which is also installable directly from GitHub.
---

## Usage

### Module layout

- `chunky.documents`: document loading and sentence splitting
- `chunky.embeddings`: embedding wrappers and model loaders
- `chunky.api`: external-facing chunking functions

The package root still re-exports the main public functions for compatibility.

### Auto Chunking

Let the model infer categories automatically:

```python
from chunky import auto_chunk

for chunk in auto_chunk(document):
    ... # further document processing
```

### CLI

After installation, the CLI is available as:

```bash
chunky --pdf "C:/path/to/document.pdf" --output "C:/path/to/output"
```

### Guided Chunking

Provide your own categories:

```python
from chunky.api import guided_chunk

document: str = doc1
categories: list[str] = ["cat1", "cat2", "cat3"]

for category, chunk in guided_chunk(document, categories):
    ... # further document processing
```

`guided_chunk` is not implemented yet.
---

## Additional arguments

- **embedding_model** Custom embedding model (defaults to a built-in open-source model)
- **max_chunk_length** Maximum chunk size (in characters)
- **overlap**: Overlap between consecutive chunks (in characters)


## Eval

```bash
Ground-truth chunks: 20
Document length: 5409 characters
| method   |   chunks |   coverage |   purity |   score |
|----------|----------|------------|----------|---------|      
| baseline |       14 |      50.79 |    56.41 |   53.60 |      
| chunky   |       31 |      49.93 |    47.63 |   48.78 |      
| delta    |       17 |      -0.86 |    -8.78 |   -4.82 | 
```
