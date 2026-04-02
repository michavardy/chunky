# chunky

HMM-based document chunking for boundary-aware text segmentation.

`chunky` detects semantic shifts across a document and groups adjacent sentences into coherent chunks.

## Installation

Core package:

```bash
pip install git+https://github.com/michavardy/chunky.git
```

With OpenAI embeddings and evaluation tooling:

```bash
pip install "chunky[openai,eval] @ git+https://github.com/michavardy/chunky.git"
```

`chunky` installs `hmmx` as a dependency.

## Usage

```python
from chunky import auto_chunk

document = "Sentence one. Sentence two. Sentence three."

chunks = auto_chunk(document, max_chunk_length=500)
```

## Example

```python
from chunky import auto_chunk, auto_chunk_from_path

chunks = auto_chunk("Sentence one. Sentence two.")
file_chunks = auto_chunk_from_path("./notes.txt", max_chunk_length=800)

for index, chunk in enumerate(file_chunks, start=1):
    ... # rag pipeline
```


## CLI

```bash
chunky --file_path "C:/path/to/document.txt" --output "C:/path/to/output"
chunky --pdf "C:/path/to/document.pdf" --output "C:/path/to/output"
chunky --url "https://example.com/article" --output "C:/path/to/output"
```

## Results

`chunky` includes a simple evaluation script for comparing HMM-based chunking against a fixed-size baseline.

```text
Ground-truth chunks: 20
Document length: 5409 characters
| method   |   chunks |   coverage |   purity |   score |
|----------|----------|------------|----------|---------|
| baseline |       14 |      50.79 |    56.41 |   53.60 |
| chunky   |       31 |      49.93 |    47.63 |   48.78 |
| delta    |       17 |      -0.86 |    -8.78 |   -4.82 |
```

This benchmark is intended as a diagnostic reference, not a claim of universal performance.

## Design

`chunky` is organized as a small pipeline: split the document into sentences, embed each sentence, compute local embedding differences, smooth and normalize those differences, then fit an HMM over the resulting sequence to infer chunk boundaries.

The design is modular. Document loading, embedding generation, feature extraction, and HMM inference are kept separate so individual components can be swapped without changing the public API.
