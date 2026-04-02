from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from chunky.api import auto_chunk
from chunky.components.documents import load_document, load_pdf_document, load_url_document
from chunky.components.embeddings import Embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk a document with chunky.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--file_path", help="Path to a plain text document.")
    source_group.add_argument("--url", help="URL to load and chunk.")
    source_group.add_argument("--pdf", help="Path to a PDF document.")
    parser.add_argument(
        "--max-chunk-length",
        type=int,
        default=1000,
        help="Maximum chunk size in characters.",
    )
    parser.add_argument(
        "--output",
        default=str(Path.cwd()),
        help="Output directory for the JSON file. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--embedding-key-env",
        help="Optional .env variable name for a supported embedding provider key, for example OPENAI_API_KEY.",
    )
    return parser.parse_args()


def get_embedding_model(env_var_name: str) -> Embeddings:
    if env_var_name != "OPENAI_API_KEY":
        raise ValueError(
            f"Unsupported embedding key env var: {env_var_name}. Supported values: OPENAI_API_KEY"
        )

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    api_key = os.getenv(env_var_name)
    if not api_key:
        raise ValueError(f"Missing {env_var_name} in .env or environment")

    client = OpenAI(api_key=api_key)

    def openai_embed(sentences: list[str]) -> np.ndarray:
        batch_size = 100
        all_embeddings = []

        for index in range(0, len(sentences), batch_size):
            batch = sentences[index:index + batch_size]
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
            )
            all_embeddings.extend(item.embedding for item in response.data)

        return np.array(all_embeddings)

    return Embeddings(model=openai_embed)


def load_source_document(args: argparse.Namespace) -> tuple[str, str]:
    if args.file_path:
        path = Path(args.file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        return str(path), load_document(path)

    if args.pdf:
        path = Path(args.pdf).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing PDF: {path}")
        return str(path), load_pdf_document(path)

    return args.url, load_url_document(args.url)


def get_output_stem(args: argparse.Namespace) -> str:
    if args.file_path:
        return Path(args.file_path).stem

    if args.pdf:
        return Path(args.pdf).stem

    parsed = urlparse(args.url)
    candidate = Path(parsed.path).stem or parsed.netloc or "document"
    sanitized = "".join(character if character.isalnum() or character in "-_" else "_" for character in candidate)
    return sanitized or "document"


def write_chunks(output_dir: Path, output_stem: str, chunks: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_stem}.json"
    output_path.write_text(
        json.dumps({"chunks": chunks}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    args = parse_args()

    try:
        source, document = load_source_document(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        embedding_model = None
        if args.embedding_key_env:
            embedding_model = get_embedding_model(args.embedding_key_env).model
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    chunks = auto_chunk(
        document,
        embedding_model=embedding_model,
        max_chunk_length=args.max_chunk_length,
    )
    output_dir = Path(args.output).expanduser().resolve()

    try:
        output_path = write_chunks(output_dir, get_output_stem(args), chunks)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"{source}: {len(chunks)} chunk(s)")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())