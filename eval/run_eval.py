from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from tabulate import tabulate
from chunky.api import auto_chunk

try:
	from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
	from langchain.text_splitter import RecursiveCharacterTextSplitter


Span = tuple[int, int]


def parse_args() -> argparse.Namespace:
	eval_dir = Path(__file__).resolve().parent

	parser = argparse.ArgumentParser(description="Evaluate chunky against a fixed-size baseline.")
	parser.add_argument(
		"--gt-path",
		default=str(eval_dir / "gt.json"),
		help="Path to the ground-truth JSON file.",
	)
	parser.add_argument(
		"--chunk-size",
		type=int,
		default=500,
		help="Baseline chunk size in characters.",
	)
	parser.add_argument(
		"--chunk-overlap",
		type=int,
		default=100,
		help="Baseline chunk overlap in characters.",
	)
	parser.add_argument(
		"--max-chunk-length",
		type=int,
		default=500,
		help="Maximum chunk size passed to chunky auto_chunk.",
	)
	parser.add_argument(
		"--embedding-key-env",
		help="Optional .env or environment variable name for a supported embedding provider key, for example OPENAI_API_KEY.",
	)
	return parser.parse_args()


def get_embedding_model(env_var_name: str):
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

	return openai_embed


def load_ground_truth(gt_path: str) -> list[str]:
	data = json.loads(Path(gt_path).read_text(encoding="utf-8"))
	if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
		raise ValueError("gt.json must contain a JSON list of strings")
	return data


def build_document(gt_chunks: list[str], separator: str = " ") -> tuple[str, list[Span]]:
	spans: list[Span] = []
	document_parts: list[str] = []
	cursor = 0

	for index, chunk in enumerate(gt_chunks):
		if index > 0:
			document_parts.append(separator)
			cursor += len(separator)

		document_parts.append(chunk)
		spans.append((cursor, cursor + len(chunk)))
		cursor += len(chunk)

	return "".join(document_parts), spans


def build_baseline_chunks(document: str, chunk_size: int, chunk_overlap: int) -> tuple[list[str], list[Span]]:
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		length_function=len,
		add_start_index=True,
		strip_whitespace=False,
	)
	docs = splitter.create_documents([document])
	chunks = [doc.page_content for doc in docs]
	spans = [
		(doc.metadata["start_index"], doc.metadata["start_index"] + len(doc.page_content))
		for doc in docs
	]
	return chunks, spans


def build_chunky_chunks(document: str, max_chunk_length: int, embedding_model=None) -> tuple[list[str], list[Span]]:
	chunks = auto_chunk(
		document,
		embedding_model=embedding_model,
		max_chunk_length=max_chunk_length,
	)

	spans: list[Span] = []
	cursor = 0
	for chunk in chunks:
		start = document.find(chunk, cursor)
		if start == -1:
			start = document.find(chunk)
		if start == -1:
			raise ValueError("Unable to align chunky output with the source document")
		end = start + len(chunk)
		spans.append((start, end))
		cursor = end

	return chunks, spans


def span_iou(left: Span, right: Span) -> float:
	intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
	if intersection == 0:
		return 0.0

	union = max(left[1], right[1]) - min(left[0], right[0])
	return intersection / union


def average_best_overlap(reference_spans: list[Span], candidate_spans: list[Span]) -> float:
	if not reference_spans or not candidate_spans:
		return 0.0

	scores = []
	for reference_span in reference_spans:
		scores.append(max(span_iou(reference_span, candidate_span) for candidate_span in candidate_spans))
	return sum(scores) / len(scores)


def evaluate_chunking(reference_spans: list[Span], candidate_spans: list[Span]) -> dict[str, float]:
	coverage = average_best_overlap(reference_spans, candidate_spans)
	purity = average_best_overlap(candidate_spans, reference_spans)
	score = (coverage + purity) / 2
	return {
		"coverage": coverage * 100,
		"purity": purity * 100,
		"score": score * 100,
	}


def main() -> int:
	args = parse_args()

	gt_chunks = load_ground_truth(args.gt_path)
	document, gt_spans = build_document(gt_chunks)

	embedding_model = None
	if args.embedding_key_env:
		embedding_model = get_embedding_model(args.embedding_key_env)

	baseline_chunks, baseline_spans = build_baseline_chunks(
		document,
		chunk_size=args.chunk_size,
		chunk_overlap=args.chunk_overlap,
	)
	chunky_chunks, chunky_spans = build_chunky_chunks(
		document,
		max_chunk_length=args.max_chunk_length,
		embedding_model=embedding_model,
	)

	baseline_scores = evaluate_chunking(gt_spans, baseline_spans)
	chunky_scores = evaluate_chunking(gt_spans, chunky_spans)

	rows = [
		{
			"method": "baseline",
			"chunks": len(baseline_chunks),
			"coverage": baseline_scores["coverage"],
			"purity": baseline_scores["purity"],
			"score": baseline_scores["score"],
		},
		{
			"method": "chunky",
			"chunks": len(chunky_chunks),
			"coverage": chunky_scores["coverage"],
			"purity": chunky_scores["purity"],
			"score": chunky_scores["score"],
		},
		{
			"method": "delta",
			"chunks": len(chunky_chunks) - len(baseline_chunks),
			"coverage": chunky_scores["coverage"] - baseline_scores["coverage"],
			"purity": chunky_scores["purity"] - baseline_scores["purity"],
			"score": chunky_scores["score"] - baseline_scores["score"],
		},
	]

	print(f"Ground-truth chunks: {len(gt_chunks)}")
	print(f"Document length: {len(document)} characters")
	print(tabulate(rows, headers="keys", tablefmt="github", floatfmt=".2f"))

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
