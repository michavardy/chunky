from collections.abc import Callable

import numpy as np


EmbeddingModel = Callable[[list[str]], np.ndarray]


def load_sentence_transformer_embedding_model(
    model_name: str = "all-MiniLM-L6-v2",
) -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name).encode


def load_embedding_model(model: EmbeddingModel | None = None) -> "Embeddings":
    return Embeddings(model=model)


class Embeddings:
    def __init__(self, model: EmbeddingModel | None = None):
        if model is None:
            model = load_sentence_transformer_embedding_model()

        if not callable(model):
            raise ValueError("Embedding model must be callable: List[str] -> np.ndarray")

        self.model = model

    def embed_sentence(self, sentence: str) -> np.ndarray:
        return self.model([sentence])[0]

    def embed_sentence_sequence(self, sentences: list[str]) -> np.ndarray:
        return np.array(self.model(sentences))