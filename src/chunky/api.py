from chunky.components.documents import load_document, split_document_into_sentences
from chunky.components.embeddings import load_embedding_model
from chunky.components.features import compute_differences, normalize_differences, smooth_differences
from chunky.components.hmm import get_auto_chunk_hmm_model, inference_chunks


def build_chunks_from_binary_states(
    sentence_sequence: list[str],
    hidden_states_sequence: list[str],
    max_chunk_length: int,
) -> list[str]:
    chunks = []
    current_chunk = ""
    for index, sentence in enumerate(sentence_sequence):
        if index == 0:
            current_chunk = sentence
        else:
            state = hidden_states_sequence[index - 1]

            if state == "new_chunk":
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            elif state == "prior_chunk":
                if len(current_chunk) + len(sentence) + 1 > max_chunk_length:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def auto_chunk(
    document: str,
    embedding_model=None,
    max_chunk_length: int = 1000,
) -> list[str]:
    sentence_sequence = split_document_into_sentences(document)
    embeddings = load_embedding_model(embedding_model)
    vector_sequence = embeddings.embed_sentence_sequence(sentence_sequence)
    diffs = compute_differences(vector_sequence)
    diffs = smooth_differences(diffs)
    diffs = normalize_differences(diffs)
    model = get_auto_chunk_hmm_model(diffs)
    hidden_states_sequence = inference_chunks(model, diffs)
    return build_chunks_from_binary_states(
        sentence_sequence,
        hidden_states_sequence,
        max_chunk_length,
    )


def auto_chunk_from_path(
    path: str,
    embedding_model=None,
    max_chunk_length: int = 1000,
) -> list[str]:
    return auto_chunk(
        load_document(path),
        embedding_model=embedding_model,
        max_chunk_length=max_chunk_length,
    )


def guided_chunk(
    document: str,
    categories: list[str],
    embedding_model=None,
    max_chunk_length: int = 1000,
):
    raise NotImplementedError("guided_chunk is not implemented yet")