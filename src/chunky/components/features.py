import numpy as np

def compute_differences(vector_sequence):
    v = vector_sequence

    if len(v) < 2:
        return np.array([], dtype=float)

    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    v_norm = v / norms

    sims = np.sum(v_norm[1:] * v_norm[:-1], axis=1)
    sims = np.clip(sims, -1.0, 1.0)

    return 1 - sims

def smooth_differences(differences, window_size: int = 5) -> np.ndarray:
    diffs = np.asarray(differences, dtype=float)

    if diffs.size == 0 or window_size <= 1:
        return diffs.copy()

    window_size = min(window_size, diffs.size)
    if window_size % 2 == 0:
        window_size -= 1

    if window_size <= 1:
        return diffs.copy()

    pad_width = window_size // 2
    padded = np.pad(diffs, pad_width, mode="edge")
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode="valid")

def normalize_differences(differences) -> np.ndarray:
    diffs = np.asarray(differences, dtype=float)

    if diffs.size == 0:
        return diffs.copy()

    min_diff = np.min(diffs)
    max_diff = np.max(diffs)

    if np.isclose(max_diff, min_diff):
        return np.zeros_like(diffs)

    return (diffs - min_diff) / (max_diff - min_diff)