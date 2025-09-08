import numpy as np
HAS_FAISS = False
try:
    import faiss as _faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    _faiss = None

def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
    return x / n

class NumpyIndexFlatIP:
    def __init__(self, d: int):
        self.d = d
        self.x = np.empty((0, d), dtype="float32")
    def add(self, xb: np.ndarray):
        self.x = _normalize(xb.astype("float32"))
    def search(self, q: np.ndarray, k: int):
        qn = _normalize(q.astype("float32"))
        if self.x.size == 0:
            return np.zeros((qn.shape[0], k), dtype="float32"), -np.ones((qn.shape[0], k), dtype="int64")
        sims = qn @ self.x.T  # косинус через скалярное произведение
        idx = np.argsort(-sims, axis=1)[:, :k]
        dist = np.take_along_axis(sims, idx, axis=1).astype("float32")
        return dist, idx.astype("int64")

def IndexFlatIP(d: int):
    if HAS_FAISS:
        return _faiss.IndexFlatIP(d)
    return NumpyIndexFlatIP(d)

def normalize_L2_inplace(x: np.ndarray):
    if HAS_FAISS:
        _faiss.normalize_L2(x)
    else:
        x[:] = _normalize(x)

