# src/tokenwatt/nnls.py
from __future__ import annotations

import math


def _matvec(A: list[list[float]], x: list[float]) -> list[float]:
    return [sum(row[j] * x[j] for j in range(len(x))) for row in A]


def _matTvec(A: list[list[float]], y: list[float]) -> list[float]:
    n = len(A[0])
    return [sum(A[i][j] * y[i] for i in range(len(A))) for j in range(n)]


def _residual_norm(A: list[list[float]], x: list[float], b: list[float]) -> float:
    ax = _matvec(A, x)
    return math.sqrt(sum((ax[i] - b[i]) ** 2 for i in range(len(b))))


def _solve_passive(A: list[list[float]], b: list[float], passive: list[int]) -> list[float]:
    """Unconstrained least squares over the passive columns via the normal
    equations (AᵀA)s = Aᵀb, solved with Gaussian elimination + partial pivoting.
    Returns a full-length vector with zeros outside `passive`."""
    p = passive
    k = len(p)
    # normal-equation matrix G = A_pᵀ A_p (k×k) and rhs c = A_pᵀ b
    G = [[sum(A[i][p[r]] * A[i][p[c]] for i in range(len(A))) for c in range(k)] for r in range(k)]
    c = [sum(A[i][p[r]] * b[i] for i in range(len(A))) for r in range(k)]
    # Gaussian elimination with partial pivoting
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(G[r][col]))
        if abs(G[piv][col]) < 1e-15:
            continue
        G[col], G[piv] = G[piv], G[col]
        c[col], c[piv] = c[piv], c[col]
        for r in range(col + 1, k):
            f = G[r][col] / G[col][col]
            for cc in range(col, k):
                G[r][cc] -= f * G[col][cc]
            c[r] -= f * c[col]
    s_p = [0.0] * k
    for r in range(k - 1, -1, -1):
        if abs(G[r][r]) < 1e-15:
            continue
        s_p[r] = (c[r] - sum(G[r][cc] * s_p[cc] for cc in range(r + 1, k))) / G[r][r]
    n = len(A[0])
    s = [0.0] * n
    for idx, col in enumerate(p):
        s[col] = s_p[idx]
    return s


def nnls(A: list[list[float]], b: list[float], *, tol: float = 1e-10,
         max_iter: int = 100) -> tuple[list[float], float]:
    """Lawson–Hanson non-negative least squares: minimize ‖A·x − b‖ s.t. x ≥ 0.
    A is m×n (m rows of n floats). Returns (x, residual_norm)."""
    n = len(A[0])
    x = [0.0] * n
    passive: list[int] = []
    active = list(range(n))
    for _ in range(max_iter):
        ax = _matvec(A, x)
        w = _matTvec(A, [b[i] - ax[i] for i in range(len(b))])
        if not active or max(w[j] for j in active) <= tol:
            break
        j = max(active, key=lambda j: w[j])
        active.remove(j)
        passive.append(j)
        while True:
            s = _solve_passive(A, b, passive)
            if all(s[j] > tol for j in passive):
                x = s
                break
            # some passive coefficient went non-positive: back off along x -> s
            ratios = [x[j] / (x[j] - s[j]) for j in passive
                      if s[j] <= tol and abs(x[j] - s[j]) > tol]
            if not ratios:                      # no feasible back-off direction (degenerate)
                x = [max(v, 0.0) for v in s]
                break
            alpha = min(ratios)
            x = [x[j] + alpha * (s[j] - x[j]) for j in range(n)]
            for j in list(passive):
                if x[j] <= tol:
                    passive.remove(j)
                    active.append(j)
            if not passive:
                break
    else:
        raise RuntimeError(f"NNLS did not converge in {max_iter} iterations")
    x = [max(v, 0.0) for v in x]
    return x, _residual_norm(A, x, b)
