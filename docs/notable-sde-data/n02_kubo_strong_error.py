"""N02 Kubo: the strong error at dt = 0.04 of four schemes against the exact rotation,
on the increments the harness supplies (seed 42, 2000 paths). Fixes `pathwise_tol`:
Stratonovich Heun 3.76%, Euler-Maruyama on the Ito-converted equation 15.1% (right
law, strong order 1/2), the exact-increment rotation 0 (a Lie-group integrator, which
the degeneracy ladder cannot tell from a transcription), the Ito-misread coefficients
69.5%. The gate is 20% so that both correct schemes clear and the misreading fails 3.5x.
`uv run python docs/notable-sde-data/n02_kubo_strong_error.py`
"""
import numpy as np

a = sig = 1.0
T = 1.0
n, nt = 2000, 25
dt = T / nt
x0 = np.array([1.0, 0.0])
J = np.array([[0, -1], [1, 0]])
rng = np.random.default_rng(42)
dW = rng.standard_normal((n, nt)) * np.sqrt(dt)
th = a * T + sig * dW.sum(axis=1)
exact = np.stack([np.cos(th), np.sin(th)], axis=1)


def rot(X):
    return X @ J.T


def heun():
    X = np.tile(x0, (n, 1)).astype(float)
    for k in range(nt):
        inc = a * dt + sig * dW[:, k][:, None]
        Xp = X + rot(X) * inc
        X = X + 0.5 * (rot(X) + rot(Xp)) * inc
    return X


def em(misread):
    X = np.tile(x0, (n, 1)).astype(float)
    for k in range(nt):
        X = X + rot(X) * (a * dt + sig * dW[:, k][:, None]) - (0.0 if misread else 0.5 * sig ** 2 * X * dt)
    return X


def exact_rotation():
    X = np.tile(x0, (n, 1)).astype(float)
    for k in range(nt):
        ph = a * dt + sig * dW[:, k]
        X = np.stack([np.cos(ph) * X[:, 0] - np.sin(ph) * X[:, 1],
                      np.sin(ph) * X[:, 0] + np.cos(ph) * X[:, 1]], axis=1)
    return X


for name, X in (("Heun (Stratonovich)", heun()), ("Ito-converted EM", em(False)),
                ("exact rotation steps", exact_rotation()), ("Ito-misread EM", em(True))):
    rel = np.sqrt(np.mean((X - exact) ** 2)) / np.sqrt(np.mean(exact ** 2))
    print(f"{name:22s} rel strong err {rel:.4f}   E|X|^2 {(X ** 2).sum(axis=1).mean():.4f}")
