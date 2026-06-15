from __future__ import annotations

import math
import time
from typing import Optional, Tuple


class OneEuroFilter:
    """One-Euro filter for jitter-free, responsive smoothing."""

    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 0.2,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def _alpha(self, t_e: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        if t is None:
            t = time.time()
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x
            return x

        t_e = max(t - self._t_prev, 1e-6)

        a_d = self._alpha(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(t_e, cutoff)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


def _distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _get_point(lm, idx: int, w: int, h: int) -> Tuple[int, int]:
    return (int(lm[idx].x * w), int(lm[idx].y * h))


def compute_eye_aspect_ratio(
    lm, w: int, h: int, top: int, bottom: int, left: int, right: int
) -> float:
    """Compute Eye Aspect Ratio (EAR) for a single eye."""
    t = _get_point(lm, top, w, h)
    b = _get_point(lm, bottom, w, h)
    l = _get_point(lm, left, w, h)
    r = _get_point(lm, right, w, h)
    v = _distance(t, b)
    h2 = _distance(l, r)
    return v / h2 if h2 != 0 else 1.0


def normalize(text: str) -> str:
    text = text.lower()

    corrections = {
        "krom": "chrome",
        "crome": "chrome",
        "serch": "search",
        "youtub": "youtube",
    }

    for source, target in corrections.items():
        text = text.replace(source, target)

    return text.strip()
