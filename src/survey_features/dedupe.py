"""Country-blind work is computed once per question, not once per cell.

The ``unprompted`` condition never names the country: ``freetext_messages``
passes ``country=None``, extraction reads only the essay, and mapping retrieves
against the survey-wide variable pool with a target-keyed exclusion set. So for
that condition the whole generate -> extract -> map chain is a pure function of
``(survey, target)`` — a question with three countries was paying for three
identical pipelines, and the country only ever entered at scoring.

Worse, it was not getting three identical answers. Generation runs at
temperature 0, but hosted providers are not deterministic at temperature 0, so
the copies came back as different essays (Afrobarometer Q18 on Kimi-K2.6: two
byte-identical, the third a wholly different essay from the same minute). That
put an unregistered generation draw inside the country-blind arm, confounded
with country — i.e. inside the baseline half of every country contrast.

``SharedByQuestion`` collapses the duplicates: one computation per question,
reused across its countries. Per-cell artifacts are still written per country,
so nothing downstream of the pipeline changes.
"""
from __future__ import annotations

import threading
from typing import Callable, Hashable, Iterable, TypeVar

T = TypeVar("T")


def question_siblings(
    cells: Iterable[tuple[str, str, str]],
) -> dict[tuple[str, str], list[str]]:
    """``(survey, target)`` -> that question's countries, in grid order."""
    siblings: dict[tuple[str, str], list[str]] = {}
    for survey, target, country in cells:
        countries = siblings.setdefault((survey, target), [])
        if country not in countries:
            countries.append(country)
    return siblings


class SharedByQuestion:
    """Compute a country-blind value once per key, across worker threads.

    ``get`` holds a per-key lock while the first caller computes, so sibling
    cells in flight at the same time wait and reuse rather than duplicating the
    call. ``reuse`` is consulted first — it recovers a value an earlier run left
    on disk — and returns ``None`` to mean "nothing usable, compute it".
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[Hashable, threading.Lock] = {}
        self._values: dict[Hashable, object] = {}
        self.computed = 0
        self.shared = 0

    def get(
        self,
        key: Hashable,
        compute: Callable[[], T],
        reuse: Callable[[], T | None] | None = None,
    ) -> T:
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            if key in self._values:
                self._bump("shared")
                return self._values[key]  # type: ignore[return-value]
            value = reuse() if reuse is not None else None
            if value is None:
                value = compute()
                self._bump("computed")
            else:
                self._bump("shared")
            self._values[key] = value
            return value

    def _bump(self, which: str) -> None:
        with self._guard:
            setattr(self, which, getattr(self, which) + 1)

    def summary(self) -> str:
        return f"computed={self.computed} shared={self.shared}"
