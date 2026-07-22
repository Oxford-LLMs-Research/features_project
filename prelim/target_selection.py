"""Select prelim targets (topic + response cardinality spread) per survey."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from survey_features.surveys import MISSING_LABEL_PATTERNS as _MISSING_LABEL_PATTERNS


def _label_missing(label: str) -> bool:
    ln = str(label).strip().lower()
    return any(p in ln for p in _MISSING_LABEL_PATTERNS)


def find_var_location(metadata: dict, var_code: str) -> tuple[str | None, dict | None]:
    for sec, block in metadata.items():
        if sec == "EXCLUDED" or not isinstance(block, dict):
            continue
        if var_code in block and isinstance(block[var_code], dict):
            return sec, block[var_code]
    return None, None


def substantive_category_count_metadata(info: dict) -> int | None:
    vals = info.get("values") or {}
    if not isinstance(vals, dict) or not vals:
        return None
    n = sum(1 for lab in vals.values() if not _label_missing(lab))
    return n if n > 0 else None


def substantive_category_count_empirical(data: pd.DataFrame, col: str, max_levels: int = 80) -> int:
    s = pd.to_numeric(data[col], errors="coerce")
    if s.notna().mean() >= 0.6:
        u = pd.Series(s.dropna().unique())
    else:
        u = pd.Series(data[col].dropna().astype(str).unique())
    u = u.dropna()
    if len(u) > max_levels:
        return max_levels + 99
    return int(len(u))


def topic_key(section: str, info: dict) -> str:
    for k in (
        "topic_tag",
        "topics",
        "topic",
        "tags",
        "tag",
        "section_tag",
        "themes",
        "theme",
        "domains",
    ):
        v = info.get(k)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            inner = ",".join(str(x) for x in v[:8])
            return f"{k}:{inner}"
        return f"{k}:{v}"
    return f"section:{section}"


def effective_count(n_meta: int | None, n_emp: int) -> int:
    return int(n_meta) if n_meta is not None else int(n_emp)


# Maximum effective category count tolerated for the "large" bucket. Targets
# above this (e.g. year-of-birth at ~80 cats) behave like regression on a
# classifier and dominate oracle wall time; we exclude them by default.
DEFAULT_LARGE_CAP = 15


def classify_bucket(n_meta: int | None, n_emp: int, large_cap: int = DEFAULT_LARGE_CAP) -> str:
    n_eff = effective_count(n_meta, n_emp)
    if n_eff < 2:
        return "skip"
    if n_eff > large_cap:
        return "too_large"
    if n_eff == 2:
        return "binary"
    if n_eff == 3:
        return "tertiary"
    if 4 <= n_eff <= 5:
        return "mid"
    return "large"


@dataclass
class Candidate:
    var_code: str
    section: str
    topic: str
    n_cats_meta: int | None
    n_cats_empirical: int
    bucket: str


def score_pick(cands: list[Candidate], avoid_topics: set[str] | None = None) -> Candidate:
    if avoid_topics:
        key = lambda c: (c.topic not in avoid_topics, c.n_cats_empirical)
        return max(cands, key=key)
    return max(cands, key=lambda c: c.n_cats_empirical)


def _select_once(
    buckets_used: dict[str, list[Candidate]],
    rng: np.random.Generator,
) -> list[Candidate]:
    """
    Quota-driven selection: 2 binary + 2 mid (4-5 cats) + 1 large (5+ cats,
    capped). Graceful fallback when a bucket is short â€” borrows from
    {tertiary, mid, large, binary} in that priority order.
    """
    chosen: list[Candidate] = []
    topics: set[str] = set()
    sections_seen: set[str] = set()

    def take_from(pool_name: str) -> bool:
        pool = buckets_used.get(pool_name) or []
        usable = [c for c in pool if c.topic not in topics]
        cand_list = usable or pool
        if not cand_list:
            return False
        c = score_pick(cand_list, topics if usable else None)
        chosen.append(c)
        topics.add(c.topic)
        sections_seen.add(c.section)
        pool.remove(c)
        return True

    # Target quota
    quota = [("binary", 2), ("mid", 2), ("large", 1)]
    deficit = 0
    for bucket, n in quota:
        for _ in range(n):
            if not take_from(bucket):
                deficit += 1

    # Backfill order: prefer 4-5 cats, then 5+, then 3, then 2 â€” keeping
    # topic / section novelty.
    fallback_order = ["mid", "large", "tertiary", "binary"]
    while len(chosen) < 5 and deficit > 0:
        progressed = False
        for bucket in fallback_order:
            if take_from(bucket):
                progressed = True
                deficit -= 1
                break
        if not progressed:
            break

    return chosen[:5]


def select_five_targets(candidates: list[Candidate], seed: int = 42) -> list[str]:
    """
    Pick up to 5 variable codes: 2 binary + 2 mid (4-5 cats) + 1 large
    (5+ cats, capped via build_candidates). Falls back to tertiary / extra
    binaries / extra mid when a bucket is short. Prefers novel topic strings
    and broader section coverage.

    Retries alternate seeds until >=3 distinct metadata sections represented
    (when enough candidates exist).
    """
    base = {
        "binary": [],
        "tertiary": [],
        "mid": [],
        "large": [],
    }
    for c in candidates:
        if c.bucket in base:
            base[c.bucket].append(c)

    best: list[str] = []
    best_div = (-1,)
    seeds = list(range(seed, seed + 50))[:12]
    for s in seeds:
        buckets_snapshot = {
            k: [x for x in v] for k, v in base.items()
        }  # copy lists
        rng = np.random.default_rng(s)
        for lst in buckets_snapshot.values():
            rng.shuffle(lst)
        chosen_c = _select_once(buckets_snapshot, rng)
        codes = [c.var_code for c in chosen_c]
        n_sec = len({c.section for c in chosen_c})
        rank = (
            min(n_sec, 3),
            n_sec,
            len(codes),
        )
        if rank > best_div or not best:
            best_div = rank
            best = codes
        if rank[0] >= 3:
            break

    return best[:5]


def build_candidates(
    data: pd.DataFrame,
    metadata: dict,
    survey_variables: dict[str, str],
    large_cap: int = DEFAULT_LARGE_CAP,
) -> list[Candidate]:
    out: list[Candidate] = []
    for var_code in sorted(survey_variables.keys()):
        if var_code not in data.columns:
            continue
        sec, info = find_var_location(metadata, var_code)
        if sec is None or info is None:
            continue
        n_meta = substantive_category_count_metadata(info)
        n_emp = substantive_category_count_empirical(data, var_code)
        bucket = classify_bucket(n_meta, n_emp, large_cap=large_cap)
        tk = topic_key(sec, info)
        if bucket in ("skip", "too_large"):
            continue
        out.append(
            Candidate(
                var_code=var_code,
                section=sec,
                topic=tk,
                n_cats_meta=n_meta,
                n_cats_empirical=effective_count(n_meta, n_emp),
                bucket=bucket,
            )
        )
    out.sort(key=lambda c: (c.bucket, -c.n_cats_empirical))
    return out


def _valid_target_rows_per_country(
    data: pd.DataFrame,
    country_col: str,
    country_code,
    var_code: str,
    metadata: dict,
    admin_cols: frozenset,
) -> int:
    """Count non-missing target rows after the same question cleanup as the oracle."""
    from survey_features.surveys import clean_question_columns as _clean_question_columns

    sub = data[data[country_col] == country_code]
    if len(sub) == 0 or var_code not in sub.columns:
        return 0
    cleaned = _clean_question_columns(
        sub.copy(), country_col, admin_cols, metadata
    )
    if var_code not in cleaned.columns:
        return 0
    return int(cleaned[var_code].notna().sum())


def filter_candidates_for_countries(
    candidates: list[Candidate],
    data: pd.DataFrame,
    country_col: str,
    country_names: list[str],
    cmap: dict[str, int | str],
    min_valid_rows: int = 30,
    metadata: dict | None = None,
    admin_cols: frozenset[str] | None = None,
) -> list[Candidate]:
    """
    Keep candidates that have at least ``min_valid_rows`` non-missing responses
    for the target variable in **every** listed country.
    When ``metadata`` and ``admin_cols`` are set, counts match ``compute_oracle``
    cleaning (missing-code stripping, etc.).
    """
    from survey_features.surveys import clean_question_columns as _clean_question_columns

    cleaned_by_country: dict[str, pd.DataFrame] | None = None
    if metadata is not None and admin_cols is not None:
        cleaned_by_country = {}
        for name in country_names:
            code = cmap.get(name)
            if code is None:
                cleaned_by_country[name] = pd.DataFrame()
                continue
            sub = data[data[country_col] == code]
            cleaned_by_country[name] = _clean_question_columns(
                sub.copy(), country_col, admin_cols, metadata
            )

    out: list[Candidate] = []
    for c in candidates:
        if c.var_code not in data.columns:
            continue
        ok = True
        for name in country_names:
            code = cmap.get(name)
            if code is None:
                ok = False
                break
            if cleaned_by_country is not None:
                tab = cleaned_by_country.get(name)
                if tab is None or tab.empty or c.var_code not in tab.columns:
                    n_valid = 0
                else:
                    n_valid = int(tab[c.var_code].notna().sum())
            else:
                sub = data[data[country_col] == code]
                n_valid = int(sub[c.var_code].notna().sum()) if len(sub) else 0
            if n_valid < min_valid_rows:
                ok = False
                break
        if ok:
            out.append(c)
    return out


def pick_spread_country_names(names: list[str], k: int = 5) -> list[str]:
    names = sorted(names)
    n = len(names)
    if n <= k:
        return names
    idx_set = sorted({min(n - 1, int(round(i * (n - 1) / max(k - 1, 1)))) for i in range(k)})
    if len(idx_set) < k:
        for j in range(n):
            if len(idx_set) >= k:
                break
            idx_set.append(j)
        idx_set = sorted(set(idx_set))[:k]
    return [names[i] for i in idx_set]
