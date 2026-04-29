"""
Phase 0b: Embedding-based mapping pipeline
Maps LLM-generated feature descriptions to WVS survey variables via cosine similarity.
"""

import json
import numpy as np
from pathlib import Path


def extract_survey_variables(metadata: dict, exclude_sections: list[str] = None) -> dict[str, str]:
    """
    Extract {var_code: question_text} from ProfileBuilder metadata.
    
    Args:
        metadata: ProfileBuilder.metadata dict
        exclude_sections: sections to skip (e.g., ["EXCLUDED"])
    
    Returns:
        {var_code: question_text}
    """
    exclude = set(exclude_sections or ["EXCLUDED"])
    variables = {}
    for section, vars_dict in metadata.items():
        if section in exclude:
            continue
        for var_code, info in vars_dict.items():
            text = (info.get("question") or info.get("description") or "").strip()
            if text:
                variables[var_code] = text
    return variables


def build_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a list of texts. Returns (n_texts, dim) array."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def map_features_to_variables(
    results: list[dict],
    survey_variables: dict[str, str],
    survey_embeddings: np.ndarray,
    var_codes: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5,
    min_threshold: float = 0.3,
    exclude_targets: bool = True,
    leakage_threshold: float = 0.85,
) -> list[dict]:
    """
    Map LLM feature descriptions to survey variables via cosine similarity.
    
    Args:
        results: list of result dicts from run_batch
        survey_variables: {var_code: question_text}
        survey_embeddings: pre-computed embeddings for survey variables
        var_codes: ordered list of var codes matching survey_embeddings rows
        model_name: embedding model (must match survey_embeddings)
        top_k: number of candidate matches to return per feature
        min_threshold: minimum similarity to include a candidate
        exclude_targets: if True, exclude the target variable from candidates
        leakage_threshold: exclude candidates with similarity > this to the
            target question text (prevents near-duplicate matches). Set None to disable.
    
    Returns:
        List of mapping dicts, one per feature across all results.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    
    # Pre-compute target question embeddings for leakage filtering
    target_codes = set(r["target"] for r in results if r["features"])
    target_embeddings = {}
    if leakage_threshold is not None:
        for tc in target_codes:
            if tc in survey_variables:
                target_embeddings[tc] = model.encode(
                    [survey_variables[tc]], normalize_embeddings=True
                )[0]
    
    mappings = []
    
    for r in results:
        if not r["features"]:
            continue
        
        target_var = r["target"]
        
        # Build exclusion set: target itself + semantically leaked variables
        excluded_codes = set()
        if exclude_targets:
            excluded_codes.add(target_var)
        
        if leakage_threshold is not None and target_var in target_embeddings:
            target_emb = target_embeddings[target_var]
            target_sims = target_emb @ survey_embeddings.T
            for j, vc in enumerate(var_codes):
                if target_sims[j] > leakage_threshold:
                    excluded_codes.add(vc)
        
        # Embed label-only and label+reasoning separately, take max similarity
        # This prevents reasoning text from pulling short labels (e.g. "age")
        # into the target's topic domain instead of matching the demographic variable
        label_texts = []
        combined_texts = []
        for f in r["features"]:
            label = f.get("feature", "")
            reasoning = f.get("reasoning", "")
            label_texts.append(label)
            combined_texts.append(f"{label}: {reasoning}" if reasoning else label)
        
        label_emb = model.encode(label_texts, normalize_embeddings=True)
        combined_emb = model.encode(combined_texts, normalize_embeddings=True)
        
        sims_label = label_emb @ survey_embeddings.T
        sims_combined = combined_emb @ survey_embeddings.T
        sims = np.maximum(sims_label, sims_combined)  # (n_features, n_variables)
        
        for i, f in enumerate(r["features"]):
            # Sort all candidates, then filter
            sorted_indices = np.argsort(sims[i])[::-1]
            candidates = []
            for idx in sorted_indices:
                if len(candidates) >= top_k:
                    break
                vc = var_codes[idx]
                if vc in excluded_codes:
                    continue
                score = float(sims[i, idx])
                if score < min_threshold:
                    break  # sorted descending, so no more above threshold
                candidates.append({
                    "var_code": vc,
                    "question_text": survey_variables[vc],
                    "similarity": round(score, 4),
                })
            
            mappings.append({
                "target": r["target"],
                "country": r["country"],
                "condition": r["condition"],
                "model": r["model"],
                "feature_label": f.get("feature", ""),
                "feature_reasoning": f.get("reasoning", ""),
                "feature_rank": i,
                "candidates": candidates,
                "top_match_code": candidates[0]["var_code"] if candidates else None,
                "top_match_score": candidates[0]["similarity"] if candidates else None,
            })
    
    return mappings


def print_mapping_summary(mappings: list[dict], ground_truth: dict[str, list[str]] = None):
    """
    Print a readable summary of mappings.
    
    Args:
        mappings: output of map_features_to_variables
        ground_truth: optional {target: [top_var_codes]} for comparison
    """
    current_key = None
    for m in mappings:
        key = (m["target"], m["country"], m["condition"])
        if key != current_key:
            current_key = key
            country_str = m["country"] or "unprompted"
            print(f"\n{'='*70}")
            print(f"{m['target']} | {country_str} | {m['condition']}")
            if ground_truth and m["target"] in ground_truth:
                print(f"  Ground truth top: {ground_truth[m['target']]}")
            print(f"{'='*70}")
        
        top = m["candidates"][0] if m["candidates"] else None
        hit_marker = ""
        if ground_truth and m["target"] in ground_truth and top:
            if top["var_code"] in ground_truth[m["target"]]:
                hit_marker = " *** HIT ***"
        
        if top:
            print(f"  [{m['feature_rank']}] {m['feature_label']}")
            print(f"      -> {top['var_code']} ({top['question_text'][:60]}) sim={top['similarity']:.3f}{hit_marker}")
        else:
            print(f"  [{m['feature_rank']}] {m['feature_label']} -> NO MATCH above threshold")


# ── Usage example ──
#
# from synthetic_sampling import ProfileBuilder
# 
# b = ProfileBuilder("wvs")
# survey_vars = extract_survey_variables(b.metadata)
# var_codes = list(survey_vars.keys())
# var_texts = list(survey_vars.values())
# 
# # Compute embeddings once (cache this)
# survey_emb = build_embeddings(var_texts)
# 
# # Load batch results
# with open("phase0b_results.json") as f:
#     results = json.load(f)
# 
# # Map
# mappings = map_features_to_variables(results, survey_vars, survey_emb, var_codes)
# 
# # Inspect
# ground_truth = {
#     "Q199": ["Q200", "Q4", "Q98"],
#     "Q235": ["Q236"],
#     "Q57": ["Q61", "Q59", "Q66"],
#     "Q47": ["Q262", "Q46"],
#     "Q164": ["Q165", "Q172", "Q6"],
# }
# print_mapping_summary(mappings, ground_truth)
# 
# # Save
# with open("phase0b_mappings.json", "w") as f:
#     json.dump(mappings, f, indent=2, ensure_ascii=False)
