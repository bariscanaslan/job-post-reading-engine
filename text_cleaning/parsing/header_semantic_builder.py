# text_cleaning/parsing/header_semantic_builder.py

from pathlib import Path
import csv
import re
import json
from collections import defaultdict
from typing import Dict, List, Tuple

# ============================================================
# LOAD HEADER TAXONOMY FROM JSON
# ============================================================

def load_header_taxonomy() -> Dict:
    """Load header classification taxonomy from JSON file"""
    base_dir = Path(__file__).resolve().parents[1]
    taxonomy_path = base_dir / "resources" / "header_taxonomy.json"
    
    if not taxonomy_path.exists():
        raise FileNotFoundError(
            f"[FAIL] Header taxonomy not found at: {taxonomy_path}\n"
            "Please create text_cleaning/resources/header_taxonomy.json"
        )
    
    with open(taxonomy_path, encoding='utf-8') as f:
        taxonomy = json.load(f)
    
    print(f"[OK] Loaded header taxonomy with {len(taxonomy['section_seeds'])} sections")
    return taxonomy


TAXONOMY = load_header_taxonomy()

# Extract components from taxonomy
SECTION_SEEDS = TAXONOMY["section_seeds"]
SECTION_THRESHOLDS = TAXONOMY["section_thresholds"]
IMPORTANT_WORDS = set(TAXONOMY["important_words"])
CONTEXTUAL_RULES = TAXONOMY["contextual_rules"]
SCORING_CONFIG = TAXONOMY["scoring_config"]
MULTI_LABEL_THRESHOLD = SCORING_CONFIG["multi_label_threshold"]

# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text: str) -> str:
    """Normalize text for comparison"""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# ============================================================
# WEIGHTED SCORING
# ============================================================

def score_phrase_weighted(phrase: str, seeds: List[str]) -> Tuple[float, List[str]]:
    """Position-aware and importance-weighted scoring"""
    score = 0.0
    matched = []
    
    config = SCORING_CONFIG

    for seed in seeds:
        # Exact match (highest priority)
        if phrase == seed:
            score += config["exact_match_score"]
            matched.append(f"{seed}(exact)")
            continue
        
        # Starts with seed
        if phrase.startswith(seed):
            score += config["starts_with_score"]
            matched.append(f"{seed}(start)")
            continue
        
        # Contains seed
        if seed in phrase:
            score += config["contains_score"]
            matched.append(seed)
            continue
        
        # Token overlap
        phrase_tokens = set(phrase.split())
        seed_tokens = set(seed.split())
        overlap = phrase_tokens & seed_tokens
        
        if overlap:
            # Weight important words higher
            if any(w in IMPORTANT_WORDS for w in overlap):
                weight = config["important_word_weight"]
            else:
                weight = config["normal_word_weight"]
            
            score += weight * len(overlap)
            matched.extend(list(overlap))

    return score, list(set(matched))

# ============================================================
# CONTEXTUAL CLASSIFICATION
# ============================================================

def apply_contextual_rules(phrase: str, scores: Dict[str, float]) -> Dict[str, float]:
    """Apply contextual rules to override or boost scores"""
    normalized = normalize(phrase)
    
    if normalized in CONTEXTUAL_RULES:
        rule = CONTEXTUAL_RULES[normalized]
        default_section = rule["default"]
        
        # Boost default section score
        if default_section in scores:
            scores[default_section] += 1.0
    
    return scores

# ============================================================
# MULTI-LABEL CLASSIFICATION
# ============================================================

def classify_with_confidence(phrase: str) -> Dict:
    """Classify phrase with confidence and alternative sections"""
    scores = {}
    all_matches = {}
    
    # Score against all sections
    for section, seeds in SECTION_SEEDS.items():
        score, matches = score_phrase_weighted(phrase, seeds)
        scores[section] = score
        all_matches[section] = matches
    
    # Apply contextual rules
    scores = apply_contextual_rules(phrase, scores)
    
    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    primary_section = sorted_scores[0][0]
    primary_score = sorted_scores[0][1]
    
    # Determine confidence
    threshold = SECTION_THRESHOLDS.get(primary_section, 1.5)
    
    if primary_score >= threshold + 1.0:
        confidence = "high"
    elif primary_score >= threshold:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Check for alternative sections
    alternative_sections = []
    if len(sorted_scores) > 1 and sorted_scores[0][1] > 0:
        ratio = sorted_scores[1][1] / sorted_scores[0][1]
        if ratio > MULTI_LABEL_THRESHOLD:
            alternative_sections.append(sorted_scores[1][0])
            confidence = "low"
    
    result = {
        "primary_section": primary_section if primary_score >= threshold else "",
        "primary_score": round(primary_score, 2),
        "confidence": confidence,
        "alternative_sections": alternative_sections,
        "matched_seeds": "|".join(all_matches.get(primary_section, [])),
        "status": "accepted" if primary_score >= threshold else "ambiguous"
    }
    
    return result

# ============================================================
# VALIDATION
# ============================================================

def validate_classification():
    """Test known cases for accuracy"""
    test_cases = {
        "what you'll do": "responsibilities",
        "required skills": "requirements",
        "nice to have": "nice_to_have",
        "how to apply": "cta",
        "about the company": "company_info",
        "key responsibilities": "responsibilities",
        "must have experience": "requirements",
        "we offer": "benefits",
        "qualifications": "requirements",
        "join our team": "cta"
    }
    
    correct = 0
    total = len(test_cases)
    
    print("\n[TEST] VALIDATION TESTS")
    print("=" * 60)
    
    for phrase, expected in test_cases.items():
        result = classify_with_confidence(phrase)
        actual = result["primary_section"]
        
        if actual == expected:
            correct += 1
            print(f"[OK] {phrase:30}  {actual}")
        else:
            print(f"[FAIL] {phrase:30}  Expected: {expected}, Got: {actual}")
    
    accuracy = correct / total * 100
    print("=" * 60)
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print()

# ============================================================
# STATISTICS
# ============================================================

def generate_stats(rows: List[Dict]):
    """Generate classification quality metrics"""
    total = len(rows)
    accepted = sum(1 for r in rows if r["status"] == "accepted")
    ambiguous = total - accepted
    
    section_dist = defaultdict(int)
    confidence_dist = defaultdict(int)
    
    for row in rows:
        if row["predicted_section"]:
            section_dist[row["predicted_section"]] += 1
        confidence_dist[row["confidence"]] += 1
    
    print("\n[STATS] CLASSIFICATION STATS")
    print("=" * 60)
    print(f"Total phrases: {total}")
    print(f"Accepted: {accepted} ({accepted/total*100:.1f}%)")
    print(f"Ambiguous: {ambiguous} ({ambiguous/total*100:.1f}%)")
    
    print("\n[CHART] Section distribution:")
    for section, count in sorted(section_dist.items(), key=lambda x: -x[1]):
        print(f"  {section:20} {count:3} ({count/total*100:.1f}%)")
    
    print("\n[TARGET] Confidence distribution:")
    for conf, count in sorted(confidence_dist.items()):
        print(f"  {conf:10} {count:3} ({count/total*100:.1f}%)")
    print("=" * 60)

# ============================================================
# REVIEW SHEET GENERATOR
# ============================================================

def create_review_sheet(rows: List[Dict], output_dir: Path):
    """Create a separate sheet for ambiguous cases to review manually"""
    review_path = output_dir / "headers_for_review.csv"
    
    ambiguous_rows = [
        row for row in rows 
        if row["status"] == "ambiguous" or row["confidence"] == "low"
    ]
    
    if not ambiguous_rows:
        print("[NEW] No ambiguous cases found!")
        return
    
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "phrase",
            "suggested_section",
            "confidence",
            "score",
            "actual_section",
            "notes"
        ])
        writer.writeheader()
        
        for row in ambiguous_rows:
            writer.writerow({
                "phrase": row["raw_phrase"],
                "suggested_section": row["predicted_section"],
                "confidence": row["confidence"],
                "score": row["score"],
                "actual_section": "",
                "notes": ""
            })
    
    print(f"[LIST] Review sheet created: {review_path}")
    print(f"   ({len(ambiguous_rows)} phrases need review)")

# ============================================================
# BUILDER
# ============================================================

def build_header_semantic_sheet():
    base_dir = Path(__file__).resolve().parents[2]

    input_path = (
        base_dir
        / "text_cleaning"
        / "resources"
        / "most_used_words_for_skills_sorted.txt"
    )

    output_dir = base_dir / "data" / "semantic"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "header_semantic_map.csv"

    rows = []

    for raw in input_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue

        result = classify_with_confidence(raw)
        
        rows.append({
            "raw_phrase": raw,
            "normalized_phrase": normalize(raw),
            "predicted_section": result["primary_section"],
            "score": result["primary_score"],
            "confidence": result["confidence"],
            "alternative_sections": "|".join(result["alternative_sections"]),
            "matched_seeds": result["matched_seeds"],
            "status": result["status"]
        })

    # Write main output
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "raw_phrase",
                "normalized_phrase",
                "predicted_section",
                "score",
                "confidence",
                "alternative_sections",
                "matched_seeds",
                "status"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Header semantic sheet created: {output_path}")
    
    # Generate stats
    generate_stats(rows)
    
    # Create review sheet
    create_review_sheet(rows, output_dir)

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    # Run validation first
    validate_classification()
    
    # Build semantic sheet
    build_header_semantic_sheet()
