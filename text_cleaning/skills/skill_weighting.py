"""
Skill Weighting

Calculates weighted importance of extracted skills
based on the sections they appear in.

Input:
- data/skill_extracted/<job_id>.json

Output:
- dict with weighted skill scores
"""

from pathlib import Path
from typing import Dict
import json


# ============================================================
# SECTION WEIGHTS (BUSINESS LOGIC)
# ============================================================

SECTION_WEIGHTS = {
    "responsibilities": 0.6,
    "requirements": 0.4,
    "intro": 0.1,
    # benefits intentionally excluded
}


# ============================================================
# WEIGHT CALCULATION
# ============================================================

def calculate_skill_weights(skills: Dict) -> Dict[str, float]:
    weighted_skills: Dict[str, float] = {}

    for skill, meta in skills.items():
        score = 0.0

        for section in meta["sections"]:
            if section in SECTION_WEIGHTS:
                score += SECTION_WEIGHTS[section]

        weighted_skills[skill] = round(score, 3)

    return weighted_skills


# ============================================================
# MAIN
# ============================================================

def weight_job_skills(input_path: Path, output_path: Path) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    job_id = payload["job_id"]
    skills = payload["skills"]

    weighted = calculate_skill_weights(skills)

    output = {
        "job_id": job_id,
        "skill_weights": weighted
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    """
    Example:
    python -m text_cleaning.skill_weighting data/skills/4328768355.json
    """

    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m text_cleaning.skill_weighting <path_to_skills_json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "data" / "skill_weighted"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / input_path.name

    weight_job_skills(input_path, output_path)

    print(f"\nSkill weights saved to: {output_path}\n")
