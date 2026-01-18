"""
Compare skill extraction results between different strategies
Supports job-based directory structure:
data/skills/<job_id>/<job_id>_semantic_*.json
"""

from pathlib import Path
import json
from typing import Dict
from collections import defaultdict


# ============================================================
# HELPERS
# ============================================================

def load_skills(filepath: Path) -> Dict:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def parse_strategy_name(filepath: Path, job_id: str) -> str:
    """
    Extract strategy name from filename.
    Example:
      4328768355_semantic_hybrid.json → semantic+hybrid
    """
    stem = filepath.stem.replace(job_id + "_", "")
    parts = stem.split("_")
    return "+".join(parts)


# ============================================================
# SINGLE JOB COMPARISON
# ============================================================

def compare_skill_extraction(job_id: str):
    skills_dir = Path("data/skills") / job_id

    if not skills_dir.exists():
        print(f"[FAIL] Skills directory not found: {skills_dir}")
        return

    skill_files = list(skills_dir.glob(f"{job_id}_*.json"))

    if len(skill_files) < 2:
        print(f"[FAIL] Need at least 2 skill extraction files for {job_id}")
        print(f"Found: {[f.name for f in skill_files]}")
        return

    results = {}
    for filepath in skill_files:
        strategy = parse_strategy_name(filepath, job_id)
        results[strategy] = load_skills(filepath)

    print(f"\n[STATS] Skill Extraction Comparison for Job {job_id}")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Strategy Overview
    # --------------------------------------------------------
    print("\n[SEARCH] Strategies Compared:")
    for strategy, data in results.items():
        skill_count = len(data.get("skills", {}))
        print(f"   {strategy:25} {skill_count} skills extracted")

    # --------------------------------------------------------
    # 2. Skill Coverage
    # --------------------------------------------------------
    print("\n[CHART] Skill Coverage:")
    print(f"{'Skill':<25}", end="")
    for strategy in results:
        print(f"{strategy:<25}", end="")
    print("\n" + "-" * 100)

    all_skills = set()
    for data in results.values():
        all_skills.update(data.get("skills", {}).keys())

    for skill in sorted(all_skills):
        print(f"{skill:<25}", end="")
        for strategy, data in results.items():
            meta = data.get("skills", {}).get(skill)
            if meta:
                count = meta.get("count", 0)
                sections = len(meta.get("sections", []))
                print(f"{count}x/{sections}sec".ljust(25), end="")
            else:
                print("".ljust(25), end="")
        print()

    # --------------------------------------------------------
    # 3. Unique Skills
    # --------------------------------------------------------
    print("\n[TARGET] Unique Skills per Strategy:")

    skill_sets = {
        strategy: set(data.get("skills", {}).keys())
        for strategy, data in results.items()
    }

    for strategy, skills in skill_sets.items():
        others = set().union(*(v for k, v in skill_sets.items() if k != strategy))
        unique = skills - others
        if unique:
            print(f"\n  Only in {strategy}:")
            for s in sorted(unique):
                print(f"    - {s}")

    # --------------------------------------------------------
    # 4. False Positives (example heuristic)
    # --------------------------------------------------------
    print("\n[WARN] Potential False Positives:")

    found_fp = False
    for strategy, data in results.items():
        skills = data.get("skills", {})
        if "nextjs" in skills:
            examples = skills["nextjs"].get("examples", [])
            if any("next evolution" in ex.lower() or "what's next" in ex.lower() for ex in examples):
                found_fp = True
                print(f"  {strategy}: nextjs false positive")
                print(f"    \"{examples[0][:120]}...\"")

    if not found_fp:
        print("  No obvious false positives detected")

    # --------------------------------------------------------
    # 5. Section Distribution
    # --------------------------------------------------------
    print("\n[CHART] Section Distribution:")

    section_stats = defaultdict(lambda: defaultdict(int))
    for strategy, data in results.items():
        for meta in data.get("skills", {}).values():
            for sec in meta.get("sections", []):
                section_stats[strategy][sec] += 1

    all_sections = sorted({s for stats in section_stats.values() for s in stats})

    print(f"{'Strategy':<25}", end="")
    for sec in all_sections:
        print(f"{sec:<18}", end="")
    print("\n" + "-" * 100)

    for strategy in results:
        print(f"{strategy:<25}", end="")
        for sec in all_sections:
            print(f"{section_stats[strategy].get(sec, 0):<18}", end="")
        print()

    # --------------------------------------------------------
    # 6. Quality Metrics
    # --------------------------------------------------------
    print("\n[STATS] Quality Metrics:")

    for strategy, data in results.items():
        skills = data.get("skills", {})
        total = len(skills)
        mentions = sum(s.get("count", 0) for s in skills.values())
        avg = mentions / total if total else 0

        high_value = sum(
            1 for s in skills.values()
            if any(sec in {"requirements", "responsibilities"} for sec in s.get("sections", []))
        )

        low_value = sum(
            1 for s in skills.values()
            if all(sec in {"intro", "benefits", "company_info"} for sec in s.get("sections", []))
        )

        print(f"\n  {strategy}:")
        print(f"    Total skills:        {total}")
        print(f"    Avg mentions/skill: {avg:.1f}")
        print(f"    High-value skills:  {high_value} ({high_value/total*100:.0f}%)")
        print(f"    Low-value skills:   {low_value} ({low_value/total*100:.0f}%)")

    print("\n" + "=" * 80)


# ============================================================
# BATCH MODE
# ============================================================

def batch_compare(job_ids: list[str] | None = None):
    base_dir = Path("data/skills")

    if job_ids is None:
        job_ids = sorted(p.name for p in base_dir.iterdir() if p.is_dir())

    print(f"\n[BATCH] Analyzing {len(job_ids)} jobs\n")

    strategy_stats = defaultdict(lambda: {
        "jobs": 0,
        "total_skills": 0,
        "false_positives": 0
    })

    for job_id in job_ids:
        job_dir = base_dir / job_id
        for filepath in job_dir.glob(f"{job_id}_*.json"):
            strategy = parse_strategy_name(filepath, job_id)
            data = load_skills(filepath)
            skills = data.get("skills", {})

            strategy_stats[strategy]["jobs"] += 1
            strategy_stats[strategy]["total_skills"] += len(skills)

            if "nextjs" in skills:
                examples = skills["nextjs"].get("examples", [])
                if any("next evolution" in ex.lower() or "what's next" in ex.lower() for ex in examples):
                    strategy_stats[strategy]["false_positives"] += 1

    print(f"{'Strategy':<25} {'Jobs':<8} {'Total Skills':<15} {'Avg/Job':<10} {'False+':<8}")
    print("-" * 80)

    for strategy, stats in sorted(strategy_stats.items()):
        avg = stats["total_skills"] / stats["jobs"] if stats["jobs"] else 0
        print(f"{strategy:<25} {stats['jobs']:<8} {stats['total_skills']:<15} "
              f"{avg:<10.1f} {stats['false_positives']:<8}")

    print()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single job: python scripts/compare_skill_extraction.py <job_id>")
        print("  Batch:      python scripts/compare_skill_extraction.py --batch")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        batch_compare()
    else:
        compare_skill_extraction(sys.argv[1])
