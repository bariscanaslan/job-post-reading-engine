# text_cleaning/cleaning/post_cleaner.py

"""
Post Cleaner

Cleans parsed job sections with two strategies:
1. Semantic: Rule-based cleaning (fast, free)
2. LLM: AI-powered cleaning (slower, costs money)

Input:
- data/processed/<job_id>/<job_id>.json

Output:
- data/clean/<job_id>/<job_id>.json
"""
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from typing import Dict, List, Literal
import json
import re
import os

CleaningStrategy = Literal["semantic", "llm"]


# ============================================================
# SEMANTIC CLEANING (Rule-based)
# ============================================================

DROP_PATTERNS = [
    r"apply via",
    r"send your resume",
    r"contact us",
    r"applications will close",
    r"join",
    r"job description",
    r"privacy notice",
    r"personal data",
    r"final hiring",
    r"show more",
    r"show less",
    r"http",
    r"@"
]


def is_noise(line: str, section: str) -> bool:
    """Rule-based noise detection"""
    line_l = line.lower()

    if section == "intro":
        intro_safe_patterns = [
            r"http",
            r"@",
            r"privacy notice",
            r"personal data"
        ]
        return any(re.search(p, line_l) for p in intro_safe_patterns)

    return any(re.search(p, line_l) for p in DROP_PATTERNS)


def clean_line(line: str) -> str:
    """Normalize a single line"""
    # Remove bullet prefix
    line = re.sub(r"^\-\s*", "", line)
    # Normalize spaces
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def clean_sections_semantic(sections: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Rule-based section cleaning"""
    cleaned: Dict[str, List[str]] = {}

    for section, lines in sections.items():
        cleaned_lines = []

        for line in lines:
            if is_noise(line, section):
                continue

            line = clean_line(line)
            if len(line) < 3:
                continue

            cleaned_lines.append(line)

        if cleaned_lines:
            cleaned[section] = cleaned_lines

    return cleaned


# ============================================================
# LLM CLEANING (AI-powered)
# ============================================================

def clean_sections_llm(sections: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """LLM-powered section cleaning"""
    try:
        from openai import OpenAI
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            print("[WARN] OPENAI_API_KEY not set. Falling back to semantic cleaning.")
            return clean_sections_semantic(sections)
        
        client = OpenAI()
        
        system_prompt = """You are a job posting cleaner. Remove noise while preserving important content.

Remove:
- Application instructions ("apply via", "send resume to")
- Privacy/legal boilerplate
- Generic calls to action
- URLs and email addresses (unless in benefits/contact section)
- Redundant/incomplete fragments

Preserve:
- All job responsibilities
- All requirements and qualifications
- All benefits and perks
- Company description
- Technical details

Return cleaned JSON with same structure."""

        user_prompt = f"""Clean this job posting. Remove noise but keep all important content:

{json.dumps(sections, indent=2, ensure_ascii=False)}

Return only valid JSON."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        cleaned_json = response.choices[0].message.content
        cleaned_sections = json.loads(cleaned_json)
        
        # Validate structure
        if not isinstance(cleaned_sections, dict):
            raise ValueError("LLM returned invalid structure")
        
        return cleaned_sections
    
    except Exception as e:
        print(f"[WARN] LLM cleaning failed: {e}")
        print("Falling back to semantic cleaning...")
        return clean_sections_semantic(sections)


# ============================================================
# UNIFIED CLEANER
# ============================================================

def clean_sections(
    sections: Dict[str, List[str]], 
    strategy: CleaningStrategy = "semantic"
) -> Dict[str, List[str]]:
    """
    Clean sections using specified strategy
    
    Args:
        sections: Raw sections from parser
        strategy: "semantic" (rule-based) or "llm" (AI-powered)
    
    Returns:
        Cleaned sections
    """
    if strategy == "llm":
        print("[LLM] Using LLM cleaning...")
        return clean_sections_llm(sections)
    else:
        print("[SEMANTIC] Using semantic cleaning...")
        return clean_sections_semantic(sections)


# ============================================================
# MAIN ENTRY
# ============================================================

def clean_job_file(
    input_path: Path, 
    output_dir: Path,
    strategy: CleaningStrategy = "semantic"
) -> Path:
    """
    Clean a processed job file
    
    Args:
        input_path: Path to processed JSON
        output_dir: Output directory for cleaned JSON
        strategy: Cleaning strategy ("semantic" or "llm")
    
    Returns:
        Path to cleaned output file
    """
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    job_id = payload["job_id"]
    sections = payload["sections"]

    cleaned_sections = clean_sections(sections, strategy=strategy)

    job_output_dir = output_dir / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"{job_id}_{strategy}.json"
    output_path = job_output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "job_id": job_id,
                "sections": cleaned_sections,
                "cleaning_strategy": strategy
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    return output_path


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    """
    Examples:
    
    Full path:
    python -m text_cleaning.cleaning.post_cleaner data/processed/4328768355/4328768355.json --llm
    
    Job ID only (auto-detect):
    python -m text_cleaning.cleaning.post_cleaner 4328768355 --llm
    """

    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m text_cleaning.cleaning.post_cleaner <path> [--llm|--semantic]")
        print("  python -m text_cleaning.cleaning.post_cleaner <job_id> [--llm|--semantic]")
        sys.exit(1)

    arg = sys.argv[1]
    
    # Check if argument is a path or just job_id
    if Path(arg).exists():
        # Full path provided
        input_path = Path(arg)
    else:
        # Job ID provided, construct path
        job_id = arg
        base_dir = Path(__file__).resolve().parents[2]
        input_path = base_dir / "data" / "processed" / job_id / f"{job_id}.json"
        
        if not input_path.exists():
            print(f"[FAIL] File not found: {input_path}")
            sys.exit(1)
    
    # Determine strategy from args
    strategy: CleaningStrategy = "semantic"
    if "--llm" in sys.argv:
        strategy = "llm"
    elif "--semantic" in sys.argv:
        strategy = "semantic"

    base_dir = Path(__file__).resolve().parents[2]
    clean_dir = base_dir / "data" / "clean"

    output_path = clean_job_file(input_path, clean_dir, strategy=strategy)

    print(f"\n[OK] Cleaned job saved to: {output_path}")
    print(f"[STATS] Strategy used: {strategy}\n")
