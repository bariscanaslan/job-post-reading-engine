# text_cleaning/extraction/skill_extractor.py

"""
Skill Extractor (Hybrid: NLP + LLM)

Extracts technical skills using:
1. Rule-based spaCy matching (fast, free)
2. LLM-powered extraction (accurate, contextual)

Skill taxonomy loaded from: text_cleaning/resources/skill_taxonomy.json
"""

from pathlib import Path
from typing import Dict, List, Literal
import json
import spacy
from spacy.matcher import PhraseMatcher
from dotenv import load_dotenv
import os

load_dotenv()

ExtractionStrategy = Literal["nlp", "llm", "hybrid"]

# ============================================================
# LOAD NLP MODEL
# ============================================================

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[WARN] Downloading spacy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# ============================================================
# LOAD SKILL TAXONOMY FROM JSON
# ============================================================

def load_skill_taxonomy() -> Dict[str, List[str]]:
    """Load skill phrases from JSON file"""
    base_dir = Path(__file__).resolve().parents[1]
    taxonomy_path = base_dir / "resources" / "skill_taxonomy.json"
    
    if not taxonomy_path.exists():
        raise FileNotFoundError(
            f"[FAIL] Skill taxonomy not found at: {taxonomy_path}\n"
            "Please create text_cleaning/resources/skill_taxonomy.json"
        )
    
    with open(taxonomy_path, encoding='utf-8') as f:
        taxonomy = json.load(f)
    
    # Flatten categories into single dict
    skill_phrases = {}
    for category, skills in taxonomy.items():
        skill_phrases.update(skills)
    
    print(f"[OK] Loaded {len(skill_phrases)} skills from taxonomy")
    return skill_phrases


SKILL_PHRASES = load_skill_taxonomy()


# ============================================================
# NLP-BASED EXTRACTION
# ============================================================

def build_matcher(nlp, skill_phrases):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for skill, phrases in skill_phrases.items():
        patterns = [nlp.make_doc(p) for p in phrases]
        matcher.add(skill, patterns)
    return matcher


matcher = build_matcher(nlp, SKILL_PHRASES)


def extract_skills_nlp(sections: Dict[str, List[str]]) -> Dict[str, Dict]:
    """Rule-based NLP skill extraction"""
    skills: Dict[str, Dict] = {}

    for section, lines in sections.items():
        for line in lines:
            doc = nlp(line)
            matches = matcher(doc)

            for match_id, start, end in matches:
                skill = nlp.vocab.strings[match_id]

                if skill not in skills:
                    skills[skill] = {
                        "count": 0,
                        "sections": set(),
                        "examples": []
                    }

                skills[skill]["count"] += 1
                skills[skill]["sections"].add(section)

                if line not in skills[skill]["examples"] and len(skills[skill]["examples"]) < 3:
                    skills[skill]["examples"].append(line)

    # Normalize sets
    for skill in skills:
        skills[skill]["sections"] = list(skills[skill]["sections"])

    return skills


# ============================================================
# LLM-BASED EXTRACTION
# ============================================================

def extract_skills_llm(sections: Dict[str, List[str]]) -> Dict[str, Dict]:
    """LLM-powered contextual skill extraction"""
    try:
        from openai import OpenAI
        
        if not os.getenv("OPENAI_API_KEY"):
            print("[WARN] OPENAI_API_KEY not set. Falling back to NLP extraction.")
            return extract_skills_nlp(sections)
        
        client = OpenAI()
        
        system_prompt = """You are a technical recruiter skill extractor. Extract ALL technical skills from job descriptions.

Return JSON:
{
  "skill_name": {
    "count": <frequency>,
    "sections": [<section_names>],
    "examples": [<up to 3 quotes>]
  }
}

Extract:
- Programming languages
- Frameworks & libraries
- Tools & platforms
- Methodologies (Agile, etc.)
- Databases
- Cloud services
- Soft skills (leadership, communication)

Normalize: "React.js"  "react", "K8s"  "kubernetes"
"""

        sections_text = json.dumps(sections, indent=2, ensure_ascii=False)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract skills from:\n\n{sections_text}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        skills = json.loads(response.choices[0].message.content)
        return skills
    
    except Exception as e:
        print(f"[WARN] LLM extraction failed: {e}")
        return extract_skills_nlp(sections)


# ============================================================
# HYBRID EXTRACTION
# ============================================================

def extract_skills_hybrid(sections: Dict[str, List[str]]) -> Dict[str, Dict]:
    """Combine NLP + LLM for best results"""
    print("[SEARCH] Running NLP extraction...")
    nlp_skills = extract_skills_nlp(sections)
    
    print("[LLM] Running LLM extraction...")
    llm_skills = extract_skills_llm(sections)
    
    # Merge (LLM takes precedence for conflicts)
    merged = {**nlp_skills}
    
    for skill, data in llm_skills.items():
        if skill in merged:
            # Merge counts and examples
            merged[skill]["count"] += data.get("count", 0)
            merged[skill]["examples"].extend(data.get("examples", []))
            merged[skill]["examples"] = merged[skill]["examples"][:3]  # Limit
        else:
            merged[skill] = data
    
    return merged


# ============================================================
# UNIFIED EXTRACTOR
# ============================================================

def extract_skills(
    sections: Dict[str, List[str]],
    strategy: ExtractionStrategy = "nlp"
) -> Dict[str, Dict]:
    """Extract skills using specified strategy"""
    
    if strategy == "llm":
        return extract_skills_llm(sections)
    elif strategy == "hybrid":
        return extract_skills_hybrid(sections)
    else:
        return extract_skills_nlp(sections)


# ============================================================
# MAIN
# ============================================================

def extract_job_skills(
    input_path: Path,
    output_dir: Path,
    strategy: ExtractionStrategy = "nlp"
) -> Path:
    """Extract skills from cleaned job description"""
    
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    job_id = payload["job_id"]
    sections = payload["sections"]
    cleaning_strategy = payload.get("cleaning_strategy", "unknown")

    print(f"[STATS] Extracting skills using {strategy} strategy...")
    extracted = extract_skills(sections, strategy=strategy)

    # Create job_id subfolder
    job_output_dir = output_dir / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as job_id/{job_id}_{cleaning}_{extraction}.json
    output_filename = f"{job_id}_{cleaning_strategy}_{strategy}.json"
    output_path = job_output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "job_id": job_id,
                "cleaning_strategy": cleaning_strategy,
                "extraction_strategy": strategy,
                "skills": extracted
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
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m text_cleaning.extraction.skill_extractor <path> [--nlp|--llm|--hybrid]")
        print("  python -m text_cleaning.extraction.skill_extractor <job_id> <cleaning_strategy> [--nlp|--llm|--hybrid]")
        sys.exit(1)

    arg = sys.argv[1]
    
    if Path(arg).exists():
        input_path = Path(arg)
    else:
        job_id = arg
        
        if len(sys.argv) > 2 and sys.argv[2] in ["semantic", "llm"]:
            cleaning_strategy = sys.argv[2]
        else:
            cleaning_strategy = "semantic"
        
        base_dir = Path(__file__).resolve().parents[2]
        input_path = base_dir / "data" / "clean" / job_id / f"{job_id}_{cleaning_strategy}.json"
        
        if not input_path.exists():
            print(f"[FAIL] File not found: {input_path}")
            sys.exit(1)
    
    strategy: ExtractionStrategy = "nlp"
    if "--llm" in sys.argv:
        strategy = "llm"
    elif "--hybrid" in sys.argv:
        strategy = "hybrid"

    base_dir = Path(__file__).resolve().parents[2]
    skills_dir = base_dir / "data" / "skills"

    output_path = extract_job_skills(input_path, skills_dir, strategy=strategy)

    print(f"\n[OK] Skills extracted to: {output_path}\n")
