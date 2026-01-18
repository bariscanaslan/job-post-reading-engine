"""
Section Parser

Splits a raw LinkedIn job description into semantic sections
using header-first, heuristic-based detection.
"""

from pathlib import Path
from typing import Dict, List
import re
import unicodedata
import json
import csv

DEBUG = True  # 🔧 DEBUG SWITCH


# ============================================================
# HEADER  SECTION MAP
# ============================================================

def load_header_map() -> Dict[str, List[str]]:
    base_dir = Path(__file__).resolve().parents[2]
    csv_path = base_dir / "data" / "semantic" / "header_semantic_map.csv"

    header_map: Dict[str, List[str]] = {}

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "accepted":
                continue

            section = row["predicted_section"]
            raw = row["raw_phrase"].lower().strip()
            normalized = row["normalized_phrase"].lower().strip()

            header_map.setdefault(section, []).append(raw)
            if normalized != raw:
                header_map[section].append(normalized)

    if DEBUG:
        print("\n[DEBUG] HEADER_MAP loaded:")
        for sec, patterns in header_map.items():
            print(f"  - {sec}: {patterns[:5]}{'...' if len(patterns) > 5 else ''}")

    return header_map


HEADER_MAP = load_header_map()

# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_unicode(text: str) -> str:
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    return unicodedata.normalize("NFKD", text)


def normalize_text(text: str) -> List[str]:
    text = normalize_unicode(text)
    text = re.sub(r"show more|show less", "", text, flags=re.IGNORECASE)
    text = re.sub(r"={3,}", "", text)
    text = text.lower()

    # protect hyphenated words
    text = re.sub(r'\b(\w+)-(\w+)\b', r'\1___HYPHEN___\2', text)

    # force breaks for explicit headers ONLY (colon-based)
    for patterns in HEADER_MAP.values():
        for pattern in patterns:
            escaped = re.escape(pattern)
            text = re.sub(
                rf'(\s|^)({escaped})\s*:',
                r'\n\n\2:\n',
                text,
                flags=re.IGNORECASE
            )

    # bullet normalization
    text = re.sub(r'\n\s*-', '\n-', text)

    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if any(x in line for x in ['http', 'job link', '===']):
            continue
        lines.append(line.replace('___HYPHEN___', '-'))

    return lines


# ============================================================
# HEADER DETECTION
# ============================================================

def is_implicit_header(line: str) -> bool:
    """
    Colon olmayan ama section başlatabilecek satırlar.
    """
    if line.startswith('-'):
        return False
    if len(line) <= 40 and line.count('.') == 0:
        return True
    return False


def detect_header_section(line: str) -> str | None:
    """
    Detects whether a line is a section header.
    """
    header = line.strip().lower().rstrip(":").strip()

    for section, patterns in HEADER_MAP.items():
        for pattern in patterns:
            pattern = pattern.lower().strip()

            # 🔒 GÜVENLİ EŞLEŞME
            if header == pattern or header.startswith(pattern):
                if DEBUG:
                    print(f"[DEBUG] HEADER MATCH → '{line}' → {section}")
                return section

    return None


# ============================================================
# SECTION SPLITTING
# ============================================================

def split_into_sections(lines: List[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {"intro": []}
    current_section = "intro"

    for line in lines:
        is_header_candidate = (
            (line.endswith(':') and not line.startswith('-'))
            or is_implicit_header(line)
        )

        if is_header_candidate:
            detected = detect_header_section(line)
            if detected:
                current_section = detected
                sections.setdefault(current_section, [])
                if DEBUG:
                    print(f"[DEBUG] SECTION SWITCH → {current_section}")
                continue

        sections[current_section].append(line)

    if DEBUG:
        print("\n[DEBUG] FINAL SECTIONS:")
        for sec, content in sections.items():
            print(f"  - {sec}: {len(content)} lines")

    return sections


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def parse_job_file(raw_txt_path: Path) -> Dict[str, List[str]]:
    with open(raw_txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    lines = normalize_text(raw_text)
    return split_into_sections(lines)


def save_processed_job(job_id: str, sections: Dict[str, List[str]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{job_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"job_id": job_id, "sections": sections}, f, ensure_ascii=False, indent=2)

    return output_path


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m text_cleaning.parsing.section_parser <job_id_or_path>")
        sys.exit(1)

    arg = sys.argv[1]
    base_dir = Path(__file__).resolve().parents[2]

    if Path(arg).exists():
        raw_path = Path(arg)
        job_id = raw_path.stem
    else:
        job_id = arg
        raw_path = base_dir / "data" / "raw" / job_id / f"{job_id}.txt"

    if not raw_path.exists():
        print(f"[FAIL] File not found: {raw_path}")
        sys.exit(1)

    sections = parse_job_file(raw_path)
    out_dir = base_dir / "data" / "processed" / job_id
    save_processed_job(job_id, sections, out_dir)

    print(f"\n[OK] Parsed job saved to data/processed/{job_id}/{job_id}.json\n")
