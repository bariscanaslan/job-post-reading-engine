# scripts/remove_emojis.py

"""
Quick script to remove emojis from all Python files
"""

from pathlib import Path
import re

# Emoji mapping
EMOJI_MAP = {
    '[OK]': '[OK]',
    '[FAIL]': '[FAIL]',
    '[RUN]': '[RUN]',
    '[SEMANTIC]': '[SEMANTIC]',
    '[LLM]': '[LLM]',
    '[STATS]': '[STATS]',
    '[SEARCH]': '[SEARCH]',
    '[WARN]': '[WARN]',
    '[SUCCESS]': '[SUCCESS]',
    '[TEST]': '[TEST]',
    '[CLEAN]': '[CLEAN]',
    '[LIST]': '[LIST]',
    '[CHART]': '[CHART]',
    '[TARGET]': '[TARGET]',
    '[NEW]': '[NEW]',
    '[DELETE]': '[DELETE]',
    '[FILE]': '[FILE]',
    '[FOLDER]': '[FOLDER]',
    '[LAUNCH]': '[LAUNCH]',
    '[IDEA]': '[IDEA]',
    '[SKIP]': '[SKIP]',
}

def remove_emojis_from_file(filepath: Path):
    """Remove emojis from a Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Replace emojis
        for emoji, replacement in EMOJI_MAP.items():
            content = content.replace(emoji, replacement)
        
        # Also remove any remaining emoji-like unicode
        content = re.sub(r'[^\x00-\x7F\u0080-\u00FF\u0100-\u017F\u0180-\u024F]', '', content)
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f" Updated: {filepath}")
            return True
        
        return False
    except Exception as e:
        print(f" Error processing {filepath}: {e}")
        return False

def main():
    base_dir = Path(__file__).resolve().parents[1]
    
    python_files = [
        *base_dir.glob("text_cleaning/**/*.py"),
        *base_dir.glob("scripts/*.py"),
        *base_dir.glob("scraper/*.py"),
    ]
    
    updated = 0
    for filepath in python_files:
        if remove_emojis_from_file(filepath):
            updated += 1
    
    print(f"\nUpdated {updated} files")

if __name__ == "__main__":
    main()