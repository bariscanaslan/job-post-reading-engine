# scripts/compare_cleaning_strategies.py

from pathlib import Path
import json

def compare_strategies(job_id: str):
    """Compare semantic vs LLM cleaning for a job"""
    
    semantic_path = Path(f"data/clean/{job_id}/{job_id}_semantic.json")
    llm_path = Path(f"data/clean/{job_id}/{job_id}_llm.json")
    
    if not semantic_path.exists() or not llm_path.exists():
        print("[FAIL] Both files must exist. Run cleaning with both strategies first.")
        return
    
    # [OK] encoding='utf-8' ekle
    with open(semantic_path, encoding='utf-8') as f:
        semantic = json.load(f)
    
    with open(llm_path, encoding='utf-8') as f:
        llm = json.load(f)
    
    print(f"\n[STATS] Comparison for Job {job_id}\n")
    print("=" * 60)
    
    # Section counts
    print("\n[CHART] Section Item Counts:")
    print(f"{'Section':<20} {'Semantic':<12} {'LLM':<12} {'Diff'}")
    print("-" * 60)
    
    all_sections = set(semantic["sections"].keys()) | set(llm["sections"].keys())
    
    for section in sorted(all_sections):
        sem_count = len(semantic["sections"].get(section, []))
        llm_count = len(llm["sections"].get(section, []))
        diff = llm_count - sem_count
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        
        print(f"{section:<20} {sem_count:<12} {llm_count:<12} {diff_str}")
    
    # New sections in LLM
    new_in_llm = set(llm["sections"].keys()) - set(semantic["sections"].keys())
    if new_in_llm:
        print(f"\n[NEW] New sections in LLM: {', '.join(new_in_llm)}")
    
    # Removed sections in LLM
    removed_in_llm = set(semantic["sections"].keys()) - set(llm["sections"].keys())
    if removed_in_llm:
        print(f"\n[DELETE]  Removed sections in LLM: {', '.join(removed_in_llm)}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_cleaning_strategies.py <job_id>")
        sys.exit(1)
    
    compare_strategies(sys.argv[1])
