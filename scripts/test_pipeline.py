# scripts/test_pipeline.py

"""
Full Pipeline Test Script

Tests the complete job posting processing pipeline from scratch.

Usage:
    python scripts/test_pipeline.py <linkedin_job_url>
    
Example:
    python scripts/test_pipeline.py "https://www.linkedin.com/jobs/view/4328768355"
"""

import sys
from pathlib import Path
import subprocess
import shutil
import json
import time

class PipelineTester:
    def __init__(self, job_url: str):
        self.job_url = job_url
        self.base_dir = Path(__file__).resolve().parents[1]
        self.job_id = self._extract_job_id(job_url)
        
        self.python_exe = sys.executable
        
        print(f"\n{'='*80}")
        print(f"[TEST] PIPELINE TEST INITIALIZED")
        print(f"{'='*80}")
        print(f"Job URL: {job_url}")
        print(f"Job ID:  {self.job_id}")
        print(f"Base:    {self.base_dir}")
        print(f"Python:  {self.python_exe}")  
        print(f"{'='*80}\n")
    
    def _extract_job_id(self, url: str) -> str:
        """Extract job ID from LinkedIn URL"""
        import re
        match = re.search(r'/jobs/view/(\d+)', url)
        if not match:
            raise ValueError(f"Invalid LinkedIn job URL: {url}")
        return match.group(1)
    
    def clean_data_directories(self):
        """Clean all data directories for this job_id"""
        print("\n[CLEAN] STEP 0: Cleaning existing data...")
        
        dirs_to_clean = [
            self.base_dir / "data" / "raw" / self.job_id,
            self.base_dir / "data" / "processed" / self.job_id,
            self.base_dir / "data" / "clean" / self.job_id,
            self.base_dir / "data" / "skills" / self.job_id,
        ]
        
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"   Removed: {dir_path}")
        
        print("[OK] Cleanup complete\n")
    
    def run_command(self, cmd: list, step_name: str):
        """Run a command and check for errors"""
        # [OK] 'python' yerine self.python_exe kullan
        if cmd[0] == "python":
            cmd[0] = self.python_exe
        
        print(f"\n{''*80}")
        print(f"[RUN] {step_name}")
        print(f"{''*80}")
        print(f"Command: {' '.join(cmd)}\n")
        
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            cwd=self.base_dir,
            capture_output=True,
            text=True
        )
        
        duration = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"\n[FAIL] FAILED in {duration:.2f}s")
            print(f"Exit code: {result.returncode}")
            raise RuntimeError(f"{step_name} failed")
        
        print(f"\n[OK] SUCCESS in {duration:.2f}s")
        return result
    
    def verify_file_exists(self, filepath: Path, description: str):
        """Verify a file was created"""
        if not filepath.exists():
            raise FileNotFoundError(f"{description} not found: {filepath}")
        
        size = filepath.stat().st_size
        print(f"   {description}: {filepath} ({size:,} bytes)")
    
    def verify_json_structure(self, filepath: Path, required_keys: list):
        """Verify JSON file has required structure"""
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing key '{key}' in {filepath}")
        
        print(f"   JSON structure valid: {required_keys}")
        return data
    
    def test_step_1_scraping(self):
        """Test: Scraping"""
        self.run_command(
            ["python", "scraper/linkedin_job_scraper.py", self.job_url],
            "STEP 1: Scraping LinkedIn Job"
        )
        
        expected_file = self.base_dir / "data" / "raw" / self.job_id / f"{self.job_id}.txt"
        self.verify_file_exists(expected_file, "Raw job description")
    
    def test_step_2_parsing(self):
        """Test: Section Parsing"""
        self.run_command(
            ["python", "-m", "text_cleaning.parsing.section_parser", self.job_id],
            "STEP 2: Section Parsing"
        )
        
        expected_file = self.base_dir / "data" / "processed" / self.job_id / f"{self.job_id}.json"
        self.verify_file_exists(expected_file, "Processed JSON")
        
        data = self.verify_json_structure(expected_file, ["job_id", "sections"])
        print(f"   Sections found: {list(data['sections'].keys())}")
    
    def test_step_3_cleaning_semantic(self):
        """Test: Semantic Cleaning"""
        self.run_command(
            ["python", "-m", "text_cleaning.cleaning.post_cleaner", self.job_id, "--semantic"],
            "STEP 3a: Semantic Cleaning"
        )
        
        expected_file = self.base_dir / "data" / "clean" / self.job_id / f"{self.job_id}_semantic.json"
        self.verify_file_exists(expected_file, "Cleaned (semantic)")
        
        data = self.verify_json_structure(expected_file, ["job_id", "sections", "cleaning_strategy"])
        assert data["cleaning_strategy"] == "semantic"
    
    def test_step_3_cleaning_llm(self):
        """Test: LLM Cleaning"""
        self.run_command(
            ["python", "-m", "text_cleaning.cleaning.post_cleaner", self.job_id, "--llm"],
            "STEP 3b: LLM Cleaning"
        )
        
        expected_file = self.base_dir / "data" / "clean" / self.job_id / f"{self.job_id}_llm.json"
        self.verify_file_exists(expected_file, "Cleaned (LLM)")
        
        data = self.verify_json_structure(expected_file, ["job_id", "sections", "cleaning_strategy"])
        assert data["cleaning_strategy"] == "llm"
    
    def test_step_4_extraction_nlp(self):
        """Test: NLP Skill Extraction"""
        self.run_command(
            ["python", "-m", "text_cleaning.extraction.skill_extractor", self.job_id, "semantic", "--nlp"],
            "STEP 4a: NLP Skill Extraction"
        )
        
        expected_file = self.base_dir / "data" / "skills" / self.job_id / f"{self.job_id}_semantic_nlp.json"
        self.verify_file_exists(expected_file, "Skills (NLP)")
        
        data = self.verify_json_structure(expected_file, ["job_id", "skills", "extraction_strategy"])
        print(f"   Skills extracted: {len(data['skills'])}")
    
    def test_step_4_extraction_llm(self):
        """Test: LLM Skill Extraction"""
        self.run_command(
            ["python", "-m", "text_cleaning.extraction.skill_extractor", self.job_id, "llm", "--llm"],
            "STEP 4b: LLM Skill Extraction"
        )
        
        expected_file = self.base_dir / "data" / "skills" / self.job_id / f"{self.job_id}_llm_llm.json"
        self.verify_file_exists(expected_file, "Skills (LLM)")
        
        data = self.verify_json_structure(expected_file, ["job_id", "skills", "extraction_strategy"])
        print(f"   Skills extracted: {len(data['skills'])}")
    
    def test_step_5_comparison_cleaning(self):
        """Test: Cleaning Comparison"""
        self.run_command(
            ["python", "scripts/compare_cleaning_strategies.py", self.job_id],
            "STEP 5a: Cleaning Strategies Comparison"
        )
    
    def test_step_5_comparison_extraction(self):
        """Test: Extraction Comparison"""
        self.run_command(
            ["python", "scripts/compare_skill_extraction.py", self.job_id],
            "STEP 5b: Skill Extraction Comparison"
        )
    
    def run_full_test(self, clean_first: bool = True):
        """Run complete pipeline test"""
        try:
            if clean_first:
                self.clean_data_directories()
            
            # Core pipeline
            self.test_step_1_scraping()
            self.test_step_2_parsing()
            self.test_step_3_cleaning_semantic()
            self.test_step_3_cleaning_llm()
            self.test_step_4_extraction_nlp()
            self.test_step_4_extraction_llm()
            
            # Comparisons
            self.test_step_5_comparison_cleaning()
            self.test_step_5_comparison_extraction()
            
            print(f"\n{'='*80}")
            print(f"[SUCCESS] ALL TESTS PASSED!")
            print(f"{'='*80}\n")
            
            self._print_summary()
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"[FAIL] TEST FAILED: {e}")
            print(f"{'='*80}\n")
            raise
    
    def _print_summary(self):
        """Print test summary"""
        print("\n[LIST] GENERATED FILES:")
        print(f"{''*80}")
        
        files = [
            ("Raw", f"data/raw/{self.job_id}/{self.job_id}.txt"),
            ("Processed", f"data/processed/{self.job_id}/{self.job_id}.json"),
            ("Clean (Semantic)", f"data/clean/{self.job_id}/{self.job_id}_semantic.json"),
            ("Clean (LLM)", f"data/clean/{self.job_id}/{self.job_id}_llm.json"),
            ("Skills (NLP)", f"data/skills/{self.job_id}/{self.job_id}_semantic_nlp.json"),
            ("Skills (LLM)", f"data/skills/{self.job_id}/{self.job_id}_llm_llm.json"),
        ]
        
        for description, filepath in files:
            full_path = self.base_dir / filepath
            if full_path.exists():
                size = full_path.stat().st_size
                print(f"   {description:20} {filepath} ({size:,} bytes)")
        
        print(f"{''*80}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_pipeline.py <linkedin_job_url>")
        print("\nExample:")
        print('  python scripts/test_pipeline.py "https://www.linkedin.com/jobs/view/4328768355"')
        sys.exit(1)
    
    job_url = sys.argv[1]
    
    tester = PipelineTester(job_url)
    tester.run_full_test(clean_first=True)


if __name__ == "__main__":
    main()
