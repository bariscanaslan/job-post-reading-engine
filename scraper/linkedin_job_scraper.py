"""
LinkedIn Job Scraper - TEST / PoC SCRIPT

[WARN] Disclaimer:
This script is for educational and testing purposes only.
LinkedIn actively prevents scraping.
Use responsibly.
"""

from playwright.sync_api import sync_playwright, TimeoutError
from bs4 import BeautifulSoup
import sys
import time
from pathlib import Path
import re

def fetch_page_html(job_url: str) -> str | None:
    """Fetch full HTML of a LinkedIn job page using Playwright."""

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800}
        )

        page = context.new_page()

        try:
            page.goto(job_url, timeout=60000)
            page.wait_for_timeout(4000)
            html = page.content()
        except TimeoutError:
            print("Page load timeout")
            html = None
        finally:
            browser.close()

        return html


# scraper/linkedin_job_scraper.py

# scraper/linkedin_job_scraper.py

def parse_job_data(html: str) -> dict:
    """Extract job title, company name and job description."""

    soup = BeautifulSoup(html, "html.parser")

    # Job Title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else None

    # Company Name
    company_tag = soup.find("a", class_="topcard__org-name-link")
    company = company_tag.get_text(strip=True) if company_tag else None

    # Job Description - IMPROVED: No duplicates
    desc_container = soup.find("div", class_="description__text")
    
    if desc_container:
        # Use get_text with separator to preserve structure
        # Strip "Show more/Show less" artifacts
        description = desc_container.get_text(separator="\n", strip=True)
        
        # Clean up
        description = re.sub(r'Show more\s*Show less', '', description, flags=re.IGNORECASE)
        description = re.sub(r'\n{3,}', '\n\n', description)  # Max 2 newlines
        
    else:
        description = None

    return {
        "title": title,
        "company": company,
        "description": description
    }

def extract_job_id(job_url: str) -> str:
    match = re.search(r"/jobs/view/(\d+)", job_url)
    return match.group(1) if match else "unknown_job"

def save_job_description(job_id: str, job_url: str, description: str):
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = BASE_DIR / "data" / "raw" / job_id
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_path = RAW_DATA_DIR / f"{job_id}.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"JOB LINK:\n{job_url}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        f.write("JOB DESCRIPTION:\n\n")
        f.write(description)

    print(f"Job description saved to {file_path}")



def sanity_checks(html: str) -> None:
    """Basic checks to detect bot blocks or captcha."""

    lowered = html.lower()
    if "captcha" in lowered or "verify" in lowered:
        print("Possible CAPTCHA detected.")
    if "linkedin.com/authwall" in lowered:
        print("Auth wall detected (login required).")


def main():
    if len(sys.argv) != 2:
        print("Usage: python linkedin_job_scraper_test.py <linkedin_job_url>")
        sys.exit(1)

    job_url = sys.argv[1]
    print(f"Fetching: {job_url}")

    html = fetch_page_html(job_url)

    if not html:
        print("Failed to retrieve HTML")
        sys.exit(1)

    sanity_checks(html)

    job_data = parse_job_data(html)

    print("\n========== SCRAPED DATA ==========")
    print(f"Title      : {job_data['title']}")
    print(f"Company    : {job_data['company']}")
    print("\n--- Job Description ---\n")

    if not job_data["description"]:
        print("Job description not found")
        sys.exit(1)

    job_id = extract_job_id(job_url)

    save_job_description(
        job_id=job_id,
        job_url=job_url,
        description=job_data["description"]
    )

    print("\n=================================\n")

if __name__ == "__main__":
    main()
