import os
import re
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from sec_edgar_downloader import Downloader
from tqdm import tqdm

COMPANY_NAME = "project"
EMAIL_ADDRESS = "chaudharysiddhant04@gmail.com"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
DOWNLOAD_DIR = "data/raw_sec"
PROCESSED_DIR = "data/processed_text"

os.makedirs(PROCESSED_DIR, exist_ok=True)

dl = Downloader(COMPANY_NAME, EMAIL_ADDRESS, DOWNLOAD_DIR)

def download_8k_filings():
    print("Initiating SEC EDGAR 8-K Download...")
    for ticker in TICKERS:
        print(f"  -> Downloading recent 8-K filings for {ticker}...")
        dl.get("8-K", ticker, after="2020-01-01", before="2025-12-31", download_details=False)

def extract_text_and_date(file_path: Path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    filing_date_match = re.search(r"FILING-DATE:\s*(\d{8})", content) or re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", content)
    if filing_date_match:
        raw_date = filing_date_match.group(1)
        filing_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
    else:
        filing_date = None

    soup = BeautifulSoup(content, "lxml")
    
    for s in soup(["script", "style", "table"]):
        s.decompose()
        
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    
    if len(text) < 200:
        return None, None

    return filing_date, text[:4000]  
def process_sec_dumps():
    print("\nParsing and cleaning SEC 8-K filings into structured tabular format...")
    parsed_records = []
    
    sec_path = Path(DOWNLOAD_DIR) / "sec-edgar-filings"
    
    for ticker in TICKERS:
        ticker_path = sec_path / ticker / "8-K"
        if not ticker_path.exists():
            continue
            
        for filing_folder in tqdm(list(ticker_path.glob("*")), desc=f"Parsing {ticker}"):
            full_txt_file = filing_folder / "full-submission.txt"
            if full_txt_file.exists():
                f_date, clean_text = extract_text_and_date(full_txt_file)
                if f_date and clean_text:
                    parsed_records.append({
                        "ticker": ticker,
                        "filing_date": f_date,
                        "accession_no": filing_folder.name,
                        "text": clean_text
                    })
                    
    df_sec = pd.DataFrame(parsed_records)
    df_sec["filing_date"] = pd.to_datetime(df_sec["filing_date"])
    df_sec = df_sec.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    
    output_path = f"{PROCESSED_DIR}/sec_8k_corpus.parquet"
    df_sec.to_parquet(output_path)
    print(f"\n✅ SEC Corpus parsed successfully: {len(df_sec)} total filings saved to '{output_path}'")

if __name__ == "__main__":
    download_8k_filings()
    process_sec_dumps()