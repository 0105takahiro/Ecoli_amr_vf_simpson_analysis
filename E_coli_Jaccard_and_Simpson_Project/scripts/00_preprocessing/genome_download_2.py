from pathlib import Path
from Bio import Entrez
import pandas as pd
import hashlib
import urllib.request
import gzip
import shutil
import time
import os
import sys

# -------------------------
# Config
# -------------------------
ROOT = Path(__file__).resolve().parents[2]

GENOMES_DIR = ROOT / "genomes"
GENOMES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR  = ROOT / "output" / "preparation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_CSV = OUTPUT_DIR / "ecoli-genomes_26789_metadata.csv"

# NCBI E-utilities (read from env for publication)
EMAIL   = "0105takahiro@gmail.com"     # e.g., export NCBI_EMAIL="you@example.com"
API_KEY = os.environ.get("NCBI_API_KEY")   # optional

if not EMAIL:
    sys.exit("ERROR: Set environment variable NCBI_EMAIL to your contact email.")
Entrez.email = EMAIL
if API_KEY:
    Entrez.api_key = API_KEY

# Rate limit for Entrez calls
DELAY = 0.34 if API_KEY else 1.0

# -------------------------
# Load filtered metadata → assembly IDs
# index is like "12345678.fasta"
# -------------------------
if not METADATA_CSV.exists():
    sys.exit(f"ERROR: Metadata not found: {METADATA_CSV}")

meta = pd.read_csv(METADATA_CSV, index_col=0)
filtered_fasta_filenames = list(meta.index)
assembly_ids = [Path(f).stem for f in filtered_fasta_filenames]  # "12345.fasta" -> "12345"

# -------------------------
# Utilities
# -------------------------
def md5_of_file(path: Path, block_size: int = 2**20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def https_from_ftp(ftp_url: str) -> str:
    # NCBI provides ftp://ftp.ncbi.nlm.nih.gov/...
    # Use HTTPS mirror for reliability.
    if ftp_url.startswith("ftp://"):
        return "https://" + ftp_url[len("ftp://"):]
    return ftp_url

def get_assembly_summary(assembly_id: str):
    with Entrez.esummary(db="assembly", id=assembly_id, report="full") as h:
        return Entrez.read(h, validate=False)

# -------------------------
# Download + MD5 verify + decompress
# -------------------------
ok_count = 0
skip_count = 0
fail_count = 0

for aid in assembly_ids:
    try:
        # 1) Entrez (rate-limited)
        summary = get_assembly_summary(aid)
        time.sleep(DELAY)

        doc = summary["DocumentSummarySet"]["DocumentSummary"][0]
        ftp_path = doc.get("FtpPath_RefSeq") or doc.get("FtpPath_GenBank")
        if not ftp_path:
            print(f"[SKIP] No FTP path for {aid}")
            skip_count += 1
            continue

        base_url = https_from_ftp(ftp_path)
        prefix   = base_url.rstrip("/").split("/")[-1]  # e.g., GCF_000005845.2_ASM584v2
        fname_gz = f"{prefix}_genomic.fna.gz"
        url_gz   = f"{base_url}/{fname_gz}"

        # 2) Download
        gz_path  = GENOMES_DIR / f"{aid}.fasta.gz"
        fasta_path = GENOMES_DIR / f"{aid}.fasta"
        # (re)download if missing
        if not gz_path.exists():
            urllib.request.urlretrieve(url_gz, gz_path)  # simple & reliable
        else:
            print(f"[INFO] Exists, skipping download: {gz_path.name}")

        # 3) MD5 verify against md5checksums.txt
        md5_url = f"{base_url}/md5checksums.txt"
        md5_text = urllib.request.urlopen(md5_url).read().decode("utf-8", errors="ignore")

        remote_md5 = None
        for line in md5_text.splitlines():
            # Lines look like: "<md5sum>  ./GCF_xxx_genomic.fna.gz"
            if line.strip().endswith(f"/{fname_gz}") or line.strip().endswith(f" {fname_gz}"):
                remote_md5 = line.split()[0]
                break

        if remote_md5 is None:
            print(f"[WARN] No md5 entry found for {fname_gz} ({aid}); continuing without check.")
        else:
            local_md5 = md5_of_file(gz_path)
            if local_md5.lower() != remote_md5.lower():
                gz_path.unlink(missing_ok=True)
                print(f"[FAIL] MD5 mismatch for {aid} (local {local_md5} != remote {remote_md5}); removed.")
                fail_count += 1
                continue

        # 4) Decompress (overwrite if exists)
        with gzip.open(gz_path, "rb") as fin, fasta_path.open("wb") as fout:
            shutil.copyfileobj(fin, fout)
        gz_path.unlink(missing_ok=True)

        print(f"[OK] {aid} → {fasta_path.name}")
        ok_count += 1

    except Exception as e:
        # Clean up partial .gz if present
        try:
            (GENOMES_DIR / f"{aid}.fasta.gz").unlink(missing_ok=True)
        except Exception:
            pass
        print(f"[ERROR] {aid}: {e}")
        fail_count += 1

print(f"[DONE] ok={ok_count}, skipped={skip_count}, failed={fail_count}")
