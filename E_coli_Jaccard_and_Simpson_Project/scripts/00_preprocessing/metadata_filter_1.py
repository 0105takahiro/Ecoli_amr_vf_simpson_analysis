from pathlib import Path
from Bio import Entrez
import pandas as pd
import re
import time
import os
import sys

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "output" / "preparation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set your email address for NCBI API access (required by NCBI)
EMAIL   = "0105takahiro@gmail.com"
API_KEY = os.environ.get("NCBI_API_KEY")

if not EMAIL:
    sys.exit("ERROR: Set environment variable NCBI_EMAIL (your contact email).")

Entrez.email = EMAIL
if API_KEY:
    Entrez.api_key = API_KEY

DELAY = 0.34 if API_KEY else 1.0

# Example: a subset of assembly IDs
assembly_list = ['19032741', '61500398', '19032721']

# Terms considered invalid for 'Host' and 'Collection_Date' fields
invalid_hosts = {
    'none', 'not applicable', 'not available', 'not collected',
    'not provided', 'missing', 'model bacteria e. coli c60'
}
invalid_dates = {
    'none', 'missing', 'bad', 'not collected', 'unknown',
    'not provided', 'not applicable', 'not available',
    'restricted access', 'nan'
}

def fetch_metadata_filtered(assembly_id: str) -> pd.DataFrame:
    """
    Retrieve metadata from NCBI for the specified Assembly ID.
    Filters out entries with missing or invalid 'Host' or 'Collection_Date'.
    Returns metadata as a one-row pandas DataFrame or an empty DataFrame if invalid.
    """
    try:
        # Retrieve assembly metadata
        with Entrez.esummary(db='assembly', id=assembly_id, report='full') as asm_handle:
            asm_summary = Entrez.read(asm_handle, validate=False)
        doc = asm_summary['DocumentSummarySet']['DocumentSummary'][0]

        organism = doc.get('Organism')
        submission_date = doc.get('SubmissionDate')
        biosample_accn = doc.get('BioSampleAccn')
        assembly_status = doc.get('AssemblyStatus')
        genbank_id = doc.get('Synonym', {}).get('Genbank', '')
        refseq_id = doc.get('Synonym', {}).get('RefSeq', '')
        coverage = doc.get('Coverage', '')

        # Initialize fields
        host = country = isolation_source = collection_date = None

        # Retrieve BioSample metadata if available
        if biosample_accn:
            with Entrez.esearch(db='biosample', term=biosample_accn) as bs_handle:
                bs_record = Entrez.read(bs_handle)
            idlist = bs_record['IdList']
            if idlist:
                with Entrez.esummary(db='biosample', id=idlist[0]) as bs_summary_handle:
                    bs_summary = Entrez.read(bs_summary_handle, validate=False)
                sample_data = bs_summary['DocumentSummarySet']['DocumentSummary'][0]['SampleData']

                # Extract metadata using regular expressions
                host_match = re.search(r'"host" display_name="host">(.*?)<', sample_data)
                country_match = re.search(r'"geo_loc_name" display_name="geographic location">(.*?)<', sample_data)
                source_match = re.search(r'"isolation_source" display_name="isolation source">(.*?)<', sample_data)
                collection_date_match = re.search(r'"collection_date" display_name="collection date">(.*?)<', sample_data)

                host = host_match.group(1).strip() if host_match else None
                country = country_match.group(1).strip() if country_match else None
                isolation_source = source_match.group(1).strip() if source_match else None
                collection_date = collection_date_match.group(1).strip() if collection_date_match else None

        # Normalize and check for invalid terms
        host_check = host.strip().lower() if host else 'none'
        date_check = collection_date.strip().lower() if collection_date else 'missing'

        # Filter based on host and collection date
        if host_check not in invalid_hosts and date_check not in invalid_dates:
            return pd.DataFrame([{
                "File": assembly_id + ".fasta",  # final output filename
                "GenBank ID": genbank_id,
                "RefSeq ID": refseq_id,
                "Organism": organism,
                "Submission date": submission_date,
                "Coverage": coverage,
                "Assembly status": assembly_status,
                "BioSample accession": biosample_accn,
                "Host": host,
                "Country": country,
                "Isolation source": isolation_source,
                "Collection date": collection_date
            }])
        else:
            return pd.DataFrame()

    except Exception as e:
        print(f"[ERROR] Assembly ID {assembly_id}: {e}")
        return pd.DataFrame()

# Loop through assembly IDs and collect metadata (with rate limiting)
metadata_df = pd.DataFrame()
for aid in assembly_list:
    time.sleep(DELAY)  # Avoid exceeding NCBI API rate limits
    df = fetch_metadata_filtered(aid)
    metadata_df = pd.concat([metadata_df, df], ignore_index=True)

# Set 'file' column as the index (used as filename identifier)
metadata_df = metadata_df.set_index('File')

# Extract 4-digit year from 'Collection_Date' field
def extract_year(date_str):
    try:
        match = re.search(r'\d{4}', str(date_str))
        if match:
            return int(match.group(0))
        else:
            return None
    except:
        return None

# Add a new column with the collection year
metadata_df['Collection year'] = metadata_df['Collection date'].apply(extract_year)

# Keep only entries with a valid collection year from 2010 or later
metadata_df = metadata_df[metadata_df['Collection year'].notna()]
metadata_df = metadata_df[metadata_df['Collection year'] >= 2010]

# Save filtered metadata to CSV
metadata_df.to_csv(OUTPUT_DIR / "ecoli_genomes_26789_metadata.csv")
