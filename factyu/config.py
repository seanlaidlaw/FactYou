import argparse
import os
import tempfile

from appdirs import user_data_dir

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--clean",
    action="store_true",
    help="Use temporary database file ignoring persistant store",
)
args = parser.parse_args()

# Application data directory configuration
DATA_FOLDER = user_data_dir("FactYou", "FactYouApp")
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Cache directory for HTML files
CACHE_DIR = os.path.join(DATA_FOLDER, "html_cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database path - use temp file if --clean flag is set
if args.clean:
    DB_PATH = os.path.join(tempfile.gettempdir(), "temp_references.db")
    # delete existing database  if already exists as we are running in clean mode
    if os.path.isfile(DB_PATH):
        os.remove(DB_PATH)
else:
    DB_PATH = os.path.join(DATA_FOLDER, "references.db")

# Constants for external services
USER_AGENT_HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
}
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ID_CONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
EUTILS_SUMMARY_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
CROSSREF_BASE_URL = "https://api.crossref.org/v1/works"
PMC_BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles"
