import argparse
import os
import tempfile

from appdirs import user_data_dir

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--clean",
    action="store_true",
    help="Use temporary database file. The database will persist during the application's lifetime but will be deleted when the application exits.",
)
parser.add_argument(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host address for the Flask web server (default: 127.0.0.1)",
)
parser.add_argument(
    "--port",
    type=int,
    default=5000,
    help="Port for the Flask web server (default: 5000)",
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

# Allow overriding database path via environment variable
ENV_DB_PATH = os.environ.get("FACTYU_DB_PATH")

# set name of Ollama binary to look for in path
OLLAMA_BINARY_NAME = "ollama"

# Database path settings:
# 1. If --clean is set: Use a temporary database (data will be lost when the application exits)
# 2. If FACTYU_DB_PATH environment variable is set: Use that path
# 3. Otherwise: Use the default location in user data directory
if args.clean:
    # DEVELOPMENT/TESTING ONLY: Use a temporary database
    print(
        "WARNING: Running in clean mode - using temporary database. Data will persist during the application's run but will be deleted when the application exits!"
    )
    DB_PATH = os.path.join(tempfile.gettempdir(), "temp_references.db")
    # Delete existing database if already exists as we are running in clean mode
    if os.path.isfile(DB_PATH):
        os.remove(DB_PATH)
elif ENV_DB_PATH:
    # PRODUCTION: Use the database path specified in environment variable
    print(f"Using database path from environment: {ENV_DB_PATH}")
    DB_PATH = ENV_DB_PATH
    # Ensure the directory exists
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
else:
    # DEFAULT: Use the database in the standard data folder
    DB_PATH = os.path.join(DATA_FOLDER, "references.db")
    print(f"Using default database path: {DB_PATH}")

# Flask configuration
FLASK_HOST = args.host
FLASK_PORT = args.port

# Constants for external services
USER_AGENT_HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
}
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ID_CONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
EUTILS_SUMMARY_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
CROSSREF_BASE_URL = "https://api.crossref.org/v1/works"
PMC_BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles"
