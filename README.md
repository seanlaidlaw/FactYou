<img src="img/FactYouBanner.png" width="380" alt="FactYou">

# FactYou

A machine-learning based tool for extracting and analyzing scientific paper references. It parses the references in a bibliography (.bib) file and allows searching of referenced sentences contained in the introduction of any of the articles in the bibliography.

<p align="center">
    <img src="img/FactYouDemo.gif" width="80%" alt="FactYou">
</p>

How it works:

- parses bibliography for article DOIs
- looks up the PMC identifiers for each of the DOIs
- parses the PMC html for Introduction/Main/Background section
- matches sentence and reference for each statement in the article's introdution section
- sentence fragments that are incomplete clauses, are reworded using Ollama to generate standalone sentences that best represent what is being stated in the article sentence.
- SentenceTransformers are used to compute the semantic embedding of each sentence which is compared to the user's search term by cosine similarity between the embeddings

## Installation

FactYou uses a few machine learning libraries most of which can be installed with pip from the `requirements.txt`. The _exception_ to this is Ollama which must be installed by the user before FactYou can be run. Installation insctructions for Ollama on desktop can be found [here](https://ollama.com/download).

```bash
# Clone the repository
git clone https://github.com/seanlaidlaw/FactYou.git
cd FactYou

# Install dependencies
pip install -e .
```

## Usage

To launch the application run the module with python:

```bash
python -m factyu.main
```

This uses a persistent database stored in your user data directory in which it stores the extracted information from the bibliography files (.bib) passed to it.

#### Custom Host/Port Configuration

The application will listen on 127.0.0.1:5000 by default.
If port is already in use, a different port can be manually set from the command line argument:

```bash
python -m factyu.main --host 0.0.0.0 --port 80
```

## Database Information

The application stores data in a SQLite database. The default location is:

- **Linux**: `~/.local/share/FactYou/references.db`
- **macOS**: `~/Library/Application Support/FactYou/references.db`
- **Windows**: `C:\Users\<Username>\AppData\Local\FactYouApp\FactYou\references.db`
