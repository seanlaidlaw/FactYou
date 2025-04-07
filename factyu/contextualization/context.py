import pickle
import shutil
import sqlite3
import subprocess

import spacy
from sentence_transformers import SentenceTransformer

from factyu.config import DB_PATH, OLLAMA_BINARY_NAME

nlp = spacy.load("en_core_web_sm")


def run_contextualization(db_path=None, progress_callback=None, final_callback=None):
    """
    Run the contextualization process on the database.
    Also, add a new column "Standalone" in the Referenced table if it doesn't exist.
    For each row (grouped by SrcDOI):
      - If is_dependent(Text) returns False, mark Standalone as True and set TextWtContext to Text.
      - Otherwise, combine the previous context with the current fragment (using TextInSentence if available)
        and mark Standalone as False.
    If db_path is None, use the default from config.
    """
    if db_path is None:
        db_path = DB_PATH

    if not check_ollama_available():
        error_msg = "Ollama is not available in the system PATH. Please install Ollama to continue."
        if progress_callback:
            progress_callback(0, error_msg)
        raise RuntimeError(error_msg)

    # First check if context already exists
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if TextWtContext column has any non-empty values
    cursor.execute(
        "SELECT COUNT(*) FROM Referenced WHERE TextWtContext IS NOT NULL AND TextWtContext != ''"
    )
    context_count = cursor.fetchone()[0]

    if context_count > 0:
        # Context already exists
        if progress_callback:
            progress_callback(100, "Context already exists.")
        if final_callback:
            final_callback()
        conn.close()
        return

    # Ensure "Standalone" column exists in the Referenced table.
    cursor.execute("PRAGMA table_info(Referenced)")
    cols = cursor.fetchall()
    col_names = [col[1] for col in cols]
    if "Standalone" not in col_names:
        cursor.execute("ALTER TABLE Referenced ADD COLUMN Standalone BOOLEAN")
        conn.commit()

    # Process rows grouped by SrcDOI.
    cursor.execute("SELECT DISTINCT SrcDOI FROM Referenced")
    src_dois = [row[0] for row in cursor.fetchall()]
    total_groups = len(src_dois)
    group_index = 0

    for src_doi in src_dois:
        group_index += 1
        cursor.execute(
            "SELECT rowid, Text, TextInSentence FROM Referenced WHERE SrcDOI = ? ORDER BY rowid",
            (src_doi,),
        )
        rows = cursor.fetchall()
        previous_context = ""
        for rowid, text, text_in_sentence in rows:
            if text is None:
                continue
            text = text.strip()
            # Determine if the fragment is incomplete (dependent) using Text.
            paraphrased = text
            if not is_dependent(text):
                standalone = True
            else:  # Incomplete fragment: add context.
                standalone = False
                paraphrased = generate_standalone_sentence_ollama(
                    text_in_sentence, incomplete_chunk=text
                )
                print(f"Incomplete: {text}\nParaphrased: {paraphrased}\n---")
            cursor.execute(
                "UPDATE Referenced SET TextWtContext = ?, Standalone = ? WHERE rowid = ?",
                (paraphrased, standalone, rowid),
            )
        conn.commit()
        # Update progress if a callback is provided.
        if progress_callback:
            progress = int((group_index / total_groups) * 100)
            progress_callback(
                progress,
                f"Processed group {group_index} of {total_groups} (SrcDOI: {src_doi})",
            )
    conn.close()
    add_embeddings_to_db()
    if final_callback:
        final_callback("Contextualization complete.")


def is_dependent(sentence: str) -> bool:
    """
    Analyze the dependency tree of the sentence and apply additional checks
    to determine if it is incomplete.

    The function checks for:
      - Very short sentences (fewer than 4 tokens).
      - Sentences that start with a verb, which often indicates a subordinate clause.
      - Absence of a nominal subject (nsubj, nsubjpass, or csubj) for the main clause.
      - (For linking verbs like "be", "seem", "become": absence of a predicate complement.)

    Returns True if the sentence appears incomplete, False if it appears complete.
    """
    doc = nlp(sentence)

    for sent in doc.sents:
        tokens = list(sent)
        # Basic length check: if the sentence has too few tokens, it's likely incomplete.
        if len(tokens) < 4:
            return True

        # If the sentence starts with a verb, it may be a fragment (e.g., "causes damage...")
        if tokens[0].pos_ == "VERB":
            return True

        # Check for the presence of a nominal subject.
        subject_found = any(
            token.dep_ in ("nsubj", "nsubjpass", "csubj") for token in sent
        )
        # Also check that there is at least one main verb (or auxiliary) in the sentence.
        verb_found = any(token.pos_ in ("VERB", "AUX") for token in sent)

        # Identify the root token (main predicate).
        root = next((token for token in sent if token.dep_ == "ROOT"), None)
        # For linking verbs, check if there is a predicate complement.
        predicate_found = any(token.dep_ in ("attr", "acomp") for token in sent)

        # If no subject or no verb is found, we consider the sentence incomplete.
        if not subject_found or not verb_found:
            return True

        # If the root is a linking verb and there's no predicate complement, flag as incomplete.
        if (
            root is not None
            and root.lemma_ in ("be", "seem", "become")
            and not predicate_found
        ):
            return True

    # If none of the checks flag the sentence as incomplete, assume it is complete.
    return False


def check_ollama_available():
    """
    Check if Ollama is available in the system PATH and the required model is installed.
    Returns True if available, False otherwise.
    """
    # Check if ollama command is available
    if shutil.which(OLLAMA_BINARY_NAME) is None:
        return False

    # Check if we can run a basic ollama command
    try:
        result = subprocess.run(
            [OLLAMA_BINARY_NAME, "list"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


def generate_standalone_sentence_ollama(context, incomplete_chunk):
    """
    Constructs a prompt for the model and runs 'ollama run' to obtain a standalone sentence.
    Assumes that 'ollama' is in your PATH and that your desired model is the default.
    """
    # Check if Ollama is available
    if not check_ollama_available():
        raise RuntimeError(
            "Ollama is not available in the system PATH. Please install Ollama to continue."
        )

    prompt = f"Rewrite the following sentence fragments to add missing the context to transform the incomplete clause (missing a subject, predicate, or verb) to be a standalone sentence with all the context needed to be understood. Do not write anything else, do not say  Here is the rewritten sentence, provide only the output requested and nothing else, and do not rewrite any part of the incomplete. Context: '{context}' Incomplete: '{incomplete_chunk}' Standalone Sentence:"

    # Build the command. Adjust the model name if needed (e.g., "llama" or another name).
    cmd = [OLLAMA_BINARY_NAME, "run", "llama3", prompt]

    # Execute the command and capture the output.
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Ollama process timed out. Please check your Ollama installation."
        )
    except subprocess.SubprocessError as e:
        raise RuntimeError(f"Error running Ollama: {str(e)}")

    # Check for errors.
    if result.returncode != 0:
        raise RuntimeError(f"Ollama call failed: {result.stderr}")

    # Return the generated sentence (stripping extra whitespace).
    return result.stdout.strip()


def add_context_to_fragment(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all unique SrcDOI values
    cursor.execute("SELECT DISTINCT SrcDOI FROM Referenced")
    src_dois = cursor.fetchall()

    for src_doi in src_dois:
        src_doi = src_doi[0]  # Get the actual value from tuple

        cursor.execute(
            "SELECT rowid, Text FROM Referenced WHERE SrcDOI = ? ORDER BY rowid",
            (src_doi,),
        )
        rows = cursor.fetchall()

        previous_text = ""

        for idx, (rowid, text) in enumerate(rows):
            if is_dependent(text):
                text_with_context = previous_text + " " + text
                cursor.execute(
                    "UPDATE Referenced SET TextWtContext = ? WHERE rowid = ?",
                    (text_with_context.strip(), rowid),
                )
                previous_text = text_with_context
            else:
                cursor.execute(
                    "UPDATE Referenced SET TextWtContext = ? WHERE rowid = ?",
                    (text, rowid),
                )
                previous_text = text

        # Commit changes after processing each SrcDOI
        conn.commit()

    conn.close()


# `embeddings` now contains the vector representation for each sentence.
def add_embeddings_to_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all unique SrcDOI values
    cursor.execute("SELECT DISTINCT SrcDOI FROM Referenced")
    src_dois = cursor.fetchall()

    # load Sentence transformer for embeddings
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    for src_doi in src_dois:
        src_doi = src_doi[0]  # Get the actual value from tuple

        cursor.execute(
            "SELECT rowid, TextWtContext FROM Referenced WHERE SrcDOI = ? ORDER BY rowid",
            (src_doi,),
        )
        rows = cursor.fetchall()

        for rowid, text in rows:
            # Compute embedding
            embedding = model.encode(text, convert_to_tensor=True)
            # Serialize the numpy array embedding
            serialized_embedding = pickle.dumps(embedding)
            # Update the TextEmbeddings column
            cursor.execute(
                "UPDATE Referenced SET TextEmbeddings = ? WHERE rowid = ?",
                (serialized_embedding, rowid),
            )

    conn.commit()
    conn.close()
