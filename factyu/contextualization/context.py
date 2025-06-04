import pickle
import shutil
import sqlite3
import subprocess

import spacy
from sentence_transformers import SentenceTransformer

from factyu.config import DB_PATH, OLLAMA_BINARY_NAME
from factyu.contextualization.fragment_detector import SentenceFragmentDetector

nlp = spacy.load("en_core_web_sm")
fragment_detector = SentenceFragmentDetector()


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

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if there are any rows that need contextualization
    cursor.execute(
        "SELECT COUNT(*) FROM Referenced WHERE TextWtContext IS NULL OR TextWtContext = ''"
    )
    rows_needing_context = cursor.fetchone()[0]

    if rows_needing_context == 0:
        # No rows need contextualization
        if progress_callback:
            progress_callback(100, "All items already have context.")
        if final_callback:
            final_callback()
        conn.close()
        return

    # Report starting contextualization
    if progress_callback:
        progress_callback(
            0, f"Starting contextualization for {rows_needing_context} items"
        )

    # Ensure "Standalone" column exists in the Referenced table.
    cursor.execute("PRAGMA table_info(Referenced)")
    cols = cursor.fetchall()
    col_names = [col[1] for col in cols]
    if "Standalone" not in col_names:
        cursor.execute("ALTER TABLE Referenced ADD COLUMN Standalone BOOLEAN")
        conn.commit()

    # Process rows grouped by SrcDOI.
    cursor.execute(
        "SELECT DISTINCT SrcDOI FROM Referenced WHERE TextWtContext IS NULL OR TextWtContext = ''"
    )
    src_dois = [row[0] for row in cursor.fetchall()]
    total_groups = len(src_dois)
    group_index = 0

    for src_doi in src_dois:
        group_index += 1
        cursor.execute(
            "SELECT rowid, Text, TextInSentence FROM Referenced WHERE (TextWtContext IS NULL OR TextWtContext = '') AND SrcDOI = ? ORDER BY rowid",
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

    # Call the final callback if provided
    if final_callback:
        final_callback()


def is_dependent(sentence: str) -> bool:
    """
    Analyze if a sentence is complete/standalone or needs contextualization.
    Uses the fine-tuned model to make the determination.
    """
    is_complete, _ = fragment_detector.is_standalone(sentence)
    return not is_complete


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


def export_sentence_analysis_to_tsv(output_path, db_path=None):
    """
    Export sentence analysis results to a TSV file.

    Args:
        output_path (str): Path where the TSV file will be saved
        db_path (str, optional): Path to the database. If None, uses default from config.
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query to get all relevant information
    cursor.execute(
        """
        SELECT Text, TextInSentence, TextWtContext, Standalone, SrcDOI, RefDOI, RefOther
        FROM Referenced
        ORDER BY SrcDOI, rowid
    """
    )
    rows = cursor.fetchall()
    conn.close()

    # Write to TSV file
    with open(output_path, "w", encoding="utf-8") as f:
        # Write header
        f.write(
            "Original Text\tFull Context\tContextualized Text\tIs Complete\tSource DOI\tReference DOI\tOther Reference\n"
        )

        # Write data rows
        for row in rows:
            (
                text,
                text_in_sentence,
                text_wt_context,
                standalone,
                src_doi,
                ref_doi,
                ref_other,
            ) = row
            # Replace tabs and newlines in text fields to avoid breaking TSV format
            text = text.replace("\t", " ").replace("\n", " ") if text else ""
            text_in_sentence = (
                text_in_sentence.replace("\t", " ").replace("\n", " ")
                if text_in_sentence
                else ""
            )
            text_wt_context = (
                text_wt_context.replace("\t", " ").replace("\n", " ")
                if text_wt_context
                else ""
            )

            # Convert boolean to Yes/No for better readability
            is_complete = "Yes" if standalone else "No"

            # Write row
            f.write(
                f"{text}\t{text_in_sentence}\t{text_wt_context}\t{is_complete}\t{src_doi}\t{ref_doi}\t{ref_other}\n"
            )
