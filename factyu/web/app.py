import os
import pickle
import sqlite3

from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   url_for)
from flask_socketio import SocketIO, emit
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util

from factyu.config import DB_PATH
from factyu.contextualization.context import run_contextualization
from factyu.database.models import ArticleDatabase
from factyu.extraction.parser import run_extraction

# Get the path to the templates directory
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = os.urandom(24)
socketio = SocketIO(app)

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


@app.route("/")
def index():
    """
    If the database exists, show the main application page (index.html);
    otherwise, show the splash screen that prompts the user to select the bibliography folder.
    """
    if db_exists():
        if db_context_exists():
            return render_template("index.html")
        else:
            return render_template("contextualize.html")
    else:
        return render_template("splash.html")


@app.route("/set_bibliography", methods=["POST"])
def set_bibliography():
    bib_folder = request.form.get("bib_folder")
    if not bib_folder or not os.path.isdir(bib_folder):
        flash("Invalid folder selected", "error")
        return redirect(url_for("splash"))
    try:
        run_extraction(bib_folder)
    except Exception as e:
        flash(f"Extraction failed: {e}", "error")
        return redirect(url_for("splash"))
    return redirect(url_for("index"))


@app.route("/contextualize_fragments", methods=["POST"])
def contextualize_fragments():
    if db_context_exists():
        # If context already exists, send a message with redirect info
        socketio.emit(
            "progress",
            {"percentage": 100, "message": "Context already exists. Redirecting..."},
        )
        return jsonify({"message": "Context already exists.", "redirect": "/"}), 200

    def progress_callback(percentage, message):
        socketio.emit("progress", {"percentage": percentage, "message": message})

    try:
        run_contextualization(progress_callback=progress_callback)
        # Ensure we send a final 100% progress update
        socketio.emit(
            "progress", {"percentage": 100, "message": "Contextualization complete!"}
        )
        return jsonify({"message": "Contextualization complete.", "redirect": "/"}), 200
    except Exception as e:
        error_message = f"Fragment contextualization failed: {e}"
        socketio.emit("progress", {"percentage": 0, "message": error_message})

        # Check if this is an Ollama-related error
        if "Ollama" in str(e):
            socketio.emit("ollama_error", {"message": str(e)})

        return jsonify({"message": error_message}), 500


@app.route("/search", methods=["POST"])
def search():
    new_sentence = request.form.get("sentence")
    matching_entries = find_most_similar_in_db(new_sentence)
    return jsonify(matching_entries=matching_entries)


def db_exists():
    return os.path.exists(DB_PATH)


def db_context_exists():
    print("Checking if database context exists...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables in database: {tables}")
    if ("Referenced",) not in tables:
        print("Referenced table not found in database")
        return False

    cursor.execute("SELECT TextWtContext FROM Referenced")
    rows = cursor.fetchall()
    non_empty_count = sum(
        1 for row in rows if row[0] is not None and row[0].strip() != ""
    )
    print(f"Found {len(rows)} total rows and {non_empty_count} rows with context")
    all_empty = all(row[0] is None or row[0].strip() == "" for row in rows)

    context_exists = not all_empty
    print(f"Context exists: {context_exists}")
    return context_exists


def find_most_similar_in_db(new_sentence):
    new_embedding = model.encode(new_sentence)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute(
        "SELECT TextWtContext, TextEmbeddings, SrcDOI, RefDOI, RefOther, Text, TextInSentence FROM Referenced"
    ).fetchall()
    conn.close()

    aggregated_results = {}

    for row in rows:
        row_text, row_embedding, row_src, refdoi, refother, text, text_in_sentence = row
        unserialized_embedding = pickle.loads(row_embedding)
        distance = cosine(new_embedding, unserialized_embedding)

        if row_text not in aggregated_results:
            aggregated_results[row_text] = {
                "RefDOIs": [],
                "RefOther": [],
                "source": row_src,
                "distance": distance,
                "Text": text,
                "TextInSentence": text_in_sentence,
            }

        aggregated_results[row_text]["RefDOIs"].append(refdoi)
        if refother:
            aggregated_results[row_text]["RefOther"].append(refother)

    # Format for display
    most_similar_entries = [
        {
            "Text": results["Text"],
            "TextWtContext": text,
            "TextInSentence": results["TextInSentence"],
            "SrcDOI": results["source"],
            "RefDOIs": results["RefDOIs"],
            "RefOther": results["RefOther"],
            "distance": results["distance"],
        }
        for text, results in aggregated_results.items()
    ]

    # Sort by distance
    most_similar_entries = sorted(most_similar_entries, key=lambda d: d["distance"])

    return most_similar_entries
