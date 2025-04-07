import os
import pickle
import shutil
import sqlite3
import tempfile
import time

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

# Initialize app state
app.process_status = {"stage": "idle", "percentage": 0}


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


@app.route("/add_bibliography", methods=["POST"])
def add_bibliography():
    """Add a new bibliography file and combine it with the existing database"""
    if "bib_file" not in request.files:
        return jsonify({"message": "No file provided"}), 400

    bib_file = request.files["bib_file"]
    if bib_file.filename == "":
        return jsonify({"message": "No file selected"}), 400

    if not bib_file.filename.endswith(".bib"):
        return jsonify({"message": "File must be a .bib file"}), 400

    # Create a temporary directory to store the uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        bib_file_path = os.path.join(temp_dir, bib_file.filename)
        bib_file.save(bib_file_path)

        # Get the user email from settings (this is required for run_extraction)
        from PyQt6.QtCore import QSettings

        settings = QSettings("FactYou", "FactYouApp")
        user_email = settings.value("user_email", "")

        if not user_email:
            return jsonify({"message": "User email not found in settings"}), 500

        # Store this in app context to track progress across threads
        app.process_status = {"stage": "starting", "percentage": 0}

        try:
            # Progress callback for extraction phase
            def extraction_progress_callback(percentage, message):
                adjusted_percentage = int(percentage * 0.5)  # 0-50% range
                app.process_status = {
                    "stage": "extraction",
                    "percentage": adjusted_percentage,
                }
                socketio.emit(
                    "progress",
                    {
                        "percentage": adjusted_percentage,
                        "message": f"Extraction: {message}",
                    },
                )

            # Callback when extraction is complete to start contextualization
            def extraction_complete_callback():
                # Send a transition message
                socketio.emit(
                    "progress",
                    {
                        "percentage": 50,
                        "message": "Extraction complete. Checking for items to contextualize...",
                    },
                )
                app.process_status = {"stage": "transition", "percentage": 50}

                # Check if there are any rows without context that need processing
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM Referenced WHERE TextWtContext IS NULL OR TextWtContext = ''"
                )
                rows_without_context = cursor.fetchone()[0]
                conn.close()

                if rows_without_context == 0:
                    # No rows need contextualization, we're done
                    app.process_status = {"stage": "complete", "percentage": 100}
                    socketio.emit(
                        "progress",
                        {
                            "percentage": 100,
                            "message": "Processing complete! No new items needed contextualization.",
                        },
                    )
                    return

                try:
                    # Run contextualization on the newly extracted entries
                    def context_progress_callback(percentage, message):
                        adjusted_percentage = 50 + int(
                            percentage * 0.5
                        )  # 50-100% range
                        app.process_status = {
                            "stage": "contextualization",
                            "percentage": adjusted_percentage,
                        }
                        socketio.emit(
                            "progress",
                            {
                                "percentage": adjusted_percentage,
                                "message": f"Contextualization: {message}",
                            },
                        )

                    # Add a slight delay to ensure the transition message is received
                    time.sleep(0.5)

                    # Run contextualization
                    run_contextualization(progress_callback=context_progress_callback)

                    # Send final completion message
                    app.process_status = {"stage": "complete", "percentage": 100}
                    socketio.emit(
                        "progress",
                        {"percentage": 100, "message": "Processing complete!"},
                    )
                except Exception as context_error:
                    error_message = f"Contextualization failed: {context_error}"
                    app.process_status = {"stage": "error", "percentage": 50}
                    socketio.emit(
                        "progress", {"percentage": 50, "message": error_message}
                    )
                    # Check if this is an Ollama-related error
                    if "Ollama" in str(context_error):
                        socketio.emit("ollama_error", {"message": str(context_error)})

            # Start the extraction process
            run_extraction(
                temp_dir,
                user_email,
                progress_callback=extraction_progress_callback,
                final_callback=extraction_complete_callback,
            )

            return jsonify({"message": "Bibliography processing started"}), 200

        except Exception as e:
            error_message = f"Failed to process bibliography: {e}"
            app.process_status = {"stage": "error", "percentage": 0}
            socketio.emit("progress", {"percentage": 0, "message": error_message})
            return jsonify({"message": error_message}), 500


@app.route("/contextualize_fragments", methods=["POST"])
def contextualize_fragments():
    # Check if any rows need contextualization
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # First check if the Referenced table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='Referenced';"
    )
    if not cursor.fetchone():
        conn.close()
        socketio.emit(
            "progress",
            {
                "percentage": 0,
                "message": "Referenced table does not exist. Run extraction first.",
            },
        )
        return jsonify({"message": "Referenced table not found", "redirect": "/"}), 400

    # Check for rows without context
    cursor.execute(
        "SELECT COUNT(*) FROM Referenced WHERE TextWtContext IS NULL OR TextWtContext = ''"
    )
    rows_without_context = cursor.fetchone()[0]
    conn.close()

    if rows_without_context == 0:
        # No rows need contextualization
        socketio.emit(
            "progress",
            {
                "percentage": 100,
                "message": "All items already have context. Redirecting...",
            },
        )
        return jsonify({"message": "All items have context.", "redirect": "/"}), 200

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


@app.route("/processing_status", methods=["GET"])
def processing_status():
    """Returns the current processing status"""
    return jsonify(app.process_status)


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
        conn.close()
        return False

    # First check if the table has any rows at all
    cursor.execute("SELECT COUNT(*) FROM Referenced")
    count = cursor.fetchone()[0]
    if count == 0:
        print("No rows in Referenced table")
        conn.close()
        return False

    # Check if there are any rows WITHOUT context that need processing
    cursor.execute(
        "SELECT COUNT(*) FROM Referenced WHERE TextWtContext IS NULL OR TextWtContext = ''"
    )
    rows_without_context = cursor.fetchone()[0]
    print(f"Found {rows_without_context} rows without context")

    # If there are rows without context, we need to process them
    if rows_without_context > 0:
        print("Some rows need contextualization")
        conn.close()
        return False

    # If we reach here, all rows have context
    print("All rows have context")
    conn.close()
    return True


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
