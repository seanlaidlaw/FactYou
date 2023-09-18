#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import sqlite3

from flask import Flask, jsonify, render_template, request
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    new_sentence = request.form.get("sentence")
    matching_entries = find_most_similar_in_db(new_sentence)
    return jsonify(matching_entries=matching_entries)


def find_most_similar_in_db(new_sentence):
    new_embedding = model.encode(new_sentence)

    conn = sqlite3.connect("articles.db")
    c = conn.cursor()
    rows = c.execute(
        "SELECT TextWtContext, TextEmbeddings, SrcDOI, RefDOI, RefOther FROM Referenced"
    ).fetchall()
    conn.close()

    aggregated_results = {}

    for row in rows:
        row_text, row_embedding, row_src, refdoi, refother = row
        unserialized_embedding = pickle.loads(row_embedding)
        distance = cosine(new_embedding, unserialized_embedding)

        if row_text not in aggregated_results:
            aggregated_results[row_text] = {
                "RefDOIs": [],
                "RefOther": [],
                "source": row_src,
                "distance": distance,
            }

        aggregated_results[row_text]["RefDOIs"].append(refdoi)
        if refother:
            aggregated_results[row_text]["RefOther"].append(refother)

    # Format for display
    most_similar_entries = [
        {
            "Text": text,
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


if __name__ == "__main__":
    app.run(debug=True)
