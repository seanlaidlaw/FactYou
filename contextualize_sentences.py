#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import sqlite3

import spacy
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


def is_dependent(fragment):
    doc = nlp(fragment)
    for token in doc:
        if "subj" in token.dep_ or "obj" in token.dep_:
            return False
    return True


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
def add_embeddings_to_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all unique SrcDOI values
    cursor.execute("SELECT DISTINCT SrcDOI FROM Referenced")
    src_dois = cursor.fetchall()

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


# Usage
db_path = "articles.db"
add_context_to_fragment(db_path)
add_embeddings_to_db(db_path)
