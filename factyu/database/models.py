import argparse
import os
import sqlite3
import threading

from factyu.config import DB_PATH

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--clean", action="store_true", help="Use temporary database file")
args = parser.parse_args()


class ArticleDatabase:
    # Use a class-level lock shared by all instances
    db_lock = threading.Lock()

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._local = threading.local()
        self._setup_tables()

    def _get_connection(self):
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
            self._setup_tables_for_connection(self._local.conn)
        return self._local.conn

    def _setup_tables(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._setup_tables_for_connection(conn)
        finally:
            conn.close()

    def _setup_tables_for_connection(self, conn):
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS Referenced (
                    Text TEXT,
                    TextInSentence TEXT,
                    TextWtContext TEXT,
                    TextEmbeddings BLOB,
                    Reference INTEGER,
                    SrcDOI TEXT,
                    RefDOI TEXT,
                    RefOther TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS Scanned (
                    DOI TEXT,
                    PMCID TEXT,
                    Parsed BOOLEAN NOT NULL CHECK (Parsed IN (0, 1)),
                    Hashed BOOLEAN NOT NULL CHECK (Hashed IN (0, 1)),
                    Skipped BOOLEAN NOT NULL CHECK (Skipped IN (0, 1))
                )
                """
            )

    def store_articles(self, content_list):
        with ArticleDatabase.db_lock:
            with self._get_connection() as conn:
                conn.executemany(
                    "INSERT INTO Referenced (Text, Reference, TextInSentence, SrcDOI, RefDOI, RefOther) VALUES (:Text, :Reference, :TextInSentence, :SrcDOI, :RefDOI, :RefOther)",
                    content_list,
                )

    def save_scanned_doi(self, doi, pmc_id, skipped=0):
        with ArticleDatabase.db_lock:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO Scanned VALUES (?, ?, ?, ?, ?)",
                    (doi, pmc_id, 0, 0, skipped),
                )

    def update_scanned_doi(self, doi, parsed=False, hashed=False):
        if parsed:
            with ArticleDatabase.db_lock:
                with self._get_connection() as conn:
                    conn.execute("UPDATE Scanned SET Parsed = 1 WHERE DOI = ?;", (doi,))
        if hashed:
            with ArticleDatabase.db_lock:
                with self._get_connection() as conn:
                    conn.execute("UPDATE Scanned SET Hashed = 1 WHERE DOI = ?;", (doi,))

    def doi_exists(self, doi):
        with ArticleDatabase.db_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM Scanned WHERE DOI = ?", (doi,))
            return cursor.fetchone() is not None

    def get_pmc_articles(self):
        with ArticleDatabase.db_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DOI, PMCID FROM Scanned WHERE Skipped == 0 AND Parsed == 0;"
            )
            return cursor.fetchall()

    def get_processed_dois(self):
        """Returns a list of DOIs that have already been processed (Parsed=1)"""
        with ArticleDatabase.db_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT DOI FROM Scanned WHERE Parsed == 1;")
            return [row[0] for row in cursor.fetchall()]

    def get_unparsed_pmc_articles(self):
        """Returns a list of articles with PMCIDs that haven't been parsed yet"""
        with ArticleDatabase.db_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DOI, PMCID FROM Scanned WHERE Skipped == 0 AND Parsed == 0 AND PMCID IS NOT NULL;"
            )
            return cursor.fetchall()

    def get_referenced_count(self):
        with ArticleDatabase.db_lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM Referenced")
                return cursor.fetchone()[0]

    def close(self):
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn

    def __del__(self):
        """Remove the temporary database file if running in clean mode, but only when the object is destroyed"""
        # Close remaining connections
        self.close()

        # Only delete the temporary database when the application is exiting (this object is being destroyed)
        if args.clean and self.db_path == DB_PATH:
            try:
                print(f"Cleaning up temporary database at {self.db_path}")
                os.remove(self.db_path)
            except (OSError, FileNotFoundError):
                pass
