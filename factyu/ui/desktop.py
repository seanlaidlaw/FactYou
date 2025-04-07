import os
import sqlite3
import threading
import time

from PyQt6.QtCore import QObject, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from factyu.config import DB_PATH, FLASK_HOST, FLASK_PORT, args
from factyu.extraction.parser import run_extraction
from factyu.web.app import app


class Bridge(QObject):
    # Signal to update progress in the HTML: sends an integer (percentage) and a string (message)
    progressUpdated = pyqtSignal(int, str)
    # Signal to disable the HTML button
    disableButton = pyqtSignal()
    # Signal to indicate extraction is complete so we can redirect.
    extractionComplete = pyqtSignal()

    def __init__(self, user_email):
        super().__init__()
        self.user_email = user_email
        self.extraction_done = False

    @pyqtSlot(result=str)
    def selectBibliography(self):
        # Open native folder selection dialog.
        folder = QFileDialog.getExistingDirectory(None, "Select Bibliography Folder")
        if not folder:
            return "No folder selected."

        # Disable the HTML button by emitting the signal.
        self.disableButton.emit()

        # Initialize database only when needed
        from factyu.database.models import ArticleDatabase

        db = ArticleDatabase()

        # Start extraction in a background thread.
        extraction_thread = threading.Thread(
            target=self.run_extraction_with_callback,
            args=(folder,),
            daemon=True,
        )
        extraction_thread.start()
        return f"Extraction started for folder:\n{folder}"

    def run_extraction_with_callback(self, folder):
        """Wrapper function to run extraction and properly signal completion"""
        try:
            print("Starting extraction process...")
            run_extraction(
                folder,
                self.user_email,
                progress_callback=self.update_progress,
                final_callback=self.on_extraction_finished,
            )
        except Exception as e:
            print(f"Error during extraction: {e}")
            self.extractionComplete.emit()  # Still emit the signal to handle the error case

    def on_extraction_finished(self):
        """Called when extraction is finished to ensure database writes are complete"""
        print("Extraction process finished. Ensuring database is updated...")
        # Add a small delay to ensure DB writes are complete
        time.sleep(1)
        self.extraction_done = True
        print("Emitting extractionComplete signal")
        self.extractionComplete.emit()

    def update_progress(self, percentage, message):
        # This callback is called from the extraction process.
        self.progressUpdated.emit(percentage, message)


class MainWindow(QMainWindow):
    def __init__(self, user_email):
        super().__init__()
        self.setWindowTitle("FactYou")
        self.setGeometry(100, 100, 1200, 800)
        self.web_view = QWebEngineView(self)
        self.setCentralWidget(self.web_view)

        # Construct Flask URL from config
        self.flask_url = f"http://{FLASK_HOST}:{FLASK_PORT}"
        print(f"Connecting to Flask server at: {self.flask_url}")

        # Set up QWebChannel.
        self.channel = QWebChannel()
        self.bridge = Bridge(user_email)
        self.channel.registerObject("bridge", self.bridge)
        self.web_view.page().setWebChannel(self.channel)

        # Connect signals to slots.
        self.bridge.disableButton.connect(self.disable_html_button)
        self.bridge.extractionComplete.connect(self.check_and_redirect)

        # Load your splash page (or Flask URL).
        self.web_view.setUrl(QUrl(self.flask_url))

    def disable_html_button(self):
        # Run JavaScript to disable the "Select Bibliography" button.
        js = "document.getElementById('selectBtn').disabled = true;"
        self.web_view.page().runJavaScript(js)

    def check_and_redirect(self):
        print("check_and_redirect called - checking if extraction was successful")
        try:
            # Check database directly using sqlite3 first for diagnostic purposes
            print(f"Checking database at: {DB_PATH}")
            show_error = False  # Initialize the show_error variable

            # If running in clean mode with a new extraction, automatically go to contextualization
            if args.clean:
                print("Running in clean mode - checking if Referenced table exists")
                direct_conn = sqlite3.connect(DB_PATH)
                direct_cursor = direct_conn.cursor()
                direct_cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='Referenced';"
                )
                if direct_cursor.fetchone():
                    print(
                        "Referenced table exists in clean mode - redirecting to contextualization"
                    )
                    direct_conn.close()
                    QTimer.singleShot(
                        500,
                        lambda: self.web_view.setUrl(QUrl(self.flask_url)),
                    )
                    return
                direct_conn.close()

            # Continue with normal checks for non-clean mode
            if not os.path.exists(DB_PATH):
                print(f"WARNING: Database file not found at {DB_PATH}")
                show_error = True
            else:
                print(f"Database file exists at {DB_PATH}")
                try:
                    # Direct check of the table contents
                    direct_conn = sqlite3.connect(DB_PATH)
                    direct_cursor = direct_conn.cursor()

                    # Check if Referenced table exists
                    direct_cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='Referenced';"
                    )
                    if not direct_cursor.fetchone():
                        print("WARNING: Referenced table does not exist")
                        show_error = True
                    else:
                        # Count records in Referenced table directly
                        direct_cursor.execute("SELECT COUNT(*) FROM Referenced")
                        direct_count = direct_cursor.fetchone()[0]
                        print(f"Direct count from database: {direct_count}")

                        if direct_count > 0:
                            print("SUCCESS: Records found via direct database access")
                            direct_conn.close()
                            print("Reloading main page to show contextualize.html")
                            QTimer.singleShot(
                                500,
                                lambda: self.web_view.setUrl(QUrl(self.flask_url)),
                            )
                            return
                        else:
                            print(
                                "WARNING: No records found via direct database access"
                            )
                            show_error = True

                    direct_conn.close()
                except Exception as db_e:
                    print(f"Error during direct database check: {db_e}")
                    show_error = True

            # As a backup, still try the ArticleDatabase approach
            from factyu.database.models import ArticleDatabase

            db = ArticleDatabase()
            count = db.get_referenced_count()
            print(f"Referenced count from ArticleDatabase: {count}")

            if count > 0:
                print(
                    "Extraction successful, reloading page to show contextualize.html"
                )
                QTimer.singleShot(
                    500, lambda: self.web_view.setUrl(QUrl(self.flask_url))
                )
            else:
                print("No records found in Referenced table")
                QMessageBox.warning(
                    self,
                    "Extraction Issue",
                    "No data was extracted from the articles. Please check the console for errors.",
                )
                js = "document.getElementById('selectBtn').disabled = false;"
                self.web_view.page().runJavaScript(js)
            db.close()
        except sqlite3.OperationalError as e:
            print(f"Database error: {e}")
            QMessageBox.warning(
                self,
                "Extraction Issue",
                "No data was extracted. The Referenced table doesn't exist.",
            )
            js = "document.getElementById('selectBtn').disabled = false;"
            self.web_view.page().runJavaScript(js)
        except Exception as e:
            print(f"Unexpected error checking extraction results: {e}")
            QMessageBox.warning(
                self,
                "Extraction Issue",
                f"An unexpected error occurred: {e}",
            )
            js = "document.getElementById('selectBtn').disabled = false;"
            self.web_view.page().runJavaScript(js)
