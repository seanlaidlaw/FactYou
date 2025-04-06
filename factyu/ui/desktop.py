import sqlite3
import threading

from PyQt6.QtCore import QObject, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from factyu.config import DB_PATH
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
            target=run_extraction,
            args=(folder, self.user_email),
            kwargs={
                "progress_callback": self.update_progress,
                "final_callback": self.extractionComplete.emit,
            },
            daemon=True,
        )
        extraction_thread.start()
        return f"Extraction started for folder:\n{folder}"

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

        # Set up QWebChannel.
        self.channel = QWebChannel()
        self.bridge = Bridge(user_email)
        self.channel.registerObject("bridge", self.bridge)
        self.web_view.page().setWebChannel(self.channel)

        # Connect signals to slots.
        self.bridge.disableButton.connect(self.disable_html_button)
        self.bridge.extractionComplete.connect(self.check_and_redirect)

        # Load your splash page (or Flask URL).
        self.web_view.setUrl(QUrl("http://127.0.0.1:5000/"))

    def disable_html_button(self):
        # Run JavaScript to disable the "Select Bibliography" button.
        js = "document.getElementById('selectBtn').disabled = true;"
        self.web_view.page().runJavaScript(js)

    def check_and_redirect(self):
        try:
            from factyu.database.models import ArticleDatabase

            db = ArticleDatabase()
            count = db.get_referenced_count()

            if count > 0:
                self.web_view.setUrl(QUrl("http://127.0.0.1:5000/"))
            else:
                QMessageBox.warning(
                    self,
                    "Extraction Issue",
                    "No data was extracted from the articles. Please check the console for errors.",
                )
                js = "document.getElementById('selectBtn').disabled = false;"
                self.web_view.page().runJavaScript(js)
        except sqlite3.OperationalError:
            QMessageBox.warning(
                self,
                "Extraction Issue",
                "No data was extracted. The Referenced table doesn't exist.",
            )
            js = "document.getElementById('selectBtn').disabled = false;"
            self.web_view.page().runJavaScript(js)
        finally:
            db.close()
