import os
import shutil
import sys
import threading
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Set Qt environment variables
os.environ["QT_QPA_PLATFORM"] = "cocoa"
os.environ["QT_MAC_WANTS_LAYER"] = "1"
os.environ["QT_DEBUG_PLUGINS"] = "1"  # Enable plugin debugging

# Get the site-packages directory
import site

site_packages = site.getsitepackages()[0]

# Add Qt plugin paths
qt_plugin_paths = [
    os.path.join(site_packages, "PyQt6", "Qt6", "plugins"),
    os.path.join(site_packages, "PyQt6_Qt6", "Qt6", "plugins"),
    os.path.join(site_packages, "PyQt6_WebEngine_Qt6", "Qt6", "plugins"),
]

# Set QT_PLUGIN_PATH
os.environ["QT_PLUGIN_PATH"] = os.pathsep.join(qt_plugin_paths)

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox

from factyu.config import FLASK_HOST, FLASK_PORT, OLLAMA_BINARY_NAME
from factyu.ui.desktop import MainWindow
from factyu.web.app import app


def check_ollama_available():
    """
    Check if Ollama is available in the system PATH.
    Returns True if available, False otherwise.
    """
    return shutil.which(OLLAMA_BINARY_NAME) is not None


def main():
    # Create the QApplication first and set its icon.
    app_qt = QApplication(sys.argv)
    icon_path = os.path.abspath("factyu/FactYouIconGPT4o.png")
    app_qt.setWindowIcon(QIcon(QPixmap(icon_path)))

    # Create QSettings object to store application settings
    settings = QSettings("FactYou", "FactYouApp")
    user_email = settings.value("user_email", "")

    while not user_email:
        user_email, ok = QInputDialog.getText(
            None, "User Email", "Enter your email for PubMed lookup:"
        )
        if ok and user_email:
            settings.setValue("user_email", user_email)
        else:
            QMessageBox.critical(
                None, "Error", "Valid email is required to use the application."
            )
            sys.exit(1)

    # Check if Ollama is available
    if not check_ollama_available():
        QMessageBox.critical(
            None,
            "Ollama Not Found",
            "Ollama could not be found in your system PATH. Please install Ollama and ensure it's properly set up before running FactYou.",
        )
        sys.exit(1)

    # Start Flask in a separate thread
    flask_thread = threading.Thread(
        target=app.run,
        kwargs={
            "host": FLASK_HOST,
            "port": FLASK_PORT,
            "debug": False,
            "use_reloader": False,  # Disable reloader in threaded context
        },
        daemon=True,
    )
    flask_thread.start()

    # Create and show the main window
    window = MainWindow(user_email)
    window.show()

    # Set up signal handling for graceful shutdown
    def signal_handler(signum, frame):
        app_qt.quit()
        sys.exit(0)

    import signal

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    sys.exit(app_qt.exec())


if __name__ == "__main__":
    main()
