import os
import sys
import threading

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox

from factyu.config import FLASK_HOST, FLASK_PORT
from factyu.ui.desktop import MainWindow
from factyu.web.app import app


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

    sys.exit(app_qt.exec())


if __name__ == "__main__":
    main()
