import threading

from factyu.config import FLASK_HOST, FLASK_PORT
from factyu.web.app import app


def start_server():
    """Start the Flask server in a separate thread."""
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
