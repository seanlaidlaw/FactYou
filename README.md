# FactYou

A tool for extracting and analyzing scientific paper references.

## Installation

```bash
# Clone the repository
git clone https://github.com/seanlaidlaw/FactYou.git
cd factyu

# Install dependencies
pip install -e .
```

## Usage

### Development Mode

For development and testing, you can run the application with:

```bash
python -m factyu.main
```

This uses a persistent database stored in your user data directory.

#### Testing with Clean Database

If you want to use a temporary database for testing purposes:

```bash
python -m factyu.main --clean
```

**Note:** With the `--clean` flag, a temporary database is created and used for the entire session. The database persists for the duration of the application run, allowing you to extract and contextualize data, but it will be deleted when the application exits. This is useful for testing without affecting your persistent database.

### Production Mode

For production use, you can specify the host and port:

```bash
python -m factyu.main --host 0.0.0.0 --port 80
```

#### Custom Database Path

In production, you may want to specify a custom database path. This can be done by setting the `FACTYU_DB_PATH` environment variable:

```bash
export FACTYU_DB_PATH=/path/to/your/database.db
python -m factyu.main
```

#### Host/Port Configuration

The application will listen on 127.0.0.1:5000 by default. For production, you should bind to 0.0.0.0 to accept connections from all interfaces:

```bash
python -m factyu.main --host 0.0.0.0 --port 80
```

## Database Information

The application stores data in a SQLite database. The default location is:

- **Linux**: `~/.local/share/FactYou/references.db`
- **macOS**: `~/Library/Application Support/FactYou/references.db`
- **Windows**: `C:\Users\<Username>\AppData\Local\FactYouApp\FactYou\references.db`

## Troubleshooting

### No Records Found

If you're running with the `--clean` flag and see "No Records Found" errors, check:

1. **Database Initialization**: Make sure you've processed at least one bibliography folder in the current session.
2. **Session Persistence**: Remember that the temporary database created with `--clean` only persists during the application's run. It will be created fresh each time you start the application, so you'll need to process your bibliography folder each time.

For persistent storage:

1. Run without the `--clean` flag
2. Process your bibliography folder once
3. The data will be preserved between application restarts

### Production Deployment

For production use, we recommend:

1. **Never use `--clean`** as it will result in data loss
2. Bind to all interfaces with `--host 0.0.0.0`
3. Consider using a custom database path by setting the `FACTYU_DB_PATH` environment variable
4. If running behind a proxy like Nginx, set the port to something like 8080 and configure the proxy accordingly

## License
