from datetime import datetime, timezone


def log(message: str) -> None:
    """Simple UTC timestamped logger."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")
