"""Load .env from project root; accept keys with spaces (e.g. TELEGRAM BOT TOKEN)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_env_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        val = rest.strip().strip('"').strip("'")
        if not key or not val:
            continue
        canon = key.replace(" ", "_").upper()
        os.environ[canon] = val


def bootstrap_env() -> None:
    root = project_root()
    for name in (".env", "env.txt"):
        p = root / name
        if p.is_file():
            _parse_env_file(p)
    load_dotenv(dotenv_path=root / ".env", override=False)
