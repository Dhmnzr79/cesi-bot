from pathlib import Path

ALIASES = {"alias.md","aliases.md","_aliases.md","index.md","_index.md"}

def is_index_like(path: Path, text: str) -> bool:
    name = path.name.lower()
    if name in ALIASES:
        return True
    if text.count("<!-- aliases:") >= 2:
        return True
    return len(text.strip()) < 400
