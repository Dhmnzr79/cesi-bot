import re, yaml
from typing import Tuple

FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)

def strip_frontmatter(text: str) -> Tuple[dict, str]:
    m = FM_RE.match(text)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except Exception:
        fm = {}
    return fm, text[m.end():]

def clean_section_for_prompt(text: str) -> str:
    _, body = strip_frontmatter(text)
    body = HTML_COMMENT_RE.sub("", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()

