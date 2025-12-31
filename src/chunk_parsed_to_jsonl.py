import os
import re
import json
import glob
import pandas as pd

PARSED_DIR = "data/parsed"
SOURCES_CSV = "sources/sources.csv"
OUT_PATH = "corpus/chunks.jsonl"

# Chunk sizing (words ~ tokens proxy)
TARGET_WORDS = 350
MAX_WORDS = 500
OVERLAP_WORDS = 60

WHITESPACE_RE = re.compile(r"\s+")
SAFE_ID_RE = re.compile(r"[^a-z0-9]+")

# -------------------------
# NEW: Cleaning / junk rules
# -------------------------

# Drop whole sections if the heading looks like site chrome.
JUNK_HEADING_PATTERNS = [
    r"\bsite navigation\b",
    r"\bservice menu\b",
    r"\bmain menu\b",
    r"\byou are here\b",
    r"\bsitemap\b",
    r"\bsearch\b",
    r"\bpage functions?\b",
    r"\blink to social media\b",
    r"\bnote on the use of cookies\b",
    r"\bcookies?\b",
    r"\bimprint\b",
    r"\bdata protection\b",
    r"\baccessibility\b",
    r"\buser notes?\b",
    r"^service$",
    r"^topics$",
    r"^texts and articles$",
    r"^further information$",
    r"^links$",  # comment this out if you want to keep link lists
]
JUNK_HEADING_RE = re.compile("|".join(JUNK_HEADING_PATTERNS), re.IGNORECASE)

# Drop whole sections if the text is clearly chrome / promo / cookie banner.
JUNK_TEXT_PATTERNS = [
    r"\baccept cookies\b",
    r"\bcookies make it easier\b",
    r"\bgo to:\b",
    r"\bprint page\b",
    r"\bsubmen[u|ü]\b",
    r"\bfacebook\b|\binstagram\b|\blinkedin\b|\bthreads\b|\bbluesky\b|\bmastodon\b|\bxing\b|\bvimeo\b",
    r"\bimprint\b|\bdata protection\b|\baccessibility\b|\bsitemap\b",
    r"\bservice for citizens\b|\bservice for other authorities\b",
    # "random teaser/news" type blocks that sometimes get appended
    r"\bannual report\b",
    r"\bfreedom of movement monitoring\b",
]
JUNK_TEXT_RE = re.compile("|".join(JUNK_TEXT_PATTERNS), re.IGNORECASE)

# Lines that are often language switchers / UI lists; we drop if they look like that.
LANG_SWITCH_RE = re.compile(r"(deutsch|english|türkçe|русский|français|العربية)", re.IGNORECASE)

def clean_text(s: str) -> str:
    s = (s or "").replace("\xa0", " ")
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s

def slugify(s: str) -> str:
    s = (s or "").lower()
    s = SAFE_ID_RE.sub("_", s).strip("_")
    return s or "main"

# -------------------------
# NEW: Section/text cleaning
# -------------------------

def heading_is_junk(heading: str) -> bool:
    h = clean_text(heading).lower()
    if not h:
        return True
    return bool(JUNK_HEADING_RE.search(h))

def strip_inline_junk(text: str) -> str:
    """
    Remove common inline UI garbage inside otherwise-useful sections.
    This is conservative: it tries to keep actual sentences.
    """
    if not text:
        return ""

    # Normalize quotes + whitespace early
    t = text.replace("“", '"').replace("”", '"')
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)

    # Remove standalone URL-only lines
    t = re.sub(r"^\s*https?://\S+\s*$", "", t, flags=re.MULTILINE)

    # Drop obvious cookie/banner lines
    t = re.sub(r"^.*cookies make it easier.*$", "", t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r"^.*accept cookies.*$", "", t, flags=re.IGNORECASE | re.MULTILINE)

    # Drop lines that look like language switchers (short lines with many language tokens)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    kept = []
    for ln in lines:
        if LANG_SWITCH_RE.search(ln):
            hits = len(LANG_SWITCH_RE.findall(ln))
            # Heuristic: if it's a short UI-ish line with lots of languages, drop it
            if len(ln) < 120 and hits >= 3:
                continue
        if re.search(r"^\s*go to:\s*", ln, flags=re.IGNORECASE):
            continue
        kept.append(ln)
    t = "\n".join(kept)

    # Final whitespace normalize for paragraph splitting
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

def dedupe_paragraphs(text: str) -> str:
    """
    Remove exact duplicate paragraphs (common in scraped pages).
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    seen = set()
    out = []
    for p in paras:
        key = re.sub(r"\s+", " ", p.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return "\n\n".join(out).strip()

def section_is_junk(heading: str, text: str) -> bool:
    """
    Decide whether to drop an entire section.
    """
    if heading_is_junk(heading):
        return True
    t = clean_text(text).lower()
    if not t:
        return True
    # If the section's text is strongly junky, drop it
    if JUNK_TEXT_RE.search(t):
        return True
    return False

def split_into_paragraphs(text: str):
    """
    Split into paragraphs. Works for most extracted HTML text.
    If no clear paragraph breaks exist, fallback to sentence-ish splitting.
    """
    text = (text or "").strip()
    # Prefer blank-line separation if present
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) >= 2:
        return parts

    # Fallback: split by sentence boundaries (roughly)
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def word_count(s: str) -> int:
    return len((s or "").split())

def make_chunks_from_paragraphs(paras, target_words=TARGET_WORDS, max_words=MAX_WORDS, overlap_words=OVERLAP_WORDS):
    """
    Greedy pack paragraphs into chunks until reaching target/max words.
    Adds overlap by carrying last overlap_words from previous chunk.
    """
    chunks = []
    cur = []
    cur_words = 0

    for p in paras:
        p = clean_text(p)
        if not p:
            continue
        pw = word_count(p)

        # If a single paragraph is huge, hard-split it by words
        if pw > max_words:
            words = p.split()
            start = 0
            while start < len(words):
                end = min(start + max_words, len(words))
                chunk_text = " ".join(words[start:end])
                chunks.append(chunk_text)
                start = end - overlap_words if end < len(words) else end
            continue

        # If adding this paragraph would exceed MAX, flush current chunk first
        if cur and (cur_words + pw) > max_words:
            chunks.append(" ".join(cur))
            # Build overlap: keep last overlap_words from previous chunk
            if overlap_words > 0:
                last_words = " ".join(cur).split()[-overlap_words:]
                cur = [" ".join(last_words)] if last_words else []
                cur_words = word_count(cur[0]) if cur else 0
            else:
                cur, cur_words = [], 0

        cur.append(p)
        cur_words += pw

        # If we've reached target, flush (but not required)
        if cur_words >= target_words:
            chunks.append(" ".join(cur))
            if overlap_words > 0:
                last_words = " ".join(cur).split()[-overlap_words:]
                cur = [" ".join(last_words)] if last_words else []
                cur_words = word_count(cur[0]) if cur else 0
            else:
                cur, cur_words = [], 0

    if cur:
        chunks.append(" ".join(cur))

    # Final cleanup
    chunks = [clean_text(c) for c in chunks if clean_text(c)]
    return chunks

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    src_df = pd.read_csv(SOURCES_CSV)
    src_df = src_df.fillna("unknown")
    meta = {row["source_id"]: row.to_dict() for _, row in src_df.iterrows()}

    parsed_files = sorted(glob.glob(os.path.join(PARSED_DIR, "*.json")))
    if not parsed_files:
        raise RuntimeError(f"No parsed JSON files found in {PARSED_DIR}")

    total_chunks = 0
    skipped_sections = 0
    cleaned_sections = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for fp in parsed_files:
            with open(fp, "r", encoding="utf-8") as f:
                doc = json.load(f)

            source_id = doc["source_id"]
            m = meta.get(source_id)
            if m is None:
                raise ValueError(f"source_id={source_id} not found in sources.csv")

            title = doc.get("title", m.get("title", source_id))
            url = doc.get("url", m.get("url", "unknown"))

            for sec in doc.get("sections", []):
                heading = sec.get("heading", "main")
                raw_text = sec.get("text", "")

                # NEW: drop junk sections early
                if section_is_junk(heading, raw_text):
                    skipped_sections += 1
                    continue

                # NEW: clean within section
                text = strip_inline_junk(raw_text)
                text = dedupe_paragraphs(text)
                text = clean_text(text)

                # Drop if it became empty or too short
                if not text or len(text) < 80:
                    skipped_sections += 1
                    continue

                cleaned_sections += 1

                paras = split_into_paragraphs(text)
                chunk_texts = make_chunks_from_paragraphs(paras)

                section_slug = slugify(heading)
                for i, chunk_text in enumerate(chunk_texts, start=1):
                    chunk_id = f"{source_id}__{section_slug}__{i:04d}"
                    record = {
                        "chunk_id": chunk_id,
                        "source_id": source_id,
                        "title": title,
                        "section": heading,
                        "authority_level": m.get("authority_level", "unknown"),
                        "jurisdiction": m.get("jurisdiction", "unknown"),
                        "document_type": m.get("document_type", "unknown"),
                        "url": url,
                        "text": chunk_text,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1

    print(f"[OK] Wrote {total_chunks} chunks to {OUT_PATH}")
    print(f"[OK] Cleaned sections kept: {cleaned_sections}")
    print(f"[OK] Sections skipped as junk/too-short: {skipped_sections}")

if __name__ == "__main__":
    main()
