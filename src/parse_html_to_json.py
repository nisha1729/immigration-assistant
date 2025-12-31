import os
import json
import re
import pandas as pd
from bs4 import BeautifulSoup, Tag

RAW_DIR = "data/raw"
PARSED_DIR = "data/parsed"
SOURCES_CSV = "sources/sources.csv"

# Keep your original heading tags (h1–h3). If you want h4+ later, expand this set.
HEADING_TAGS = {"h1", "h2", "h3"}

# Keep your original text tags, plus dt/dd which often hold structured content on gov sites
TEXT_TAGS = {"p", "li", "blockquote", "td", "th", "dt", "dd"}

WHITESPACE_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


def pick_main_container(soup: BeautifulSoup) -> Tag:
    """
    Minimal-but-robust: gather reasonable candidates and pick the one that most likely
    contains the full article text. Missing headings often means we picked a too-narrow container.
    """
    candidates = []

    main = soup.find("main")
    if main:
        candidates.append(main)

    for c in [
        soup.find(id="content"),
        soup.find(id="main"),
        soup.find("div", class_=re.compile(r"(content|main|page-content|article)", re.I)),
        soup.find("article"),
    ]:
        if isinstance(c, Tag):
            candidates.append(c)

    body = soup.find("body")
    if body:
        candidates.append(body)

    if not candidates:
        return soup

    # Prefer container with more h2/h3s, then more text. This helps avoid dropping sections.
    def score(tag: Tag) -> tuple[int, int]:
        h_count = len(tag.find_all(["h2", "h3"]))
        txt_len = len(tag.get_text(" ", strip=True))
        return (h_count, txt_len)

    return max(candidates, key=score)


def is_service_box_heading(el: Tag) -> bool:
    """
    BAMF (and other CMS pages) sometimes put nested h2 elements inside widgets/service boxes.
    We do NOT want those nested headings to become new top-level sections (they can cause the
    parent section like 'Competent authorities' to be flushed while still empty).
    """
    if not isinstance(el, Tag):
        return False

    cls = " ".join(el.get("class", [])).strip()
    if "c-service-box__heading" in cls:
        return True

    # Any heading inside a service box should not split sections
    parent = el.find_parent("div", class_=re.compile(r"\bc-service-box\b"))
    return parent is not None


def extract_sections(container: Tag):
    """
    Walk the container in document order. Start a new section at each H1/H2/H3,
    but DO NOT split on nested headings inside service boxes (treat them as inline subheadings).
    Collect text from paragraphs, list items, and table/definition list cells.
    """
    sections = []
    current = {"heading": "main", "text_parts": []}

    def flush_current():
        text = clean_text(" ".join(current["text_parts"]))
        if text:
            sections.append({"heading": current["heading"], "text": text})

    for el in container.descendants:
        if not isinstance(el, Tag):
            continue

        tag = el.name.lower()

        # Start new section on headings (except service-box headings)
        if tag in HEADING_TAGS:
            heading = clean_text(el.get_text(" ", strip=True))
            if heading:
                if is_service_box_heading(el):
                    # Keep nested box heading inside current section so parent section isn't lost
                    current["text_parts"].append(heading)
                else:
                    flush_current()
                    current = {"heading": heading, "text_parts": []}
            continue

        # Collect text blocks
        if tag in TEXT_TAGS:
            txt = clean_text(el.get_text(" ", strip=True))
            # Lower threshold so short-but-meaningful lines don't get dropped
            if txt and len(txt) >= 5:
                current["text_parts"].append(txt)

    flush_current()

    # Fallback if we ended up with nothing usable
    if not sections:
        full_text = clean_text(container.get_text(" ", strip=True))
        if full_text:
            sections = [{"heading": "main", "text": full_text}]

    return sections


def parse_one_html(source_id: str, url: str, title_fallback: str) -> dict:
    raw_path = os.path.join(RAW_DIR, f"{source_id}.html")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing raw HTML for {source_id}: {raw_path}")

    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")

    # Remove obvious boilerplate tags
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = clean_text(soup.title.string)
    if not title:
        title = title_fallback

    container = pick_main_container(soup)
    sections = extract_sections(container)

    # If extraction failed badly, fallback to full-container text as one section
    if not sections:
        full_text = clean_text(container.get_text(" ", strip=True))
        if full_text:
            sections = [{"heading": "main", "text": full_text}]

    return {
        "source_id": source_id,
        "url": url,
        "title": title,
        "sections": sections,
    }


def main():
    os.makedirs(PARSED_DIR, exist_ok=True)

    df = pd.read_csv(SOURCES_CSV)

    needed = {"source_id", "title", "document_type", "url"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"sources.csv missing columns: {missing}")

    html_df = df[df["document_type"].astype(str).str.lower() == "webpage"]

    for _, row in html_df.iterrows():
        source_id = str(row["source_id"])
        url = str(row["url"])
        title = str(row["title"])

        try:
            parsed = parse_one_html(source_id, url, title)
            out_path = os.path.join(PARSED_DIR, f"{source_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)
            print(f"[OK] {source_id} -> {out_path} (sections: {len(parsed['sections'])})")
        except Exception as e:
            print(f"[FAIL] {source_id}: {e}")


if __name__ == "__main__":
    main()