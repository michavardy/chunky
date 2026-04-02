from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urlparse

from PyPDF2 import PdfReader
import requests
import tqdm
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


DEFAULT_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_tag_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._ignored_tag_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_tag_depth > 0:
            self._ignored_tag_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_tag_depth > 0:
            return
        stripped = data.strip()
        if stripped:
            self.parts.append(stripped)

    def get_text(self) -> str:
        return " ".join(self.parts)


def load_document(path: str | Path, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding)


def _is_html_response(url: str, content_type: str) -> bool:
    normalized_content_type = content_type.lower()
    if "html" in normalized_content_type or "xhtml" in normalized_content_type:
        return True

    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix in {".html", ".htm", ""}


def load_url_document(url: str, timeout: int = 30, verify_ssl: bool = False) -> str:
    response = requests.get(
        url,
        timeout=timeout,
        verify=verify_ssl,
        headers=DEFAULT_REQUEST_HEADERS,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if _is_html_response(url, content_type):
        parser = _HTMLTextExtractor()
        parser.feed(response.text)
        return parser.get_text()

    return response.text


def load_pdf_document(path: str | Path) -> str:
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def load_documents(paths: list[str | Path], encoding: str = "utf-8") -> dict[str, str]:
    return {str(path): load_document(path, encoding=encoding) for path in paths}


def split_document_into_sentences(document: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in tqdm.tqdm(
            re.split(r"(?<=[.!?])\s+", document),
            desc="Splitting document into sentences",
        )
        if sentence.strip()
    ]