"""
Text cleaning and normalization utilities.

Why text cleaning matters:
- Noisy text (extra whitespace, encoding artifacts, special characters) reduces 
  embedding quality because embeddings capture semantic meaning from text patterns
- Inconsistent formatting can cause retrieval failures (e.g., "  word  " vs "word")
- PDF artifacts like page numbers, headers, footers add noise to chunks
- Clean text improves both embedding generation and LLM comprehension
"""

import re
from typing import List


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    - Collapse multiple spaces/newlines into single
    - Strip leading/trailing whitespace
    - Preserve paragraph structure (double newlines)
    """
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Collapse multiple spaces into single
    text = re.sub(r' +', ' ', text)
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Collapse more than 2 newlines into 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def remove_page_numbers(text: str) -> str:
    """Remove common page number patterns (e.g., 'Page 1', '1 of 10')."""
    # Patterns like "Page 1", "Page 1 of 10", "- 1 -", "1 / 10"
    patterns = [
        r'\bPage\s+\d+\s*(of\s+\d+)?\b',
        r'-\s*\d+\s*-',
        r'\b\d+\s*/\s*\d+\b',
        r'^\d+$',  # Lines with just a number
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    return text


def remove_headers_footers(text: str, common_patterns: List[str] = None) -> str:
    """
    Remove common header/footer patterns.
    
    Args:
        text: Input text
        common_patterns: List of regex patterns to remove
    """
    if common_patterns is None:
        common_patterns = [
            r'Copyright\s+©.*?\d{4}.*?\n',
            r'All\s+rights\s+reserved.*?\n',
            r'Confidential.*?\n',
            r'Draft.*?\n',
        ]
    
    for pattern in common_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text


def fix_encoding_issues(text: str) -> str:
    """Fix common encoding artifacts."""
    # Common replacements
    replacements = {
        '\x00': '',  # Null bytes
        '\x0b': ' ',  # Vertical tab
        '\x0c': ' ',  # Form feed
        '\xa0': ' ',  # Non-breaking space
        '\u200b': '',  # Zero-width space
        '\ufeff': '',  # BOM
        'â€™': "'",   # Smart quote
        'â€œ': '"',   # Smart quote
        'â€': '"',    # Smart quote
        'â€"': '-',   # Em dash
        'â€"': '--',  # En dash
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def remove_urls(text: str, replace_with: str = '[URL]') -> str:
    """Replace URLs with placeholder."""
    url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
    return re.sub(url_pattern, replace_with, text)


def clean_text(
    text: str,
    remove_urls_flag: bool = False,
    remove_page_nums: bool = True,
    remove_headers: bool = True
) -> str:
    """
    Apply full cleaning pipeline to text.
    
    Args:
        text: Raw input text
        remove_urls_flag: Whether to replace URLs
        remove_page_nums: Whether to remove page numbers
        remove_headers: Whether to remove headers/footers
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Fix encoding first
    text = fix_encoding_issues(text)
    
    # Remove artifacts
    if remove_page_nums:
        text = remove_page_numbers(text)
    
    if remove_headers:
        text = remove_headers_footers(text)
    
    if remove_urls_flag:
        text = remove_urls(text)
    
    # Normalize whitespace last
    text = normalize_whitespace(text)
    
    return text


def clean_documents(documents: List) -> List:
    """
    Clean content of a list of Document objects.
    
    Args:
        documents: List of Document objects
    
    Returns:
        Documents with cleaned content
    """
    for doc in documents:
        if hasattr(doc, 'content') and doc.content:
            doc.content = clean_text(doc.content)
        elif hasattr(doc, 'page_content') and doc.page_content:
            doc.page_content = clean_text(doc.page_content)
    
    return documents
