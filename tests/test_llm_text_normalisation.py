# tests/test_llm_text_normalisation.py
from dataclasses import dataclass
from app.deps import normalise_llm_content


@dataclass
class TextBlock:
    type: str
    text: str


def test_normalise_llm_content_string():
    assert normalise_llm_content("Hello, world!") == "Hello, world!"


def test_normalise_llm_content_dict_blocks():
    blocks = [
        {"type": "text", "text": "Part 1. "},
        {"type": "image_url", "image_url": "http://example.com/img.png"},
        {"type": "text", "text": "Part 2."},
    ]
    assert normalise_llm_content(blocks) == "Part 1. Part 2."


def test_normalise_llm_content_object_blocks():
    blocks = [
        TextBlock(type="text", text="Alpha "),
        TextBlock(type="other", text="ignore"),
        TextBlock(type="text", text="Beta"),
    ]
    assert normalise_llm_content(blocks) == "Alpha Beta"


def test_normalise_llm_content_fallback():
    assert normalise_llm_content(12345) == "12345"

