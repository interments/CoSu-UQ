import logging
import re
from typing import Callable, List, Optional


_SPACY_NLP = None


def sentence_split_step_answer(response: str) -> List[str]:
    if not response:
        return []
    pattern = (
        r"(?=\*\*Step\s*\d+\s*:)|(?=(?<!\*)Step\s*\d+\s*:)|"
        r"(?=\*\*Final\s*Answer\s*:)|(?=(?<!\*)Final\s*Answer\s*:)"
    )
    parts = re.split(pattern, response, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


def sentence_split_spacy(text: str, logger: Optional[logging.Logger] = None) -> List[str]:
    if not text:
        return []
    try:
        global _SPACY_NLP
        if _SPACY_NLP is None:
            import spacy

            _SPACY_NLP = spacy.load("en_core_web_sm")
        segments = [seg.strip() for seg in text.split("\n\n") if seg.strip()]
        sentences = []
        for seg in segments:
            doc = _SPACY_NLP(seg)
            sentences.extend([sent.text for sent in doc.sents])
        return sentences
    except Exception as exc:
        if logger:
            logger.error("Sentence splitting failed with spaCy: %s", exc)
        return [text]


def sentence_split_nltk(text: str, logger: Optional[logging.Logger] = None) -> List[str]:
    if not text:
        return []
    try:
        import nltk

        return nltk.sent_tokenize(text)
    except Exception as exc:
        if logger:
            logger.error("Sentence splitting failed with NLTK: %s", exc)
        return [text]


def get_sentence_splitter(split_method: str, logger: Optional[logging.Logger] = None) -> Callable[[str], List[str]]:
    if split_method == "step_answer":
        return sentence_split_step_answer
    if split_method == "spacy":
        return lambda text: sentence_split_spacy(text, logger=logger)
    if split_method == "nltk":
        import nltk

        nltk.download("punkt", quiet=True)
        return lambda text: sentence_split_nltk(text, logger=logger)
    raise ValueError(f"Unsupported split method: {split_method}")
