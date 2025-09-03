"""Keyword extraction tool (clean implementation)."""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import nltk
import spacy
from textblob import TextBlob
from crewai.tools import BaseTool
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)


@dataclass
class KeywordResult:
    keywords: List[str]
    keyword_scores: Dict[str, float]
    entities: List[Dict[str, str]]
    topics: List[str]
    sentiment: Optional[Dict[str, float]] = None
    processing_time: float = 0.0


class KeywordExtractionTool(BaseTool):
    name: str = "keyword_extractor"
    description: str = "Extracts keywords from text; basic entities/sentiment optional"

    _nlp = PrivateAttr(default=None)
    _stop_words = PrivateAttr(default_factory=set)
    _filter_words = PrivateAttr(default_factory=set)
    _domain_keywords = PrivateAttr(default_factory=dict)

    def __init__(self):
        super().__init__()
        self._ensure_nltk()
        self._init_spacy()
        self._init_terms()

    # Setup helpers
    def _ensure_nltk(self):
        required = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
        for r in required:
            try:
                if r == 'stopwords':
                    nltk.data.find('corpora/stopwords')
                elif 'tagger' in r:
                    nltk.data.find('taggers/averaged_perceptron_tagger')
                else:
                    nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download(r, quiet=True)

    def _init_spacy(self):
        try:
            self._nlp = spacy.load('en_core_web_sm')
        except Exception:
            self._nlp = None

    def _init_terms(self):
        try:
            self._stop_words = set(nltk.corpus.stopwords.words('english'))
        except LookupError:
            self._stop_words = set()
        self._filter_words = {
            'paper','study','research','author','authors','article','section','figure','table','abstract','conclusion','introduction','method','methods','result','results','discussion','references','citation','citations','page','pages','et','al','doi','http','https','www'
        }
        self._domain_keywords = {
            'ml': ['neural','network','deep','learning','model','training','algorithm'],
            'nlp': ['language','text','transformer','bert','gpt'],
        }

    # Core utilities
    def _clean(self, text: str) -> str:
        text = re.sub(r'http[s]?://\S+', ' ', text)
        # Place hyphen at end of character class to avoid range interpretation
        text = re.sub(r'[^a-zA-Z0-9_\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _keywords(self, text: str, top_k: int) -> List[Tuple[str,float]]:
        try:
            tokens = nltk.word_tokenize(text.lower())
            tokens = [t for t in tokens if len(t) > 2 and t.isalpha() and t not in self._stop_words and t not in self._filter_words]
            tags = nltk.pos_tag(tokens)
            filtered = [w for w, pos in tags if pos.startswith(('NN','JJ','VB'))]
            freq = Counter(filtered)
            scored = [(w, f*(1+len(w)/10)) for w,f in freq.items()]
            return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
        except Exception as e:
            logger.error(f"Keyword extraction core failed: {e}")
            return []

    def _entities(self, text: str) -> List[Dict[str,str]]:
        if not self._nlp:
            return []
        try:
            doc = self._nlp(text)
            ents = []
            for ent in doc.ents:
                if ent.label_ in {'PERSON','ORG','EVENT','PRODUCT'}:
                    ents.append({'text': ent.text, 'label': ent.label_})
            return ents
        except Exception:
            return []

    def _sentiment(self, text: str) -> Optional[Dict[str,float]]:
        try:
            blob = TextBlob(text)
            return {'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity}
        except Exception:
            return None

    def _run(self, text: str, top_k: int = 20, include_entities: bool = True, include_sentiment: bool = False) -> str:
        import time
        start = time.time()
        if not text or len(text.strip()) < 10:
            return "Error: Text is too short for meaningful keyword extraction"
        cleaned = self._clean(text)
        kws = self._keywords(cleaned, top_k)
        keywords = [w for w,_ in kws]
        lowered = cleaned.lower()
        for group in self._domain_keywords.values():
            for kw in group:
                if kw in lowered and kw not in keywords:
                    keywords.append(kw)
        if include_entities:
            _ = self._entities(text)
        if include_sentiment:
            _ = self._sentiment(text)
        _ = time.time() - start  # processing time ignored in simple run
        return ", ".join(keywords)

    def extract_detailed(self, text: str, top_k: int = 20, include_entities: bool = True, include_sentiment: bool = True) -> KeywordResult:
        import time
        start = time.time()
        if not text or len(text.strip()) < 10:
            return KeywordResult([], {}, [], [], None, 0.0)
        cleaned = self._clean(text)
        kws = self._keywords(cleaned, top_k)
        keywords = [w for w,_ in kws]
        scores = dict(kws)
        lowered = cleaned.lower()
        for group in self._domain_keywords.values():
            for kw in group:
                if kw in lowered and kw not in scores:
                    keywords.append(kw)
                    scores[kw] = 1.0
        entities = self._entities(text) if include_entities else []
        sentiment = self._sentiment(text) if include_sentiment else None
        processing_time = time.time() - start
        return KeywordResult(keywords, scores, entities, [], sentiment, processing_time)


def extract_keywords(text: str, top_k: int = 20) -> List[str]:
    tool = KeywordExtractionTool()
    result = tool._run(text, top_k)
    if result.startswith("Error:"):
        return []
    return result.split(", ")
