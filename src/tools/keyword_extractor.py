"""
Enhanced keyword extraction tool with advanced NLP capabilities.
"""

import logging
import re
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter
from dataclasses import dataclass

import nltk
import spacy
from textblob import TextBlob
from transformers import pipeline
from crewai_tools import BaseTool

from config.settings import settings


logger = logging.getLogger(__name__)


@dataclass
class KeywordResult:
    """Result of keyword extraction."""
    keywords: List[str]
    keyword_scores: Dict[str, float]
    entities: List[Dict[str, str]]
    topics: List[str]
    sentiment: Optional[Dict[str, float]] = None
    processing_time: Optional[float] = None


class KeywordExtractionTool(BaseTool):
    """Advanced keyword extraction with multiple NLP approaches."""
    
    name: str = "keyword_extractor"
    description: str = "Extracts keywords, entities, and topics from text using advanced NLP"
    
    def __init__(self):
        super().__init__()
        self._setup_nltk()
        self._setup_spacy()
        self._setup_transformers()
        
        # Common stop words and filter words
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.filter_words = {
            'paper', 'study', 'research', 'author', 'authors', 'article',
            'section', 'figure', 'table', 'abstract', 'conclusion',
            'introduction', 'method', 'methods', 'result', 'results',
            'discussion', 'references', 'citation', 'citations',
            'page', 'pages', 'et', 'al', 'doi', 'http', 'https', 'www'
        }
        
        # Domain-specific terms for AI/ML publications
        self.domain_keywords = {
            'machine_learning': [
                'neural', 'network', 'deep', 'learning', 'model', 'training',
                'algorithm', 'classification', 'regression', 'clustering',
                'supervised', 'unsupervised', 'reinforcement'
            ],
            'ai_agents': [
                'agent', 'agents', 'multi-agent', 'autonomous', 'intelligent',
                'collaboration', 'coordination', 'workflow', 'orchestration'
            ],
            'nlp': [
                'language', 'text', 'nlp', 'tokenization', 'embedding',
                'transformer', 'bert', 'gpt', 'attention', 'sequence'
            ],
            'computer_vision': [
                'vision', 'image', 'detection', 'recognition', 'segmentation',
                'cnn', 'convolution', 'feature', 'extraction'
            ]
        }
    
    def _setup_nltk(self):
        """Setup NLTK resources."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
    
    def _setup_spacy(self):
        """Setup spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _setup_transformers(self):
        """Setup transformer models for advanced analysis."""
        try:
            if not settings.debug:  # Only load heavy models in production
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                self.topic_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
            else:
                self.sentiment_analyzer = None
                self.topic_classifier = None
        except Exception as e:
            logger.warning(f"Could not load transformer models: {e}")
            self.sentiment_analyzer = None
            self.topic_classifier = None
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', '', text)
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^a-zA-Z0-9\\s\\-_]', ' ', text)
        text = re.sub(r'\\s+', ' ', text)
        
        return text.strip()
    
    def _extract_nltk_keywords(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Extract keywords using NLTK."""
        try:
            tokens = nltk.word_tokenize(text.lower())
            
            # Filter tokens
            tokens = [
                token for token in tokens
                if (len(token) > 2 and
                    token.isalpha() and
                    token not in self.stop_words and
                    token not in self.filter_words)
            ]
            
            # POS tagging to keep only nouns and adjectives
            pos_tags = nltk.pos_tag(tokens)
            filtered_tokens = [
                word for word, pos in pos_tags
                if pos.startswith(('NN', 'JJ', 'VB'))
            ]
            
            # Count frequencies
            freq_dist = Counter(filtered_tokens)
            
            # Score keywords by frequency and length
            scored_keywords = []
            for word, freq in freq_dist.items():
                score = freq * (1 + len(word) / 10)  # Slight preference for longer words
                scored_keywords.append((word, score))
            
            return sorted(scored_keywords, key=lambda x: x[1], reverse=True)[:top_k]
            
        except Exception as e:
            logger.error(f"NLTK keyword extraction failed: {e}")
            return []
    
    def _extract_spacy_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities using spaCy."""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'WORK_OF_ART', 'EVENT']:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'confidence': 1.0  # spaCy doesn't provide confidence scores
                    })
            
            return entities
            
        except Exception as e:
            logger.error(f"spaCy entity extraction failed: {e}")
            return []
    
    def _extract_domain_keywords(self, text: str) -> List[str]:
        """Extract domain-specific AI/ML keywords."""
        text_lower = text.lower()
        found_keywords = []
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
        
        return list(set(found_keywords))
    
    def _analyze_sentiment(self, text: str) -> Optional[Dict[str, float]]:
        """Analyze sentiment using transformer model."""
        if not self.sentiment_analyzer:
            # Fallback to TextBlob
            try:
                blob = TextBlob(text)
                return {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
                return None
        
        try:
            # Use first 512 characters to avoid token limits
            sample_text = text[:512]
            results = self.sentiment_analyzer(sample_text)
            
            sentiment_scores = {}
            for result in results[0]:
                sentiment_scores[result['label'].lower()] = result['score']
            
            return sentiment_scores
            
        except Exception as e:
            logger.error(f"Transformer sentiment analysis failed: {e}")
            return None
    
    def _classify_topics(self, text: str) -> List[str]:
        """Classify text into predefined topics."""
        if not self.topic_classifier:
            return []
        
        try:
            # Define candidate topics for AI/ML publications
            candidate_labels = [
                "machine learning",
                "artificial intelligence",
                "natural language processing",
                "computer vision",
                "robotics",
                "data science",
                "deep learning",
                "reinforcement learning",
                "neural networks",
                "multi-agent systems"
            ]
            
            # Use first 1000 characters to avoid token limits
            sample_text = text[:1000]
            result = self.topic_classifier(sample_text, candidate_labels)
            
            # Return topics with confidence > 0.5
            topics = [
                label for label, score in zip(result['labels'], result['scores'])
                if score > 0.5
            ]
            
            return topics[:3]  # Return top 3 topics
            
        except Exception as e:
            logger.error(f"Topic classification failed: {e}")
            return []
    
    def _run(self, text: str, top_k: int = 20, include_entities: bool = True, 
            include_sentiment: bool = False, include_topics: bool = False) -> str:
        """Extract keywords and other information from text."""
        import time
        start_time = time.time()
        
        if not text or len(text.strip()) < 10:
            return "Error: Text is too short for meaningful keyword extraction"
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        try:
            # Extract keywords using NLTK
            nltk_keywords = self._extract_nltk_keywords(cleaned_text, top_k)
            keywords = [kw for kw, score in nltk_keywords]
            keyword_scores = dict(nltk_keywords)
            
            # Add domain-specific keywords
            domain_keywords = self._extract_domain_keywords(cleaned_text)
            for dk in domain_keywords:
                if dk not in keyword_scores:
                    keywords.append(dk)
                    keyword_scores[dk] = 1.0  # Base score for domain keywords
            
            # Extract entities if requested
            entities = []
            if include_entities:
                entities = self._extract_spacy_entities(text)
            
            # Analyze sentiment if requested
            sentiment = None
            if include_sentiment:
                sentiment = self._analyze_sentiment(text)
            
            # Classify topics if requested
            topics = []
            if include_topics:
                topics = self._classify_topics(text)
            
            processing_time = time.time() - start_time
            
            # Create result object
            result = KeywordResult(
                keywords=keywords,
                keyword_scores=keyword_scores,
                entities=entities,
                topics=topics,
                sentiment=sentiment,
                processing_time=processing_time
            )
            
            # For CrewAI tool compatibility, return keywords as comma-separated string
            return ", ".join(keywords)
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return f"Error: Keyword extraction failed - {str(e)}"
    
    def extract_detailed(self, text: str, **kwargs) -> KeywordResult:
        """Extract detailed keyword information (returns full result object)."""
        import time
        start_time = time.time()
        
        if not text or len(text.strip()) < 10:
            return KeywordResult(
                keywords=[],
                keyword_scores={},
                entities=[],
                topics=[],
                sentiment=None
            )
        
        cleaned_text = self._clean_text(text)
        
        # Extract all information
        nltk_keywords = self._extract_nltk_keywords(cleaned_text, kwargs.get('top_k', 20))
        keywords = [kw for kw, score in nltk_keywords]
        keyword_scores = dict(nltk_keywords)
        
        domain_keywords = self._extract_domain_keywords(cleaned_text)
        for dk in domain_keywords:
            if dk not in keyword_scores:
                keywords.append(dk)
                keyword_scores[dk] = 1.0
        
        entities = self._extract_spacy_entities(text)
        sentiment = self._analyze_sentiment(text)
        topics = self._classify_topics(text)
        
        processing_time = time.time() - start_time
        
        return KeywordResult(
            keywords=keywords,
            keyword_scores=keyword_scores,
            entities=entities,
            topics=topics,
            sentiment=sentiment,
            processing_time=processing_time
        )


# Legacy function for backward compatibility
def extract_keywords(text: str, top_k: int = 20) -> List[str]:
    """Legacy function for backward compatibility."""
    tool = KeywordExtractionTool()
    result = tool._run(text, top_k)
    if result.startswith("Error:"):
        return []
    return result.split(", ")
