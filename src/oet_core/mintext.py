"""MinText: Lightweight text analysis primitives for oet_core.

Provides simple, dependency-free text analysis for research workflows including
tokenization, frequency statistics, entropy, sentiment, and vectorization.
"""

from __future__ import annotations

import json
import math
import re
import string
from typing import Any, Dict, List, Optional, Tuple, Union

# Import SQLiteHelper for persistence (avoid circular import by importing at function level)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils import SQLiteHelper

# Verbose logging support
_VERBOSE_LOGGING = False


def set_mintext_verbose_logging(enabled: bool) -> None:
    """Enable or disable verbose logging for mintext module."""
    global _VERBOSE_LOGGING
    _VERBOSE_LOGGING = bool(enabled)


def _log_if_verbose(message: str) -> None:
    """Log message if verbose logging is enabled."""
    if _VERBOSE_LOGGING:
        from .utils import log
        log(message, level="info")


class Text:
    """Lightweight text analysis with tokenization, frequency, entropy, and sentiment.
    
    Provides simple, dependency-free text analysis primitives for research workflows.
    No external NLP frameworks required - uses Python stdlib and optional NumPy.
    """

    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Initialize a Text object.
        
        Parameters
        ----------
        content:
            Raw text content to analyze.
        metadata:
            Optional metadata dictionary (id, source, timestamp, etc.).
        """
        _log_if_verbose(f"Text.__init__ called with content length={len(content)}")

        if not isinstance(content, str):
            raise TypeError("content must be a string")

        self.content = content
        self.metadata = metadata or {}
        self._tokens: Optional[List[str]] = None
        self._frequencies: Optional[Dict[str, int]] = None

    def tokenize(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        min_length: int = 1,
    ) -> List[str]:
        """Tokenize text into words.
        
        Parameters
        ----------
        lowercase:
            Convert tokens to lowercase.
        remove_punctuation:
            Strip punctuation from tokens.
        min_length:
            Minimum token length to include.
        
        Returns
        -------
        List[str]
            List of tokens.
        """
        _log_if_verbose("Text.tokenize called")

        tokens = self.content.split()

        if lowercase:
            tokens = [t.lower() for t in tokens]

        if remove_punctuation:
            # Remove punctuation from each token
            translator = str.maketrans("", "", string.punctuation)
            tokens = [t.translate(translator) for t in tokens]

        # Filter by minimum length and remove empty strings
        tokens = [t for t in tokens if len(t) >= min_length]

        self._tokens = tokens
        return tokens

    def get_tokens(self, refresh: bool = False, **tokenize_kwargs) -> List[str]:
        """Get cached tokens or tokenize if needed.
        
        Parameters
        ----------
        refresh:
            Force re-tokenization.
        **tokenize_kwargs:
            Arguments passed to tokenize() if needed.
        
        Returns
        -------
        List[str]
            List of tokens.
        """
        if self._tokens is None or refresh:
            return self.tokenize(**tokenize_kwargs)
        return self._tokens

    def frequencies(self, refresh: bool = False, **tokenize_kwargs) -> Dict[str, int]:
        """Compute token frequency distribution.
        
        Parameters
        ----------
        refresh:
            Force re-computation.
        **tokenize_kwargs:
            Arguments passed to tokenize() if needed.
        
        Returns
        -------
        Dict[str, int]
            Token frequency mapping.
        """
        _log_if_verbose("Text.frequencies called")

        if self._frequencies is not None and not refresh:
            return self._frequencies

        tokens = self.get_tokens(refresh=refresh, **tokenize_kwargs)
        freq: Dict[str, int] = {}

        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

        self._frequencies = freq
        return freq

    def top_words(self, n: int = 10, **tokenize_kwargs) -> List[Tuple[str, int]]:
        """Get the n most frequent tokens.
        
        Parameters
        ----------
        n:
            Number of top tokens to return.
        **tokenize_kwargs:
            Arguments passed to tokenize() if needed.
        
        Returns
        -------
        List[Tuple[str, int]]
            List of (token, count) tuples sorted by frequency.
        """
        freq = self.frequencies(**tokenize_kwargs)
        return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]

    def entropy(self, **tokenize_kwargs) -> float:
        """Calculate Shannon entropy of token distribution.
        
        Higher entropy indicates more diverse/unpredictable text.
        
        Parameters
        ----------
        **tokenize_kwargs:
            Arguments passed to tokenize() if needed.
        
        Returns
        -------
        float
            Shannon entropy in bits.
        """
        _log_if_verbose("Text.entropy called")

        freq = self.frequencies(**tokenize_kwargs)
        if not freq:
            return 0.0

        total = sum(freq.values())
        entropy_val = 0.0

        for count in freq.values():
            if count > 0:
                p = count / total
                entropy_val -= p * math.log2(p)

        return entropy_val

    def sentiment(self, **tokenize_kwargs) -> Dict[str, Any]:
        """Simple lexicon-based sentiment analysis.
        
        Uses a minimal built-in lexicon for dependency-free sentiment scoring.
        
        Parameters
        ----------
        **tokenize_kwargs:
            Arguments passed to tokenize() if needed.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with 'score', 'positive', 'negative', and 'neutral' counts.
        """
        _log_if_verbose("Text.sentiment called")

        # Minimal sentiment lexicon (expandable)
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "positive", "love", "best", "better", "happy", "joy", "like",
            "success", "successful", "perfect", "superior", "awesome", "brilliant",
        }

        negative_words = {
            "bad", "terrible", "awful", "horrible", "worst", "worse", "hate",
            "negative", "poor", "fail", "failure", "sad", "angry", "wrong",
            "inferior", "disappointing", "disappointed", "useless", "pathetic",
        }

        tokens = self.get_tokens(**tokenize_kwargs)
        positive_count = 0
        negative_count = 0

        for token in tokens:
            if token in positive_words:
                positive_count += 1
            elif token in negative_words:
                negative_count += 1

        neutral_count = len(tokens) - positive_count - negative_count
        score = positive_count - negative_count

        return {
            "score": score,
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
        }

    def vectorize(
        self,
        vocabulary: Optional[List[str]] = None,
        binary: bool = False,
    ) -> Union[List[float], Any]:
        """Convert text to term frequency vector.
        
        Parameters
        ----------
        vocabulary:
            Fixed vocabulary list. If None, uses tokens from this text.
        binary:
            If True, use binary (0/1) instead of counts.
        
        Returns
        -------
        List[float] or numpy.ndarray
            Vector representation (numpy array if numpy available, else list).
        """
        _log_if_verbose("Text.vectorize called")

        freq = self.frequencies()

        if vocabulary is None:
            vocabulary = sorted(freq.keys())

        # Build vector
        vector = []
        for word in vocabulary:
            count = freq.get(word, 0)
            if binary:
                vector.append(1.0 if count > 0 else 0.0)
            else:
                vector.append(float(count))

        # Try to return numpy array if available
        try:
            import numpy as np  # type: ignore
            return np.array(vector)
        except ImportError:
            return vector

    def stats(self, **tokenize_kwargs) -> Dict[str, Any]:
        """Compute comprehensive text statistics.
        
        Parameters
        ----------
        **tokenize_kwargs:
            Arguments passed to tokenize() if needed.
        
        Returns
        -------
        Dict[str, Any]
            Statistics including token count, unique tokens, entropy, etc.
        """
        tokens = self.get_tokens(**tokenize_kwargs)
        freq = self.frequencies(**tokenize_kwargs)
        sentiment_data = self.sentiment(**tokenize_kwargs)

        return {
            "char_count": len(self.content),
            "token_count": len(tokens),
            "unique_tokens": len(freq),
            "type_token_ratio": len(freq) / len(tokens) if tokens else 0.0,
            "entropy": self.entropy(**tokenize_kwargs),
            "sentiment_score": sentiment_data["score"],
            "avg_token_length": sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export text and metadata to dictionary for serialization.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with content and metadata.
        """
        return {
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Text":
        """Create Text object from dictionary.
        
        Parameters
        ----------
        data:
            Dictionary with 'content' and optional 'metadata' keys.
        
        Returns
        -------
        Text
            New Text instance.
        """
        return cls(
            content=data["content"],
            metadata=data.get("metadata"),
        )

    def __len__(self) -> int:
        """Return character count."""
        return len(self.content)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Text(chars={len(self.content)}, preview='{preview}')"


class Corpus:
    """Collection of Text objects with batch operations and persistence.
    
    Provides aggregation, filtering, and SQLite-based storage for text collections.
    """

    def __init__(self, texts: Optional[List[Text]] = None) -> None:
        """Initialize a Corpus.
        
        Parameters
        ----------
        texts:
            Optional initial list of Text objects.
        """
        _log_if_verbose(f"Corpus.__init__ called with {len(texts) if texts else 0} texts")

        if texts is not None and not isinstance(texts, list):
            raise TypeError("texts must be a list")

        self.texts = texts or []

    def add(self, text: Text) -> None:
        """Add a Text object to the corpus.
        
        Parameters
        ----------
        text:
            Text object to add.
        """
        if not isinstance(text, Text):
            raise TypeError("text must be a Text instance")
        self.texts.append(text)

    def add_from_string(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Text:
        """Create and add a Text object from string content.
        
        Parameters
        ----------
        content:
            Raw text content.
        metadata:
            Optional metadata.
        
        Returns
        -------
        Text
            The created Text object.
        """
        text = Text(content, metadata)
        self.add(text)
        return text

    def filter(self, predicate) -> "Corpus":
        """Filter corpus by predicate function.
        
        Parameters
        ----------
        predicate:
            Function that takes a Text object and returns bool.
        
        Returns
        -------
        Corpus
            New corpus with filtered texts.
        """
        filtered_texts = [t for t in self.texts if predicate(t)]
        return Corpus(filtered_texts)

    def vocabulary(self, min_freq: int = 1, **tokenize_kwargs) -> List[str]:
        """Build vocabulary from all texts.
        
        Parameters
        ----------
        min_freq:
            Minimum frequency threshold for inclusion.
        **tokenize_kwargs:
            Arguments passed to tokenize().
        
        Returns
        -------
        List[str]
            Sorted list of tokens appearing at least min_freq times.
        """
        _log_if_verbose("Corpus.vocabulary called")

        term_counts: Dict[str, int] = {}

        for text in self.texts:
            freq = text.frequencies(**tokenize_kwargs)
            for term, count in freq.items():
                term_counts[term] = term_counts.get(term, 0) + count

        vocab = [term for term, count in term_counts.items() if count >= min_freq]
        return sorted(vocab)

    def vectorize_all(
        self,
        vocabulary: Optional[List[str]] = None,
        binary: bool = False,
        **tokenize_kwargs,
    ) -> Union[List[List[float]], Any]:
        """Vectorize all texts using shared vocabulary.
        
        Parameters
        ----------
        vocabulary:
            Shared vocabulary. If None, builds from corpus.
        binary:
            Use binary encoding.
        **tokenize_kwargs:
            Arguments passed to tokenize().
        
        Returns
        -------
        List[List[float]] or numpy.ndarray
            Matrix of vectors (numpy 2D array if available, else list of lists).
        """
        _log_if_verbose("Corpus.vectorize_all called")

        if vocabulary is None:
            vocabulary = self.vocabulary(**tokenize_kwargs)

        vectors = []
        for text in self.texts:
            vec = text.vectorize(vocabulary=vocabulary, binary=binary)
            # Convert numpy array to list if needed for consistency
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            vectors.append(vec)

        # Try to return as numpy 2D array
        try:
            import numpy as np  # type: ignore
            return np.array(vectors)
        except ImportError:
            return vectors

    def aggregate_stats(self, **tokenize_kwargs) -> Dict[str, Any]:
        """Compute aggregate statistics across corpus.
        
        Parameters
        ----------
        **tokenize_kwargs:
            Arguments passed to tokenize().
        
        Returns
        -------
        Dict[str, Any]
            Aggregate statistics.
        """
        if not self.texts:
            return {
                "text_count": 0,
                "total_tokens": 0,
                "total_chars": 0,
                "avg_tokens_per_text": 0.0,
                "avg_chars_per_text": 0.0,
                "vocabulary_size": 0,
            }

        total_tokens = 0
        total_chars = 0

        for text in self.texts:
            tokens = text.get_tokens(**tokenize_kwargs)
            total_tokens += len(tokens)
            total_chars += len(text.content)

        vocab_size = len(self.vocabulary(**tokenize_kwargs))

        return {
            "text_count": len(self.texts),
            "total_tokens": total_tokens,
            "total_chars": total_chars,
            "avg_tokens_per_text": total_tokens / len(self.texts),
            "avg_chars_per_text": total_chars / len(self.texts),
            "vocabulary_size": vocab_size,
        }

    def save_to_db(self, db: "SQLiteHelper", table: str = "corpus") -> None:
        """Persist corpus to SQLite database.
        
        Parameters
        ----------
        db:
            SQLiteHelper instance.
        table:
            Table name to store texts.
        """
        _log_if_verbose(f"Corpus.save_to_db called with table={table}")

        # Create table if needed
        schema = {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "content": "TEXT NOT NULL",
            "metadata": "TEXT",  # JSON-encoded
        }
        db.create_table(table, schema)

        # Insert texts
        rows = []
        for text in self.texts:
            metadata_json = json.dumps(text.metadata) if text.metadata else None
            rows.append((text.content, metadata_json))

        if rows:
            db.bulk_insert(table, ["content", "metadata"], rows)

    @classmethod
    def load_from_db(
        cls,
        db: "SQLiteHelper",
        table: str = "corpus",
        where: Optional[str] = None,
        params: Optional[Tuple] = None,
    ) -> "Corpus":
        """Load corpus from SQLite database.
        
        Parameters
        ----------
        db:
            SQLiteHelper instance.
        table:
            Table name to load from.
        where:
            Optional WHERE clause.
        params:
            Parameters for WHERE clause.
        
        Returns
        -------
        Corpus
            New Corpus instance with loaded texts.
        """
        _log_if_verbose(f"Corpus.load_from_db called with table={table}")

        query = f"SELECT content, metadata FROM {table}"
        if where:
            query += f" WHERE {where}"

        rows = db.fetch_all(query, params)

        texts = []
        for row in rows:
            content = row["content"]
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            texts.append(Text(content, metadata))

        return cls(texts)

    def __len__(self) -> int:
        """Return number of texts in corpus."""
        return len(self.texts)

    def __getitem__(self, index: int) -> Text:
        """Get text by index."""
        return self.texts[index]

    def __iter__(self):
        """Iterate over texts."""
        return iter(self.texts)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Corpus(texts={len(self.texts)})"
