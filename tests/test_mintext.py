"""Tests for mintext module (Text and Corpus classes)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oet_core.mintext import Text, Corpus
from oet_core.utils import SQLiteHelper


class TestText:
    """Test cases for Text class."""

    def test_init(self):
        """Test Text initialization."""
        text = Text("Hello world")
        assert text.content == "Hello world"
        assert text.metadata == {}
        assert len(text) == 11

    def test_init_with_metadata(self):
        """Test Text initialization with metadata."""
        metadata = {"source": "test", "id": 1}
        text = Text("Hello", metadata=metadata)
        assert text.metadata == metadata

    def test_init_invalid(self):
        """Test Text initialization with invalid input."""
        try:
            Text(123)
            assert False, "Should raise TypeError"
        except TypeError:
            pass

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        text = Text("Hello world! How are you?")
        tokens = text.tokenize()
        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens

    def test_tokenize_no_lowercase(self):
        """Test tokenization without lowercasing."""
        text = Text("Hello World")
        tokens = text.tokenize(lowercase=False)
        assert "Hello" in tokens
        assert "World" in tokens

    def test_tokenize_keep_punctuation(self):
        """Test tokenization keeping punctuation."""
        text = Text("Hello, world!")
        tokens = text.tokenize(remove_punctuation=False)
        # Should have punctuation attached
        assert any("," in t or "!" in t for t in tokens)

    def test_tokenize_min_length(self):
        """Test tokenization with minimum length."""
        text = Text("I am a big cat")
        tokens = text.tokenize(min_length=3)
        assert "big" in tokens
        assert "cat" in tokens
        assert "i" not in tokens
        assert "am" not in tokens

    def test_get_tokens_caching(self):
        """Test token caching."""
        text = Text("Hello world")
        tokens1 = text.get_tokens()
        tokens2 = text.get_tokens()
        # Should be the same cached instance
        assert tokens1 is tokens2

    def test_get_tokens_refresh(self):
        """Test token refresh."""
        text = Text("Hello world")
        tokens1 = text.get_tokens(lowercase=True)
        tokens2 = text.get_tokens(refresh=True, lowercase=False)
        assert "hello" in tokens1
        assert "Hello" in tokens2

    def test_frequencies(self):
        """Test frequency computation."""
        text = Text("the cat and the dog")
        freq = text.frequencies()
        assert freq["the"] == 2
        assert freq["cat"] == 1
        assert freq["dog"] == 1

    def test_frequencies_caching(self):
        """Test frequency caching."""
        text = Text("Hello world")
        freq1 = text.frequencies()
        freq2 = text.frequencies()
        assert freq1 is freq2

    def test_top_words(self):
        """Test top words extraction."""
        text = Text("the cat and the dog and the bird")
        top = text.top_words(n=2)
        assert len(top) == 2
        assert top[0] == ("the", 3)
        assert top[1] == ("and", 2)

    def test_entropy_empty(self):
        """Test entropy on empty text."""
        text = Text("")
        assert text.entropy() == 0.0

    def test_entropy_single_token(self):
        """Test entropy with single unique token."""
        text = Text("hello hello hello")
        entropy = text.entropy()
        assert entropy == 0.0  # No uncertainty

    def test_entropy_diverse(self):
        """Test entropy with diverse tokens."""
        text = Text("apple banana cherry date")
        entropy = text.entropy()
        assert entropy > 0.0  # Should have uncertainty
        # Should be 2.0 for 4 equally likely outcomes
        assert 1.9 < entropy < 2.1

    def test_sentiment_positive(self):
        """Test sentiment with positive text."""
        text = Text("This is great and wonderful! I love it.")
        result = text.sentiment()
        assert result["score"] > 0
        assert result["positive"] > 0

    def test_sentiment_negative(self):
        """Test sentiment with negative text."""
        text = Text("This is terrible and awful. I hate it.")
        result = text.sentiment()
        assert result["score"] < 0
        assert result["negative"] > 0

    def test_sentiment_neutral(self):
        """Test sentiment with neutral text."""
        text = Text("The chair is near the table.")
        result = text.sentiment()
        assert result["score"] == 0
        assert result["neutral"] > 0

    def test_sentiment_mixed(self):
        """Test sentiment with mixed text."""
        text = Text("It was good at first but then became terrible.")
        result = text.sentiment()
        # Should have both positive and negative
        assert result["positive"] > 0
        assert result["negative"] > 0

    def test_vectorize_default(self):
        """Test vectorization with default vocabulary."""
        text = Text("cat dog cat")
        vector = text.vectorize()
        # Should have 2 dimensions (cat, dog)
        assert len(vector) == 2
        # Check counts
        vec_list = vector.tolist() if hasattr(vector, "tolist") else vector
        assert 2.0 in vec_list  # cat appears twice
        assert 1.0 in vec_list  # dog appears once

    def test_vectorize_custom_vocab(self):
        """Test vectorization with custom vocabulary."""
        text = Text("cat dog")
        vocab = ["cat", "dog", "bird"]
        vector = text.vectorize(vocabulary=vocab)
        vec_list = vector.tolist() if hasattr(vector, "tolist") else vector
        assert len(vec_list) == 3
        assert vec_list[2] == 0.0  # bird not in text

    def test_vectorize_binary(self):
        """Test binary vectorization."""
        text = Text("cat cat dog")
        vocab = ["cat", "dog"]
        vector = text.vectorize(vocabulary=vocab, binary=True)
        vec_list = vector.tolist() if hasattr(vector, "tolist") else vector
        assert vec_list[0] == 1.0  # cat present
        assert vec_list[1] == 1.0  # dog present

    def test_stats(self):
        """Test comprehensive statistics."""
        text = Text("The quick brown fox jumps.")
        stats = text.stats()
        
        assert "char_count" in stats
        assert "token_count" in stats
        assert "unique_tokens" in stats
        assert "type_token_ratio" in stats
        assert "entropy" in stats
        assert "sentiment_score" in stats
        assert "avg_token_length" in stats
        
        assert stats["token_count"] == 5
        assert stats["unique_tokens"] == 5
        assert stats["type_token_ratio"] == 1.0

    def test_to_dict(self):
        """Test dictionary export."""
        metadata = {"id": 1}
        text = Text("Hello", metadata=metadata)
        data = text.to_dict()
        
        assert data["content"] == "Hello"
        assert data["metadata"] == metadata

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "content": "Hello world",
            "metadata": {"source": "test"}
        }
        text = Text.from_dict(data)
        
        assert text.content == "Hello world"
        assert text.metadata == {"source": "test"}

    def test_repr(self):
        """Test string representation."""
        text = Text("Short text")
        repr_str = repr(text)
        assert "Text" in repr_str
        assert "chars=" in repr_str


class TestCorpus:
    """Test cases for Corpus class."""

    def test_init_empty(self):
        """Test empty corpus initialization."""
        corpus = Corpus()
        assert len(corpus) == 0
        assert corpus.texts == []

    def test_init_with_texts(self):
        """Test corpus initialization with texts."""
        texts = [Text("Hello"), Text("World")]
        corpus = Corpus(texts)
        assert len(corpus) == 2

    def test_init_invalid(self):
        """Test corpus initialization with invalid input."""
        try:
            Corpus(texts="not a list")
            assert False, "Should raise TypeError"
        except TypeError:
            pass

    def test_add(self):
        """Test adding text to corpus."""
        corpus = Corpus()
        text = Text("Hello")
        corpus.add(text)
        assert len(corpus) == 1
        assert corpus[0] is text

    def test_add_invalid(self):
        """Test adding invalid object."""
        corpus = Corpus()
        try:
            corpus.add("not a text object")
            assert False, "Should raise TypeError"
        except TypeError:
            pass

    def test_add_from_string(self):
        """Test creating and adding text from string."""
        corpus = Corpus()
        text = corpus.add_from_string("Hello world", metadata={"id": 1})
        
        assert len(corpus) == 1
        assert isinstance(text, Text)
        assert text.content == "Hello world"
        assert text.metadata["id"] == 1

    def test_filter(self):
        """Test filtering corpus."""
        corpus = Corpus([
            Text("Hello", metadata={"lang": "en"}),
            Text("Bonjour", metadata={"lang": "fr"}),
            Text("Hola", metadata={"lang": "es"}),
        ])
        
        filtered = corpus.filter(lambda t: t.metadata.get("lang") == "en")
        assert len(filtered) == 1
        assert filtered[0].content == "Hello"

    def test_vocabulary(self):
        """Test vocabulary building."""
        corpus = Corpus([
            Text("the cat and the dog"),
            Text("the bird and the fish"),
        ])
        
        vocab = corpus.vocabulary()
        assert "the" in vocab
        assert "cat" in vocab
        assert "bird" in vocab
        assert len(vocab) == 6  # the, cat, dog, and, bird, fish

    def test_vocabulary_min_freq(self):
        """Test vocabulary with minimum frequency."""
        corpus = Corpus([
            Text("the cat"),
            Text("the dog"),
            Text("the bird"),
        ])
        
        vocab = corpus.vocabulary(min_freq=3)
        assert vocab == ["the"]  # Only "the" appears 3 times

    def test_vectorize_all(self):
        """Test vectorizing all texts."""
        corpus = Corpus([
            Text("cat dog"),
            Text("dog bird"),
            Text("cat bird"),
        ])
        
        vectors = corpus.vectorize_all()
        # Should have 3 vectors (one per text)
        assert len(vectors) == 3
        
        # Each vector should have same dimensionality
        if hasattr(vectors, "shape"):  # numpy array
            assert vectors.shape[1] == 3  # vocab size
        else:  # list of lists
            assert len(vectors[0]) == 3

    def test_vectorize_all_custom_vocab(self):
        """Test vectorizing with custom vocabulary."""
        corpus = Corpus([
            Text("cat dog"),
            Text("dog bird"),
        ])
        
        vocab = ["cat", "dog", "bird", "fish"]
        vectors = corpus.vectorize_all(vocabulary=vocab)
        
        # Should use all 4 vocab words
        if hasattr(vectors, "shape"):
            assert vectors.shape[1] == 4
        else:
            assert len(vectors[0]) == 4

    def test_vectorize_all_binary(self):
        """Test binary vectorization."""
        corpus = Corpus([
            Text("cat cat dog"),
            Text("bird"),
        ])
        
        vectors = corpus.vectorize_all(binary=True)
        vec_list = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        
        # First text should have 1s for cat and dog
        assert 1.0 in vec_list[0]
        # Values should be 0 or 1
        for vec in vec_list:
            for val in vec:
                assert val in [0.0, 1.0]

    def test_aggregate_stats_empty(self):
        """Test aggregate stats on empty corpus."""
        corpus = Corpus()
        stats = corpus.aggregate_stats()
        
        assert stats["text_count"] == 0
        assert stats["total_tokens"] == 0

    def test_aggregate_stats(self):
        """Test aggregate statistics."""
        corpus = Corpus([
            Text("hello world"),
            Text("hello there friend"),
        ])
        
        stats = corpus.aggregate_stats()
        
        assert stats["text_count"] == 2
        assert stats["total_tokens"] == 5  # 2 + 3
        assert stats["vocabulary_size"] == 4  # hello, world, there, friend
        assert stats["avg_tokens_per_text"] == 2.5

    def test_save_and_load_db(self):
        """Test saving and loading corpus to/from database."""
        # Create corpus
        corpus = Corpus([
            Text("Hello world", metadata={"id": 1}),
            Text("Goodbye world", metadata={"id": 2}),
        ])
        
        # Save to database
        db = SQLiteHelper(":memory:")
        corpus.save_to_db(db, table="test_corpus")
        
        # Load from database
        loaded = Corpus.load_from_db(db, table="test_corpus")
        
        assert len(loaded) == 2
        assert loaded[0].content == "Hello world"
        assert loaded[0].metadata["id"] == 1
        assert loaded[1].content == "Goodbye world"
        
        db.close()

    def test_save_db_empty_metadata(self):
        """Test saving texts with no metadata."""
        corpus = Corpus([Text("Hello")])
        
        db = SQLiteHelper(":memory:")
        corpus.save_to_db(db)
        
        loaded = Corpus.load_from_db(db)
        assert len(loaded) == 1
        assert loaded[0].metadata == {}
        
        db.close()

    def test_load_db_with_where(self):
        """Test loading corpus with WHERE clause."""
        corpus = Corpus([
            Text("First", metadata={"priority": "high"}),
            Text("Second", metadata={"priority": "low"}),
            Text("Third", metadata={"priority": "high"}),
        ])
        
        db = SQLiteHelper(":memory:")
        corpus.save_to_db(db)
        
        # Load all (can't actually filter by metadata in simple query,
        # but test the parameter passing)
        loaded = Corpus.load_from_db(db, where="id > ?", params=(1,))
        assert len(loaded) == 2  # Should skip first row
        
        db.close()

    def test_getitem(self):
        """Test indexing corpus."""
        texts = [Text("First"), Text("Second"), Text("Third")]
        corpus = Corpus(texts)
        
        assert corpus[0].content == "First"
        assert corpus[1].content == "Second"
        assert corpus[2].content == "Third"

    def test_iter(self):
        """Test iterating over corpus."""
        texts = [Text("First"), Text("Second")]
        corpus = Corpus(texts)
        
        collected = [t.content for t in corpus]
        assert collected == ["First", "Second"]

    def test_repr(self):
        """Test string representation."""
        corpus = Corpus([Text("Hello"), Text("World")])
        repr_str = repr(corpus)
        assert "Corpus" in repr_str
        assert "2" in repr_str


def run_tests():
    """Run all tests and report results."""
    test_classes = [TestText, TestCorpus]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"PASS {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"FAIL {method_name}: {e}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Test Summary")
    print('='*60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
        return False
    else:
        print(f"\nAll tests passed!")
        return True


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
