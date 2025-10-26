# MinText Guide

**MinText** is a lightweight, dependency-free text analysis module for `oet-core`. It provides simple primitives for tokenization, frequency analysis, entropy, sentiment, and vectorization—perfect for research workflows that need basic text processing without heavy NLP frameworks.

## Philosophy

- **Zero required dependencies**: Uses Python stdlib only (optional NumPy for efficient arrays)
- **Minimal complexity**: Simple, readable implementations
- **Research-friendly**: Easy to understand and modify for specific needs
- **Persistence**: Built-in SQLite integration for storing text collections

## Quick Start

```python
from oet_core import Text, Corpus

# Analyze a single text
text = Text("The quick brown fox jumps over the lazy dog.")
tokens = text.tokenize()
print(f"Tokens: {tokens}")
print(f"Top words: {text.top_words(5)}")
print(f"Entropy: {text.entropy():.2f} bits")

# Work with multiple texts
corpus = Corpus()
corpus.add_from_string("First document about data science.")
corpus.add_from_string("Second document about machine learning.")
print(f"Vocabulary: {corpus.vocabulary()}")
print(f"Stats: {corpus.aggregate_stats()}")
```

## The Text Class

The `Text` class represents a single document and provides methods for analysis.

### Creating Text Objects

```python
from oet_core import Text

# Basic text
text = Text("Hello world! This is a simple example.")

# With metadata
text = Text(
    "Research findings show positive results.",
    metadata={
        "id": "doc_001",
        "source": "journal_article",
        "year": 2024,
        "author": "Smith et al."
    }
)
```

### Tokenization

Convert text into words with various options:

```python
text = Text("Hello World! How are you doing today?")

# Default: lowercase, remove punctuation, min_length=1
tokens = text.tokenize()
# ['hello', 'world', 'how', 'are', 'you', 'doing', 'today']

# Keep original case
tokens = text.tokenize(lowercase=False)
# ['Hello', 'World', 'How', 'are', 'you', 'doing', 'today']

# Keep punctuation
tokens = text.tokenize(remove_punctuation=False)
# ['hello', 'world!', 'how', 'are', 'you', 'doing', 'today?']

# Filter short words
tokens = text.tokenize(min_length=4)
# ['hello', 'world', 'doing', 'today']

# Cached access
tokens = text.get_tokens()  # Uses cached result
tokens = text.get_tokens(refresh=True)  # Forces re-tokenization
```

### Frequency Analysis

Count token occurrences:

```python
text = Text("the cat and the dog and the bird")

# Get frequency dictionary
freq = text.frequencies()
# {'the': 3, 'cat': 1, 'and': 2, 'dog': 1, 'bird': 1}

# Top N most frequent words
top = text.top_words(3)
# [('the', 3), ('and', 2), ('cat', 1)]

# Find specific word frequency
freq = text.frequencies()
count = freq.get('the', 0)  # 3
```

### Entropy

Measure text diversity with Shannon entropy:

```python
# Repetitive text (low entropy)
text1 = Text("the the the the the")
print(text1.entropy())  # ~0.0 bits

# Diverse text (high entropy)
text2 = Text("quick brown fox jumps over lazy dog")
print(text2.entropy())  # ~2.8 bits

# Use case: detect duplicate or low-quality content
if text.entropy() < 1.0:
    print("Warning: Very repetitive text detected")
```

### Sentiment Analysis

Simple lexicon-based sentiment scoring:

```python
text1 = Text("This is a great and wonderful product! I love it.")
sentiment1 = text1.sentiment()
# {
#     'score': 3,           # positive - negative
#     'positive': 3,        # great, wonderful, love
#     'negative': 0,
#     'neutral': 7
# }

text2 = Text("This is terrible and awful. I hate it.")
sentiment2 = text2.sentiment()
# {
#     'score': -3,
#     'positive': 0,
#     'negative': 3,        # terrible, awful, hate
#     'neutral': 4
# }

# Classify sentiment
if sentiment1['score'] > 0:
    print("Positive sentiment")
elif sentiment1['score'] < 0:
    print("Negative sentiment")
else:
    print("Neutral sentiment")
```

**Note**: The sentiment lexicon is minimal and built-in. For more sophisticated sentiment analysis, consider extending the positive/negative word sets or using a dedicated sentiment library.

### Vectorization

Convert text to numerical vectors for machine learning:

```python
text = Text("machine learning is great")

# Auto vocabulary (from this text)
vector = text.vectorize()
# [1.0, 1.0, 1.0, 1.0] for ['great', 'is', 'learning', 'machine']

# Fixed vocabulary (useful for multiple texts)
vocab = ['machine', 'learning', 'data', 'great']
vector = text.vectorize(vocabulary=vocab)
# [1.0, 1.0, 0.0, 1.0]

# Binary encoding (presence/absence)
vector = text.vectorize(vocabulary=vocab, binary=True)
# [1.0, 1.0, 0.0, 1.0]

# Returns numpy array if NumPy is installed, otherwise list
```

### Statistics

Get comprehensive text statistics:

```python
text = Text("The quick brown fox jumps over the lazy dog")
stats = text.stats()
# {
#     'char_count': 43,
#     'token_count': 9,
#     'unique_tokens': 9,
#     'type_token_ratio': 1.0,
#     'entropy': 3.169925001442312,
#     'sentiment_score': 0,
#     'avg_token_length': 3.888888888888889
# }

# Type-token ratio indicates vocabulary diversity
# 1.0 = all unique words (very diverse)
# Low ratio = many repeated words
```

### Serialization

Save and load Text objects:

```python
# Export to dictionary
text = Text("Hello world", metadata={"id": 1})
data = text.to_dict()
# {'content': 'Hello world', 'metadata': {'id': 1}}

# Import from dictionary
restored = Text.from_dict(data)

# Use with JSON
import json
json_str = json.dumps(text.to_dict())
loaded = Text.from_dict(json.loads(json_str))
```

## The Corpus Class

The `Corpus` class manages collections of texts with batch operations.

### Creating and Building Corpus

```python
from oet_core import Corpus, Text

# Empty corpus
corpus = Corpus()

# Add Text objects
corpus.add(Text("First document"))
corpus.add(Text("Second document"))

# Add from strings
corpus.add_from_string("Third document", metadata={"id": 3})

# Initialize with texts
texts = [Text("Doc 1"), Text("Doc 2")]
corpus = Corpus(texts)

# Access
print(len(corpus))  # 2
print(corpus[0])    # First text
for text in corpus:
    print(text)
```

### Filtering

Filter corpus by conditions:

```python
corpus = Corpus()
corpus.add_from_string("Short", metadata={"category": "A"})
corpus.add_from_string("This is a longer document", metadata={"category": "B"})
corpus.add_from_string("Medium length text", metadata={"category": "A"})

# Filter by token count
long_docs = corpus.filter(lambda t: len(t.get_tokens()) > 3)

# Filter by metadata
category_a = corpus.filter(lambda t: t.metadata.get("category") == "A")

# Filter by content
contains_word = corpus.filter(lambda t: "longer" in t.content)

# Chain filters
result = corpus.filter(
    lambda t: len(t.get_tokens()) > 2
).filter(
    lambda t: t.metadata.get("category") == "A"
)
```

### Vocabulary Building

Extract unique terms across all documents:

```python
corpus = Corpus()
corpus.add_from_string("machine learning is great")
corpus.add_from_string("data science is awesome")
corpus.add_from_string("machine learning and data science")

# Get all unique tokens
vocab = corpus.vocabulary()
# ['and', 'awesome', 'data', 'great', 'is', 'learning', 'machine', 'science']

# Filter by minimum frequency (must appear at least N times)
common_vocab = corpus.vocabulary(min_freq=2)
# ['data', 'is', 'learning', 'machine', 'science']

# Custom tokenization
vocab = corpus.vocabulary(lowercase=True, min_length=3)
```

### Batch Vectorization

Convert all texts to vectors using shared vocabulary:

```python
corpus = Corpus()
corpus.add_from_string("machine learning")
corpus.add_from_string("data science")
corpus.add_from_string("machine learning and data science")

# Vectorize with corpus vocabulary
vectors = corpus.vectorize_all()
# Returns 2D array/matrix (numpy if available, else list of lists)
# Each row is a document vector

# Use custom vocabulary
vocab = ['machine', 'learning', 'data', 'science', 'deep']
vectors = corpus.vectorize_all(vocabulary=vocab)

# Binary encoding
binary_vectors = corpus.vectorize_all(binary=True)

# Use in machine learning
# vectors can be passed directly to sklearn, etc.
from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=2)
# clusters = kmeans.fit_predict(vectors)
```

### Aggregate Statistics

Get corpus-wide statistics:

```python
corpus = Corpus()
corpus.add_from_string("First document here")
corpus.add_from_string("Second document")
corpus.add_from_string("Third one")

stats = corpus.aggregate_stats()
# {
#     'text_count': 3,
#     'total_tokens': 7,
#     'total_chars': 45,
#     'avg_tokens_per_text': 2.333...,
#     'avg_chars_per_text': 15.0,
#     'vocabulary_size': 7
# }

# Use for corpus overview
print(f"Corpus contains {stats['text_count']} documents")
print(f"Average document length: {stats['avg_tokens_per_text']:.1f} tokens")
print(f"Vocabulary diversity: {stats['vocabulary_size']} unique terms")
```

### Persistence with SQLite

Save and load corpus to/from database:

```python
from oet_core import Corpus, SQLiteHelper

# Create corpus
corpus = Corpus()
corpus.add_from_string("Document 1", metadata={"id": 1, "tag": "important"})
corpus.add_from_string("Document 2", metadata={"id": 2, "tag": "draft"})

# Save to database
db = SQLiteHelper("my_corpus.db")
corpus.save_to_db(db, table="documents")

# Load entire corpus
loaded = Corpus.load_from_db(db, table="documents")

# Load with filter
important_docs = Corpus.load_from_db(
    db,
    table="documents",
    where="json_extract(metadata, '$.tag') = ?",
    params=("important",)
)

# The database schema is:
# CREATE TABLE documents (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     content TEXT NOT NULL,
#     metadata TEXT  -- JSON-encoded
# )
```

## Advanced Examples

### Building a Document Similarity Tool

```python
from oet_core import Corpus

def cosine_similarity(vec1, vec2):
    """Simple cosine similarity."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = sum(a * a for a in vec1) ** 0.5
    mag2 = sum(b * b for b in vec2) ** 0.5
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0

# Build corpus
corpus = Corpus()
corpus.add_from_string("machine learning algorithms")
corpus.add_from_string("data science techniques")
corpus.add_from_string("deep learning neural networks")

# Vectorize
vectors = corpus.vectorize_all()

# Compare documents
similarity_0_1 = cosine_similarity(vectors[0], vectors[1])
similarity_0_2 = cosine_similarity(vectors[0], vectors[2])

print(f"Doc 0-1 similarity: {similarity_0_1:.3f}")
print(f"Doc 0-2 similarity: {similarity_0_2:.3f}")
```

### Content Quality Filter

```python
from oet_core import Text

def is_quality_content(text: Text) -> bool:
    """Check if text meets quality criteria."""
    stats = text.stats()
    
    # Filter criteria
    min_tokens = 10
    min_entropy = 2.0
    min_type_token_ratio = 0.5
    
    return (
        stats['token_count'] >= min_tokens and
        stats['entropy'] >= min_entropy and
        stats['type_token_ratio'] >= min_type_token_ratio
    )

# Filter corpus
corpus = Corpus()
corpus.add_from_string("the the the the the")  # Low quality
corpus.add_from_string("Machine learning is transforming how we analyze data")  # Good

quality_corpus = corpus.filter(is_quality_content)
print(f"Quality documents: {len(quality_corpus)}/{len(corpus)}")
```

### Sentiment Timeline

```python
from oet_core import Corpus

# Documents with timestamps
corpus = Corpus()
corpus.add_from_string("Great results today!", metadata={"date": "2024-01-01"})
corpus.add_from_string("Terrible performance.", metadata={"date": "2024-01-02"})
corpus.add_from_string("Excellent progress!", metadata={"date": "2024-01-03"})

# Analyze sentiment over time
timeline = []
for text in corpus:
    sentiment = text.sentiment()
    timeline.append({
        "date": text.metadata.get("date"),
        "score": sentiment['score']
    })

# Sort and display
timeline.sort(key=lambda x: x['date'])
for entry in timeline:
    print(f"{entry['date']}: {entry['score']:+d}")
```

### Research Paper Analysis

```python
from oet_core import Corpus, SQLiteHelper

# Load papers from database
db = SQLiteHelper("papers.db")
corpus = Corpus.load_from_db(db, table="abstracts")

# Build vocabulary of common research terms
vocab = corpus.vocabulary(min_freq=5)
print(f"Common terms: {', '.join(vocab[:20])}")

# Find papers about specific topic
ml_papers = corpus.filter(
    lambda t: any(term in t.get_tokens() for term in ['machine', 'learning'])
)

# Analyze sentiment of research findings
positive_papers = corpus.filter(
    lambda t: t.sentiment()['score'] > 0
)

print(f"Total papers: {len(corpus)}")
print(f"ML papers: {len(ml_papers)}")
print(f"Positive findings: {len(positive_papers)}")

# Get statistics
stats = corpus.aggregate_stats()
print(f"Average abstract length: {stats['avg_tokens_per_text']:.1f} tokens")
print(f"Total vocabulary: {stats['vocabulary_size']} terms")
```

## Verbose Logging

Enable detailed logging for debugging:

```python
from oet_core import set_mintext_verbose_logging, Text

# Enable verbose mode
set_mintext_verbose_logging(True)

# Now see detailed logs
text = Text("Hello world")
text.tokenize()  # Logs: "Text.tokenize called"
text.entropy()   # Logs: "Text.entropy called"

# Disable when done
set_mintext_verbose_logging(False)
```

## Performance Tips

1. **Cache tokens**: Use `get_tokens()` instead of calling `tokenize()` repeatedly
2. **Batch operations**: Use `Corpus.vectorize_all()` rather than individual `Text.vectorize()`
3. **Filter early**: Apply filters before expensive operations like vectorization
4. **Use binary vectors**: When you only need presence/absence, set `binary=True`
5. **Install NumPy**: Optional but provides faster array operations
6. **Vocabulary size**: Limit vocabulary with `min_freq` to reduce dimensionality

```python
# Efficient pattern
corpus = Corpus()
# ... add documents ...

# Filter first
filtered = corpus.filter(lambda t: len(t.get_tokens()) > 10)

# Build compact vocabulary
vocab = filtered.vocabulary(min_freq=2)

# Vectorize once with shared vocabulary
vectors = filtered.vectorize_all(vocabulary=vocab, binary=True)
```

## Extending MinText

MinText is designed to be simple and extensible. Common modifications:

### Custom Sentiment Lexicon

```python
# Extend sentiment with domain-specific terms
from oet_core.mintext import Text

class CustomText(Text):
    def sentiment(self, **tokenize_kwargs):
        # Add your domain-specific words
        positive_words = {
            "good", "great", "excellent",
            "profitable", "revenue", "growth"  # Financial terms
        }
        
        negative_words = {
            "bad", "terrible", "awful",
            "loss", "decline", "bankruptcy"  # Financial terms
        }
        
        tokens = self.get_tokens(**tokenize_kwargs)
        pos = sum(1 for t in tokens if t in positive_words)
        neg = sum(1 for t in tokens if t in negative_words)
        
        return {
            "score": pos - neg,
            "positive": pos,
            "negative": neg,
            "neutral": len(tokens) - pos - neg
        }
```

### Custom Tokenizer

```python
import re

class RegexText(Text):
    def tokenize(self, pattern=r'\w+', **kwargs):
        """Tokenize using regex pattern."""
        tokens = re.findall(pattern, self.content.lower())
        self._tokens = tokens
        return tokens

# Use it
text = RegexText("hello@email.com is my email")
tokens = text.tokenize(pattern=r'\S+')  # Keep punctuation
# ['hello@email.com', 'is', 'my', 'email']
```

## Common Pitfalls

1. **Forgetting to tokenize**: Some methods require tokenization first
   ```python
   text = Text("Hello world")
   # freq = text.frequencies()  # Works - auto-tokenizes
   # But for custom params, tokenize explicitly:
   text.tokenize(min_length=3)
   freq = text.frequencies()
   ```

2. **Mixed vocabularies**: Ensure consistent vocabulary for document comparison
   ```python
   # Bad: Different vocabularies per document
   vec1 = text1.vectorize()  # vocab from text1
   vec2 = text2.vectorize()  # vocab from text2
   
   # Good: Shared vocabulary
   corpus = Corpus([text1, text2])
   vocab = corpus.vocabulary()
   vec1 = text1.vectorize(vocabulary=vocab)
   vec2 = text2.vectorize(vocabulary=vocab)
   ```

3. **Ignoring metadata**: Use metadata for filtering and organization
   ```python
   text = Text("content", metadata={"source": "web", "date": "2024-01-01"})
   corpus.add(text)
   # Later: corpus.filter(lambda t: t.metadata.get("source") == "web")
   ```

## Integration Examples

### With scikit-learn

```python
from oet_core import Corpus
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

corpus = Corpus()
# ... add documents ...

# Get vectors
X = corpus.vectorize_all()

# Clustering
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)

# Dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

### With pandas

```python
from oet_core import Corpus
import pandas as pd

corpus = Corpus()
# ... add documents ...

# Convert to DataFrame
data = []
for text in corpus:
    stats = text.stats()
    stats['content'] = text.content[:50]  # Preview
    stats['sentiment'] = text.sentiment()['score']
    data.append(stats)

df = pd.DataFrame(data)
print(df.describe())
```

### With matplotlib

```python
from oet_core import Corpus
import matplotlib.pyplot as plt

corpus = Corpus()
# ... add documents ...

# Plot entropy distribution
entropies = [text.entropy() for text in corpus]
plt.hist(entropies, bins=20)
plt.xlabel('Entropy (bits)')
plt.ylabel('Frequency')
plt.title('Text Diversity Distribution')
plt.show()
```

## Summary

MinText provides essential text analysis building blocks:

- **Text**: Single document analysis (tokenization, frequency, entropy, sentiment, stats)
- **Corpus**: Collection management (filtering, vocabulary, batch vectorization, persistence)
- **Zero dependencies**: Pure Python, optional NumPy
- **Simple & extensible**: Easy to understand and customize
- **Research-ready**: Designed for exploratory analysis and prototyping

For production NLP pipelines, consider using dedicated libraries like spaCy, NLTK, or transformers. MinText excels at lightweight, dependency-free text processing for research and rapid prototyping.

## Further Reading

- See `docs/API_DOCS.md` for complete API reference
- Check `tests/test_mintext.py` for more usage examples
- Explore `src/oet_core/mintext.py` for implementation details
