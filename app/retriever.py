"""
Lexical retrieval module for SpecGuard.

Provides local-first retrieval using keyword scoring (TF-IDF style).
No external services or embeddings required for MVP.

The retriever indexes document chunks and retrieves relevant excerpts
based on requirement keywords for compliance checking.
"""

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Optional

from .models import DocumentChunk, ParsedDocument, Requirement

logger = logging.getLogger(__name__)


class LexicalRetriever:
    """
    Simple lexical retriever using TF-IDF-like scoring.

    Indexes document chunks by their terms and retrieves
    relevant chunks for a given query or requirement.
    """

    def __init__(self):
        """Initialize the retriever."""
        self.documents: dict[str, ParsedDocument] = {}
        self.chunk_index: dict[str, list[DocumentChunk]] = {}
        self.term_document_freq: Counter = Counter()
        self.chunk_term_freq: dict[int, Counter] = {}
        self.total_chunks: int = 0

    def index_document(self, document: ParsedDocument) -> None:
        """
        Index a parsed document for retrieval.

        Args:
            document: Parsed document with chunks
        """
        logger.info(f"Indexing document: {document.filename}")

        self.documents[document.filename] = document
        self.chunk_index[document.filename] = document.chunks

        for chunk in document.chunks:
            chunk_id = self._chunk_id(document.filename, chunk.chunk_index)
            terms = self._tokenize(chunk.content)
            self.chunk_term_freq[chunk_id] = Counter(terms)

            # Update document frequency (unique terms per chunk)
            unique_terms = set(terms)
            for term in unique_terms:
                self.term_document_freq[term] += 1

            self.total_chunks += 1

        logger.info(f"Indexed {len(document.chunks)} chunks, {len(self.term_document_freq)} unique terms")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_filter: Optional[str] = None
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Retrieve top-k relevant chunks for a query.

        Args:
            query: Search query string
            top_k: Number of results to return
            document_filter: Only search in this document

        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scores: dict[int, float] = {}

        # Score each chunk
        for filename, chunks in self.chunk_index.items():
            if document_filter and filename != document_filter:
                continue

            for chunk in chunks:
                chunk_id = self._chunk_id(filename, chunk.chunk_index)
                score = self._score_chunk(chunk_id, query_terms)
                if score > 0:
                    scores[chunk_id] = score

        # Sort by score and return top-k
        sorted_chunks = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

        results = []
        for chunk_id, score in sorted_chunks:
            chunk = self._get_chunk_by_id(chunk_id)
            if chunk:
                results.append((chunk, score))

        return results

    def retrieve_for_requirement(
        self,
        requirement: Requirement,
        design_document: str,
        top_k: int = 5
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Retrieve relevant design chunks for a requirement.

        Uses requirement description and keywords for matching.

        Args:
            requirement: The requirement to find evidence for
            design_document: Filename of design document to search
            top_k: Number of results to return

        Returns:
            List of (chunk, score) tuples
        """
        # Build query from requirement
        query_parts = [requirement.description]
        query_parts.extend(requirement.keywords)
        if requirement.acceptance_criteria:
            query_parts.append(requirement.acceptance_criteria)

        query = " ".join(query_parts)

        return self.retrieve(query, top_k=top_k, document_filter=design_document)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms for indexing/querying."""
        # Lowercase and extract words
        text = text.lower()
        tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)

        # Remove very short tokens and stopwords
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then', 'once',
            'and', 'but', 'or', 'if', 'because', 'until', 'while',
            'this', 'that', 'these', 'those', 'it', 'its', 'not', 'no'
        }

        return [t for t in tokens if len(t) > 2 and t not in stop_words]

    def _score_chunk(self, chunk_id: int, query_terms: list[str]) -> float:
        """
        Score a chunk against query terms using TF-IDF.

        Args:
            chunk_id: ID of chunk to score
            query_terms: Tokenized query terms

        Returns:
            Relevance score
        """
        if chunk_id not in self.chunk_term_freq:
            return 0.0

        chunk_tf = self.chunk_term_freq[chunk_id]
        score = 0.0

        for term in query_terms:
            tf = chunk_tf.get(term, 0)
            if tf == 0:
                continue

            # IDF calculation
            df = self.term_document_freq.get(term, 1)
            idf = math.log(self.total_chunks / df) + 1

            # TF-IDF score
            score += (1 + math.log(tf)) * idf

        return score

    def _chunk_id(self, filename: str, chunk_index: int) -> int:
        """Generate unique ID for a chunk."""
        return hash((filename, chunk_index))

    def _get_chunk_by_id(self, chunk_id: int) -> Optional[DocumentChunk]:
        """Retrieve chunk object by ID."""
        for filename, chunks in self.chunk_index.items():
            for chunk in chunks:
                if self._chunk_id(filename, chunk.chunk_index) == chunk_id:
                    return chunk
        return None

    def clear(self) -> None:
        """Clear all indexed documents."""
        self.documents.clear()
        self.chunk_index.clear()
        self.term_document_freq.clear()
        self.chunk_term_freq.clear()
        self.total_chunks = 0


class KeywordMatcher:
    """
    Simple keyword-based matcher for finding evidence.

    Looks for exact and fuzzy keyword matches in text.
    """

    @staticmethod
    def find_matches(
        text: str,
        keywords: list[str],
        context_chars: int = 200
    ) -> list[dict]:
        """
        Find keyword matches with surrounding context.

        Args:
            text: Text to search
            keywords: Keywords to look for
            context_chars: Characters of context to include

        Returns:
            List of match dictionaries with quote and position
        """
        matches = []
        text_lower = text.lower()

        for keyword in keywords:
            keyword_lower = keyword.lower()
            start = 0

            while True:
                pos = text_lower.find(keyword_lower, start)
                if pos == -1:
                    break

                # Extract context around match
                context_start = max(0, pos - context_chars)
                context_end = min(len(text), pos + len(keyword) + context_chars)

                # Find sentence boundaries
                quote = text[context_start:context_end]
                if context_start > 0:
                    quote = "..." + quote
                if context_end < len(text):
                    quote = quote + "..."

                matches.append({
                    "keyword": keyword,
                    "quote": quote.strip(),
                    "position": pos,
                    "line": text[:pos].count('\n') + 1
                })

                start = pos + 1

        # Deduplicate overlapping matches
        return _deduplicate_matches(matches)


def _deduplicate_matches(matches: list[dict]) -> list[dict]:
    """Remove overlapping matches, keeping highest quality ones."""
    if not matches:
        return []

    # Sort by position
    sorted_matches = sorted(matches, key=lambda x: x["position"])

    deduped = [sorted_matches[0]]
    for match in sorted_matches[1:]:
        last = deduped[-1]
        # Check for overlap (within same context window)
        if match["position"] - last["position"] > 100:
            deduped.append(match)

    return deduped


def create_retriever() -> LexicalRetriever:
    """Factory function to create a new retriever instance."""
    return LexicalRetriever()
