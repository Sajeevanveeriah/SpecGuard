"""
Requirement extraction module for SpecGuard.

Extracts testable, verifiable requirements from specification documents.
Uses rule-based extraction enhanced with LLM prompts for complex cases.

This module provides both:
- Local rule-based extraction (no external dependencies)
- LLM-assisted extraction (when LLM adapter is configured)
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .models import (
    DocumentChunk,
    ParsedDocument,
    Priority,
    Requirement,
    RequirementType,
    SourceLocation,
)

logger = logging.getLogger(__name__)

# RFC 2119 keywords for requirement identification
RFC2119_KEYWORDS = {
    "shall": RequirementType.SHALL,
    "shall not": RequirementType.SHALL,
    "must": RequirementType.MUST,
    "must not": RequirementType.MUST,
    "required": RequirementType.MUST,
    "should": RequirementType.SHOULD,
    "should not": RequirementType.SHOULD,
    "recommended": RequirementType.SHOULD,
    "may": RequirementType.MAY,
    "optional": RequirementType.MAY,
}

# Priority indicators
PRIORITY_INDICATORS = {
    "high": ["critical", "essential", "mandatory", "required", "must", "safety", "security"],
    "medium": ["important", "should", "recommended", "necessary"],
    "low": ["optional", "may", "nice to have", "if possible", "desirable"],
}


def extract_requirements(
    document: ParsedDocument,
    use_llm: bool = False,
    llm_adapter: Optional[callable] = None
) -> list[Requirement]:
    """
    Extract all testable requirements from a parsed document.

    Args:
        document: Parsed specification document
        use_llm: Whether to use LLM for enhanced extraction
        llm_adapter: Callable that takes prompt and returns LLM response

    Returns:
        List of extracted requirements with stable IDs
    """
    logger.info(f"Extracting requirements from: {document.filename}")

    if use_llm and llm_adapter:
        requirements = _extract_with_llm(document, llm_adapter)
    else:
        requirements = _extract_rule_based(document)

    # Deduplicate by ID
    seen_ids = set()
    unique_requirements = []
    for req in requirements:
        if req.id not in seen_ids:
            unique_requirements.append(req)
            seen_ids.add(req.id)

    logger.info(f"Extracted {len(unique_requirements)} unique requirements")
    return unique_requirements


def _extract_rule_based(document: ParsedDocument) -> list[Requirement]:
    """
    Extract requirements using rule-based pattern matching.

    Looks for RFC 2119 keywords and structured requirement patterns.
    """
    requirements = []

    for chunk in document.chunks:
        chunk_requirements = _extract_from_chunk(chunk)
        requirements.extend(chunk_requirements)

    return requirements


def _extract_from_chunk(chunk: DocumentChunk) -> list[Requirement]:
    """Extract requirements from a single chunk."""
    requirements = []
    text = chunk.content

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check for RFC 2119 keywords
        req_type = _identify_requirement_type(sentence)
        if req_type:
            # Extract clause reference if present
            clause = _extract_clause_reference(sentence, text)

            # Generate acceptance criteria
            acceptance = _generate_acceptance_criteria(sentence)

            # Determine priority
            priority = _infer_priority(sentence, text)

            # Extract keywords
            keywords = _extract_requirement_keywords(sentence)

            requirement = Requirement(
                clause=clause,
                description=sentence,
                acceptance_criteria=acceptance,
                type=req_type,
                priority=priority,
                keywords=keywords,
                source_location=chunk.source_location
            )
            requirements.append(requirement)

    return requirements


def _identify_requirement_type(sentence: str) -> Optional[RequirementType]:
    """Identify if sentence contains RFC 2119 requirement language."""
    sentence_lower = sentence.lower()

    # Check for multi-word keywords first
    for keyword, req_type in sorted(RFC2119_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if keyword in sentence_lower:
            return req_type

    return None


def _extract_clause_reference(sentence: str, context: str) -> Optional[str]:
    """
    Extract clause reference (e.g., '5.2.1') from sentence or nearby context.

    IMPORTANT: Only extracts actually present clause numbers, never invents them.
    """
    # Look for clause patterns in the sentence itself
    patterns = [
        r'\[(\d+(?:\.\d+)*)\]',  # [5.2.1]
        r'(?:Section|Clause|§|Req\.?|REQ)\s*(\d+(?:\.\d+)*)',  # Section 5.2.1
        r'^(\d+(?:\.\d+)+)\s*[:\.]?\s+',  # 5.2.1: or 5.2.1. at start
    ]

    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return match.group(1)

    # Look in nearby context (previous lines)
    lines = context.split('\n')
    for i, line in enumerate(lines):
        if sentence in line or sentence[:50] in line:
            # Check previous 3 lines for heading with clause number
            for j in range(max(0, i - 3), i):
                for pattern in patterns:
                    match = re.search(pattern, lines[j], re.IGNORECASE)
                    if match:
                        return match.group(1)
            break

    return None


def _generate_acceptance_criteria(sentence: str) -> str:
    """
    Generate measurable acceptance criteria from requirement statement.

    For MVP, creates criteria based on the requirement action.
    """
    # Extract the main action/verb
    sentence_lower = sentence.lower()

    # Common patterns for criteria generation
    criteria_templates = {
        "shall provide": "Verify that {subject} is provided and accessible",
        "shall support": "Verify that {subject} is supported and functional",
        "shall be": "Verify that the condition is met",
        "shall include": "Verify that {subject} is included",
        "shall ensure": "Verify that {subject} is ensured through testing",
        "shall not": "Verify that the prohibited action does not occur",
        "must": "Verify compliance with mandatory requirement",
        "should": "Verify recommended behavior is implemented",
    }

    for pattern, template in criteria_templates.items():
        if pattern in sentence_lower:
            # Try to extract subject
            match = re.search(rf'{pattern}\s+(.+?)(?:\.|$)', sentence_lower)
            subject = match.group(1) if match else "the requirement"
            return template.format(subject=subject[:100])

    # Default criteria
    return f"Verify: {sentence[:100]}..." if len(sentence) > 100 else f"Verify: {sentence}"


def _infer_priority(sentence: str, context: str) -> Priority:
    """Infer requirement priority from keywords and context."""
    combined_text = (sentence + " " + context).lower()

    for priority_level, indicators in PRIORITY_INDICATORS.items():
        for indicator in indicators:
            if indicator in combined_text:
                return Priority(priority_level)

    return Priority.UNKNOWN


def _extract_requirement_keywords(sentence: str) -> list[str]:
    """Extract significant keywords from requirement for retrieval."""
    # Remove common words and extract technical terms
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'shall', 'should', 'must', 'may',
        'will', 'be', 'to', 'of', 'and', 'or', 'in', 'for', 'with', 'that',
        'this', 'from', 'by', 'on', 'as', 'at', 'not', 'it', 'all', 'can'
    }

    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', sentence.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            unique.append(kw)
            seen.add(kw)

    return unique[:20]


def _extract_with_llm(
    document: ParsedDocument,
    llm_adapter: callable
) -> list[Requirement]:
    """
    Extract requirements using LLM assistance.

    Loads prompt template and processes document chunks.
    """
    # Load prompt template
    prompt_path = Path(__file__).parent.parent / "prompts" / "extract_requirements.md"
    if not prompt_path.exists():
        logger.warning("LLM prompt template not found, falling back to rule-based")
        return _extract_rule_based(document)

    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    requirements = []

    for chunk in document.chunks:
        # Build prompt with chunk content
        prompt = prompt_template.replace("{{SPEC_TEXT}}", chunk.content)
        prompt = prompt.replace("{{FILENAME}}", document.filename)

        try:
            response = llm_adapter(prompt)
            chunk_requirements = _parse_llm_requirements(response, chunk.source_location)
            requirements.extend(chunk_requirements)
        except Exception as e:
            logger.error(f"LLM extraction failed for chunk {chunk.chunk_index}: {e}")
            # Fall back to rule-based for this chunk
            requirements.extend(_extract_from_chunk(chunk))

    return requirements


def _parse_llm_requirements(
    llm_response: str,
    source_location: SourceLocation
) -> list[Requirement]:
    """Parse LLM response into Requirement objects."""
    requirements = []

    # Extract JSON from response
    try:
        # Look for JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', llm_response)
        if json_match:
            data = json.loads(json_match.group())
        else:
            # Try parsing entire response as JSON
            data = json.loads(llm_response)

        if not isinstance(data, list):
            data = [data]

        for item in data:
            req = Requirement(
                clause=item.get("clause"),
                description=item.get("description", ""),
                acceptance_criteria=item.get("acceptance_criteria", ""),
                type=RequirementType(item.get("type", "other")),
                priority=Priority(item.get("priority", "unknown")),
                keywords=item.get("keywords", []),
                source_location=source_location
            )
            requirements.append(req)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")

    return requirements


def requirements_to_json(requirements: list[Requirement]) -> str:
    """Serialize requirements list to JSON string."""
    return json.dumps(
        [req.model_dump() for req in requirements],
        indent=2,
        default=str
    )


def requirements_from_json(json_str: str) -> list[Requirement]:
    """Deserialize requirements from JSON string."""
    data = json.loads(json_str)
    return [Requirement(**item) for item in data]
