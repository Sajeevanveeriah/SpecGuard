"""
Compliance checking module for SpecGuard.

Evaluates design artifacts against extracted requirements.
Produces verdicts with evidence, gaps, and risk assessments.

Supports both rule-based and LLM-assisted compliance evaluation.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import (
    ComplianceResult,
    DocumentChunk,
    Evidence,
    Gap,
    ParsedDocument,
    Requirement,
    RiskLevel,
    SourceLocation,
    Verdict,
)
from .retriever import KeywordMatcher, LexicalRetriever

logger = logging.getLogger(__name__)

# Minimum score threshold for considering evidence relevant
RELEVANCE_THRESHOLD = 0.3


def check_compliance(
    requirements: list[Requirement],
    design_document: ParsedDocument,
    retriever: LexicalRetriever,
    use_llm: bool = False,
    llm_adapter: Optional[callable] = None,
    top_k: int = 5
) -> list[ComplianceResult]:
    """
    Check compliance of design against all requirements.

    Args:
        requirements: List of requirements to check
        design_document: Parsed design artifact
        retriever: Indexed retriever for finding relevant sections
        use_llm: Whether to use LLM for evaluation
        llm_adapter: Callable for LLM inference
        top_k: Number of chunks to retrieve for evidence per requirement

    Returns:
        List of compliance results for each requirement
    """
    logger.info(f"Checking compliance for {len(requirements)} requirements")

    results = []
    for i, requirement in enumerate(requirements):
        logger.debug(f"Checking requirement {i + 1}/{len(requirements)}: {requirement.id}")

        if use_llm and llm_adapter:
            result = _check_with_llm(requirement, design_document, retriever, llm_adapter, top_k=top_k)
        else:
            result = _check_rule_based(requirement, design_document, retriever, top_k=top_k)

        results.append(result)

    logger.info(f"Compliance check complete. Results: "
                f"{sum(1 for r in results if r.verdict == Verdict.COMPLIANT)} compliant, "
                f"{sum(1 for r in results if r.verdict == Verdict.PARTIAL)} partial, "
                f"{sum(1 for r in results if r.verdict == Verdict.MISSING)} missing, "
                f"{sum(1 for r in results if r.verdict == Verdict.UNKNOWN)} unknown")

    return results


def _check_rule_based(
    requirement: Requirement,
    design_document: ParsedDocument,
    retriever: LexicalRetriever,
    top_k: int = 5
) -> ComplianceResult:
    """
    Check compliance using rule-based analysis.

    Uses keyword matching and retrieval to find evidence.
    """
    # Retrieve relevant chunks
    relevant_chunks = retriever.retrieve_for_requirement(
        requirement,
        design_document.filename,
        top_k=top_k
    )

    # Analyze retrieved chunks for evidence
    evidence_list = []
    max_relevance = 0.0

    for chunk, score in relevant_chunks:
        # Normalize score to 0-1 range
        normalized_score = min(1.0, score / 10.0)
        max_relevance = max(max_relevance, normalized_score)

        # Find specific matches in chunk
        matches = KeywordMatcher.find_matches(
            chunk.content,
            requirement.keywords,
            context_chars=150
        )

        for match in matches:
            evidence = Evidence(
                quote=match["quote"],
                source_location=SourceLocation(
                    filename=chunk.source_location.filename,
                    page=chunk.source_location.page,
                    section=chunk.source_location.section,
                    line_start=match.get("line", chunk.source_location.line_start)
                ),
                relevance_score=normalized_score
            )
            evidence_list.append(evidence)

        # Also add chunk as general evidence if no specific matches but high score
        if not matches and normalized_score > RELEVANCE_THRESHOLD:
            # Extract most relevant sentence from chunk
            snippet = _extract_relevant_snippet(chunk.content, requirement.keywords)
            if snippet:
                evidence = Evidence(
                    quote=snippet,
                    source_location=chunk.source_location,
                    relevance_score=normalized_score
                )
                evidence_list.append(evidence)

    # Determine verdict based on evidence
    verdict, rationale, gaps = _determine_verdict(
        requirement, evidence_list, max_relevance
    )

    # Assess risk
    risk, risk_reason = _assess_risk(requirement, verdict, evidence_list)

    return ComplianceResult(
        requirement_id=requirement.id,
        verdict=verdict,
        rationale=rationale,
        evidence=evidence_list[:5],  # Limit to top 5 pieces of evidence
        gaps=gaps,
        risk=risk,
        risk_reason=risk_reason
    )


def _check_with_llm(
    requirement: Requirement,
    design_document: ParsedDocument,
    retriever: LexicalRetriever,
    llm_adapter: callable,
    top_k: int = 5
) -> ComplianceResult:
    """
    Check compliance using LLM evaluation.

    Retrieves relevant design excerpts and asks LLM to evaluate.
    """
    # Load prompt template
    prompt_path = Path(__file__).parent.parent / "prompts" / "compliance_check.md"
    if not prompt_path.exists():
        logger.warning("LLM prompt template not found, falling back to rule-based")
        return _check_rule_based(requirement, design_document, retriever, top_k=top_k)

    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # Retrieve relevant chunks
    relevant_chunks = retriever.retrieve_for_requirement(
        requirement,
        design_document.filename,
        top_k=top_k
    )

    # Build design excerpts string
    excerpts = []
    for chunk, score in relevant_chunks:
        location = str(chunk.source_location)
        excerpts.append(f"[{location}]\n{chunk.content}")

    design_excerpts = "\n\n---\n\n".join(excerpts) if excerpts else "No relevant excerpts found."

    # Build prompt
    prompt = prompt_template.replace("{{REQUIREMENT_ID}}", requirement.id)
    prompt = prompt.replace("{{REQUIREMENT_DESCRIPTION}}", requirement.description)
    prompt = prompt.replace("{{ACCEPTANCE_CRITERIA}}", requirement.acceptance_criteria)
    prompt = prompt.replace("{{DESIGN_EXCERPTS}}", design_excerpts)

    try:
        response = llm_adapter(prompt)
        return _parse_llm_compliance_result(response, requirement.id)
    except Exception as e:
        logger.error(f"LLM compliance check failed: {e}")
        return _check_rule_based(requirement, design_document, retriever, top_k=top_k)


def _parse_llm_compliance_result(llm_response: str, requirement_id: str) -> ComplianceResult:
    """Parse LLM response into ComplianceResult."""
    try:
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(llm_response)

        # Parse evidence
        evidence_list = []
        for ev in data.get("evidence", []):
            evidence_list.append(Evidence(
                quote=ev.get("quote", ""),
                source_location=SourceLocation(
                    filename=ev.get("filename", "unknown"),
                    section=ev.get("section"),
                    line_start=ev.get("line")
                ),
                relevance_score=ev.get("relevance", 0.5)
            ))

        # Parse gaps
        gaps_list = []
        for gap in data.get("gaps", []):
            gaps_list.append(Gap(
                description=gap.get("description", ""),
                suggested_evidence=gap.get("suggested_evidence", ""),
                suggested_test=gap.get("suggested_test")
            ))

        return ComplianceResult(
            requirement_id=requirement_id,
            verdict=Verdict(data.get("verdict", "Unknown")),
            rationale=data.get("rationale", ""),
            evidence=evidence_list,
            gaps=gaps_list,
            risk=RiskLevel(data.get("risk", "medium")),
            risk_reason=data.get("risk_reason", "")
        )

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse LLM response: {e}")
        return ComplianceResult(
            requirement_id=requirement_id,
            verdict=Verdict.UNKNOWN,
            rationale=f"Failed to parse LLM response: {str(e)}",
            evidence=[],
            gaps=[],
            risk=RiskLevel.MEDIUM,
            risk_reason="Unable to evaluate due to parsing error"
        )


def _determine_verdict(
    requirement: Requirement,
    evidence: list[Evidence],
    max_relevance: float
) -> tuple[Verdict, str, list[Gap]]:
    """
    Determine compliance verdict based on evidence analysis.

    Returns:
        Tuple of (verdict, rationale, gaps)
    """
    gaps = []

    if not evidence:
        # No evidence found
        gaps.append(Gap(
            description="No evidence found in design document addressing this requirement",
            suggested_evidence=f"Documentation or implementation addressing: {requirement.description[:100]}",
            suggested_test=f"Test to verify: {requirement.acceptance_criteria[:100]}"
        ))
        return (
            Verdict.MISSING,
            "No evidence found in design document. The requirement may not be addressed.",
            gaps
        )

    # Analyze evidence quality
    high_relevance_evidence = [e for e in evidence if e.relevance_score > 0.6]
    medium_relevance_evidence = [e for e in evidence if 0.3 < e.relevance_score <= 0.6]

    if high_relevance_evidence:
        # Strong evidence found
        if len(high_relevance_evidence) >= 2:
            return (
                Verdict.COMPLIANT,
                f"Strong evidence found ({len(high_relevance_evidence)} relevant excerpts) "
                f"addressing the requirement.",
                []
            )
        else:
            # Some evidence but may be incomplete
            gaps.append(Gap(
                description="Limited evidence found; additional verification recommended",
                suggested_evidence="Additional documentation or implementation details",
                suggested_test=requirement.acceptance_criteria
            ))
            return (
                Verdict.PARTIAL,
                f"Partial evidence found ({len(high_relevance_evidence)} highly relevant excerpt). "
                f"May require additional verification.",
                gaps
            )

    elif medium_relevance_evidence:
        # Moderate evidence - partial compliance
        gaps.append(Gap(
            description="Evidence found but with moderate relevance; explicit confirmation needed",
            suggested_evidence=f"Explicit documentation for: {requirement.description[:80]}",
            suggested_test=requirement.acceptance_criteria
        ))
        return (
            Verdict.PARTIAL,
            f"Moderate evidence found ({len(medium_relevance_evidence)} excerpts) "
            f"but relevance is not conclusive.",
            gaps
        )

    else:
        # Only weak evidence
        gaps.append(Gap(
            description="Only weak/tangential evidence found",
            suggested_evidence=f"Direct evidence for: {requirement.description[:80]}",
            suggested_test=requirement.acceptance_criteria
        ))
        return (
            Verdict.UNKNOWN,
            "Only weak evidence found. Unable to determine compliance with confidence.",
            gaps
        )


def _assess_risk(
    requirement: Requirement,
    verdict: Verdict,
    evidence: list[Evidence]
) -> tuple[RiskLevel, str]:
    """
    Assess risk level for a compliance result.

    Considers requirement priority, verdict, and evidence quality.
    """
    # High priority requirements with issues are high risk
    if requirement.priority.value == "high":
        if verdict in (Verdict.MISSING, Verdict.UNKNOWN):
            return (
                RiskLevel.HIGH,
                "High-priority requirement with missing or unknown compliance status"
            )
        elif verdict == Verdict.PARTIAL:
            return (
                RiskLevel.MEDIUM,
                "High-priority requirement with partial compliance"
            )

    # SHALL/MUST requirements are higher risk
    if requirement.type.value in ("shall", "must"):
        if verdict == Verdict.MISSING:
            return (
                RiskLevel.HIGH,
                f"Mandatory requirement ({requirement.type.value.upper()}) not addressed"
            )
        elif verdict == Verdict.PARTIAL:
            return (
                RiskLevel.MEDIUM,
                f"Mandatory requirement ({requirement.type.value.upper()}) partially addressed"
            )

    # Default risk based on verdict
    risk_map = {
        Verdict.COMPLIANT: (RiskLevel.LOW, "Requirement satisfied with evidence"),
        Verdict.PARTIAL: (RiskLevel.MEDIUM, "Partial compliance requires follow-up"),
        Verdict.MISSING: (RiskLevel.HIGH, "Requirement not addressed in design"),
        Verdict.UNKNOWN: (RiskLevel.MEDIUM, "Unable to determine compliance status"),
    }

    return risk_map.get(verdict, (RiskLevel.MEDIUM, "Default risk assessment"))


def _extract_relevant_snippet(text: str, keywords: list[str], max_length: int = 200) -> Optional[str]:
    """
    Extract the most relevant snippet from text based on keywords.

    Returns a sentence or phrase containing the most keyword matches.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    best_sentence = None
    best_score = 0

    text_lower_keywords = [kw.lower() for kw in keywords]

    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for kw in text_lower_keywords if kw in sentence_lower)

        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_sentence and best_score > 0:
        if len(best_sentence) > max_length:
            return best_sentence[:max_length] + "..."
        return best_sentence

    return None


def compliance_summary(results: list[ComplianceResult]) -> dict:
    """
    Generate summary statistics from compliance results.

    Returns:
        Dictionary with counts and percentages
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "compliant": 0,
            "partial": 0,
            "missing": 0,
            "unknown": 0,
            "compliance_rate": 0.0
        }

    counts = {
        "total": total,
        "compliant": sum(1 for r in results if r.verdict == Verdict.COMPLIANT),
        "partial": sum(1 for r in results if r.verdict == Verdict.PARTIAL),
        "missing": sum(1 for r in results if r.verdict == Verdict.MISSING),
        "unknown": sum(1 for r in results if r.verdict == Verdict.UNKNOWN),
        "high_risk": sum(1 for r in results if r.risk == RiskLevel.HIGH),
        "medium_risk": sum(1 for r in results if r.risk == RiskLevel.MEDIUM),
        "low_risk": sum(1 for r in results if r.risk == RiskLevel.LOW),
    }

    counts["compliance_rate"] = (counts["compliant"] / total) * 100

    return counts
