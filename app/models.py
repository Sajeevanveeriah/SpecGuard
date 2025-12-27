"""
Pydantic models for SpecGuard.

Defines all data structures used throughout the application:
- Requirements from specifications
- Compliance verdicts and evidence
- Analysis results and reports
"""

import hashlib
import re
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class RequirementType(str, Enum):
    """RFC 2119 requirement types."""
    SHALL = "shall"
    SHOULD = "should"
    MUST = "must"
    MAY = "may"
    OTHER = "other"


class Priority(str, Enum):
    """Requirement priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class Verdict(str, Enum):
    """Compliance verdict options."""
    COMPLIANT = "Compliant"
    PARTIAL = "Partial"
    MISSING = "Missing"
    UNKNOWN = "Unknown"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SourceLocation(BaseModel):
    """Location reference within a document."""
    filename: str = Field(..., description="Source filename")
    page: Optional[int] = Field(None, description="Page number if applicable")
    section: Optional[str] = Field(None, description="Section identifier")
    line_start: Optional[int] = Field(None, description="Starting line number")
    line_end: Optional[int] = Field(None, description="Ending line number")

    def __str__(self) -> str:
        parts = [self.filename]
        if self.page:
            parts.append(f"p.{self.page}")
        if self.section:
            parts.append(f"§{self.section}")
        if self.line_start:
            if self.line_end and self.line_end != self.line_start:
                parts.append(f"L{self.line_start}-{self.line_end}")
            else:
                parts.append(f"L{self.line_start}")
        return ":".join(parts)


class Requirement(BaseModel):
    """
    A single extracted requirement from a specification document.

    The ID is deterministically generated from clause + description to ensure
    stable references across analysis runs.
    """
    clause: Optional[str] = Field(None, description="Clause reference (e.g., '5.2.1')")
    description: str = Field(..., description="Imperative requirement statement")
    acceptance_criteria: str = Field(..., description="Measurable/testable criteria")
    type: RequirementType = Field(RequirementType.OTHER, description="RFC 2119 keyword type")
    priority: Priority = Field(Priority.UNKNOWN, description="Inferred priority level")
    keywords: list[str] = Field(default_factory=list, description="Keywords for retrieval")
    source_location: Optional[SourceLocation] = Field(None, description="Location in source doc")

    @computed_field
    @property
    def id(self) -> str:
        """Generate deterministic ID from clause and normalized description."""
        normalized = re.sub(r'\s+', ' ', self.description.lower().strip())
        content = f"{self.clause or 'none'}:{normalized}"
        return f"REQ-{hashlib.sha256(content.encode()).hexdigest()[:12].upper()}"


class Evidence(BaseModel):
    """Evidence supporting a compliance verdict."""
    quote: str = Field(..., description="Direct quote or snippet from design artifact")
    source_location: SourceLocation = Field(..., description="Where evidence was found")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="How relevant to requirement")


class Gap(BaseModel):
    """Identified gap in compliance."""
    description: str = Field(..., description="What is missing or incomplete")
    suggested_evidence: str = Field(..., description="What evidence would satisfy this")
    suggested_test: Optional[str] = Field(None, description="Suggested verification test")


class ComplianceResult(BaseModel):
    """
    Compliance evaluation result for a single requirement.

    Always includes rationale and either evidence or explicit statement of no evidence.
    """
    requirement_id: str = Field(..., description="Reference to requirement ID")
    verdict: Verdict = Field(..., description="Compliance determination")
    rationale: str = Field(..., description="Engineering justification for verdict")
    evidence: list[Evidence] = Field(default_factory=list, description="Supporting evidence")
    gaps: list[Gap] = Field(default_factory=list, description="Identified gaps")
    risk: RiskLevel = Field(RiskLevel.MEDIUM, description="Risk level")
    risk_reason: str = Field("", description="Explanation of risk assessment")
    checked_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def has_evidence(self) -> bool:
        """Whether any evidence was found."""
        return len(self.evidence) > 0


class DocumentChunk(BaseModel):
    """A chunk of text from a parsed document."""
    content: str = Field(..., description="Text content of the chunk")
    source_location: SourceLocation = Field(..., description="Location in source document")
    chunk_index: int = Field(..., description="Sequential index of chunk")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords")


class ParsedDocument(BaseModel):
    """Result of parsing a document."""
    filename: str
    file_type: str
    total_pages: Optional[int] = None
    full_text: str
    chunks: list[DocumentChunk]
    parsed_at: datetime = Field(default_factory=datetime.utcnow)


class AnalysisRequest(BaseModel):
    """Request to analyze spec against design."""
    spec_filename: str
    design_filename: str
    options: dict = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    analysis_id: str
    spec_filename: str
    design_filename: str
    requirements: list[Requirement]
    verdicts: list[ComplianceResult]
    summary: "AnalysisSummary"
    artifacts: "GeneratedArtifacts"
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class AnalysisSummary(BaseModel):
    """Summary statistics of the analysis."""
    total_requirements: int
    compliant_count: int
    partial_count: int
    missing_count: int
    unknown_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int

    @computed_field
    @property
    def compliance_rate(self) -> float:
        """Percentage of fully compliant requirements."""
        if self.total_requirements == 0:
            return 0.0
        return (self.compliant_count / self.total_requirements) * 100


class GeneratedArtifacts(BaseModel):
    """Paths to generated output artifacts."""
    traceability_matrix_csv: str
    compliance_report_md: str
    requirements_json: str
    full_results_json: str
    traceability_matrix_xlsx: Optional[str] = None


# Update forward references
AnalysisResult.model_rebuild()
