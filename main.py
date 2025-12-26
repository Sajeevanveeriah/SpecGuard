"""
SpecGuard - Engineering Specification Compliance Checker

FastAPI application for analyzing specification documents against design artifacts.
Produces audit-ready traceability matrices and compliance reports.

Usage:
    uvicorn main:app --reload
"""

import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.compliance_checker import check_compliance, compliance_summary
from app.models import (
    AnalysisResult,
    AnalysisSummary,
    ComplianceResult,
    GeneratedArtifacts,
    Requirement,
    Verdict,
    RiskLevel,
)
from app.parser import parse_from_bytes
from app.report_generator import generate_artifacts, generate_xlsx_matrix
from app.requirements import extract_requirements
from app.retriever import create_retriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="SpecGuard",
    description="Engineering Specification Compliance Checker - Local-first requirement traceability and compliance verification",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


class AnalysisResponse(BaseModel):
    """Response model for /analyze endpoint."""
    analysis_id: str
    spec_filename: str
    design_filename: str
    requirements: list[dict]
    verdicts: list[dict]
    evidence: list[dict]
    risks: list[dict]
    summary: dict
    artifacts: dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    spec_file: UploadFile = File(..., description="Specification document (PDF, DOCX, or MD)"),
    design_file: UploadFile = File(..., description="Design artifact (PDF, DOCX, or MD)")
):
    """
    Analyze a specification against a design artifact.

    Extracts requirements from the specification, checks compliance against the design,
    and generates a traceability matrix and compliance report.

    **Supported formats:** PDF, DOCX, Markdown (.md), Plain text (.txt)

    **Returns:**
    - requirements: Extracted requirements with IDs, clauses, and acceptance criteria
    - verdicts: Compliance verdicts for each requirement
    - evidence: Supporting evidence from design document
    - risks: Risk assessments for each finding
    - summary: Overall compliance statistics
    - artifacts: Paths to generated report files

    **Output files generated:**
    - traceability_matrix_{id}.csv
    - compliance_report_{id}.md
    - requirements_{id}.json
    - full_results_{id}.json
    """
    # Generate analysis ID
    analysis_id = _generate_analysis_id(spec_file.filename, design_file.filename)
    logger.info(f"Starting analysis {analysis_id}: {spec_file.filename} vs {design_file.filename}")

    try:
        # Read file contents
        spec_content = await spec_file.read()
        design_content = await design_file.read()

        # Parse documents
        logger.info("Parsing specification document...")
        spec_doc = parse_from_bytes(spec_content, spec_file.filename)

        logger.info("Parsing design document...")
        design_doc = parse_from_bytes(design_content, design_file.filename)

        # Extract requirements
        logger.info("Extracting requirements...")
        requirements = extract_requirements(spec_doc, use_llm=False)

        if not requirements:
            logger.warning("No requirements extracted from specification")
            raise HTTPException(
                status_code=400,
                detail="No requirements found in specification document. "
                       "Ensure the document contains requirement statements with "
                       "SHALL, SHOULD, MUST, or MAY keywords."
            )

        # Index design document for retrieval
        logger.info("Indexing design document...")
        retriever = create_retriever()
        retriever.index_document(design_doc)

        # Check compliance
        logger.info("Checking compliance...")
        results = check_compliance(requirements, design_doc, retriever, use_llm=False)

        # Generate artifacts
        logger.info("Generating artifacts...")
        artifacts = generate_artifacts(
            requirements=requirements,
            results=results,
            spec_filename=spec_file.filename,
            design_filename=design_file.filename,
            output_dir=OUTPUT_DIR,
            analysis_id=analysis_id
        )

        # Try to generate XLSX (optional)
        xlsx_path = generate_xlsx_matrix(requirements, results, OUTPUT_DIR, analysis_id)
        if xlsx_path:
            logger.info(f"XLSX matrix generated: {xlsx_path}")

        # Build response
        summary = compliance_summary(results)

        # Extract evidence list
        evidence_list = []
        for result in results:
            for ev in result.evidence:
                evidence_list.append({
                    "requirement_id": result.requirement_id,
                    "quote": ev.quote,
                    "source": str(ev.source_location),
                    "relevance": ev.relevance_score
                })

        # Extract risk list
        risk_list = [
            {
                "requirement_id": r.requirement_id,
                "risk": r.risk.value,
                "reason": r.risk_reason
            }
            for r in results
        ]

        logger.info(f"Analysis complete: {summary['total']} requirements, "
                    f"{summary['compliance_rate']:.1f}% compliance")

        return AnalysisResponse(
            analysis_id=analysis_id,
            spec_filename=spec_file.filename,
            design_filename=design_file.filename,
            requirements=[req.model_dump(mode="json") for req in requirements],
            verdicts=[res.model_dump(mode="json") for res in results],
            evidence=evidence_list,
            risks=risk_list,
            summary=summary,
            artifacts=artifacts.model_dump()
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Missing dependency: {e}. Install required packages."
        )
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/artifacts/{filename}")
async def get_artifact(filename: str):
    """
    Download a generated artifact file.

    Available after running /analyze endpoint.
    """
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".csv": "text/csv",
        ".md": "text/markdown",
        ".json": "application/json",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pdf": "application/pdf"
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )


@app.get("/artifacts")
async def list_artifacts():
    """List all generated artifacts."""
    artifacts = []
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            artifacts.append({
                "filename": f.name,
                "size": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat()
            })

    return {"artifacts": sorted(artifacts, key=lambda x: x["created"], reverse=True)}


def _generate_analysis_id(spec_filename: str, design_filename: str) -> str:
    """Generate a unique but reproducible analysis ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    content = f"{spec_filename}:{design_filename}:{timestamp}"
    short_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
    return f"{timestamp}_{short_hash}"


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SpecGuard - Specification Compliance Checker")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
