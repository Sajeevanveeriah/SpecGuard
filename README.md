# SpecGuard

**Engineering Specification Compliance Checker**

A local-first tool that ingests specification documents and design artifacts, extracts testable requirements, checks compliance with evidence, and outputs audit-ready traceability matrices and gap reports.

## Features

- **Requirement Extraction**: Automatically extract testable requirements from specifications (PDF, DOCX, Markdown)
- **Compliance Checking**: Evaluate design artifacts against requirements with evidence-based verdicts
- **Traceability Matrix**: Generate CSV/XLSX matrices linking requirements to evidence
- **Compliance Reports**: Produce Markdown reports with executive summaries and risk assessments
- **Local-First**: No external API calls during runtime - fully offline capable
- **Audit-Ready**: All outputs include evidence citations and source locations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SpecGuard.git
cd SpecGuard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the API

```bash
# Start the FastAPI server
uvicorn main:app --reload

# Or use the CLI
python main.py --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`

- Interactive docs: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### API Usage

#### Analyze Specification vs Design

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "spec_file=@specification.pdf" \
  -F "design_file=@design.md"
```

#### Response Format

```json
{
  "analysis_id": "20240115_143022_a1b2c3d4",
  "spec_filename": "specification.pdf",
  "design_filename": "design.md",
  "requirements": [
    {
      "id": "REQ-A1B2C3D4E5F6",
      "clause": "5.2.1",
      "description": "The system shall authenticate users...",
      "acceptance_criteria": "Verify that users can log in...",
      "type": "shall",
      "priority": "high",
      "keywords": ["authentication", "users", "login"]
    }
  ],
  "verdicts": [
    {
      "requirement_id": "REQ-A1B2C3D4E5F6",
      "verdict": "Compliant",
      "rationale": "Strong evidence found addressing the requirement.",
      "evidence": [...],
      "gaps": [],
      "risk": "low",
      "risk_reason": "Requirement satisfied with evidence"
    }
  ],
  "summary": {
    "total": 15,
    "compliant": 10,
    "partial": 3,
    "missing": 2,
    "unknown": 0,
    "compliance_rate": 66.7
  },
  "artifacts": {
    "traceability_matrix_csv": "outputs/traceability_matrix_20240115_143022_a1b2c3d4.csv",
    "compliance_report_md": "outputs/compliance_report_20240115_143022_a1b2c3d4.md",
    "requirements_json": "outputs/requirements_20240115_143022_a1b2c3d4.json",
    "full_results_json": "outputs/full_results_20240115_143022_a1b2c3d4.json"
  }
}
```

## Architecture

```
SpecGuard/
├── main.py                    # FastAPI application
├── app/
│   ├── __init__.py
│   ├── models.py              # Pydantic data models
│   ├── parser.py              # Document parsing (PDF, DOCX, MD)
│   ├── requirements.py        # Requirement extraction
│   ├── retriever.py           # Lexical retrieval (TF-IDF)
│   ├── compliance_checker.py  # Compliance evaluation
│   └── report_generator.py    # Output generation
├── prompts/
│   ├── extract_requirements.md  # LLM prompt for extraction
│   └── compliance_check.md      # LLM prompt for evaluation
├── outputs/                   # Generated artifacts
├── tests/                     # Unit tests
├── requirements.txt
└── README.md
```

## Data Flow

```
┌─────────────────┐     ┌─────────────────┐
│  Specification  │     │  Design Artifact │
│  (PDF/DOCX/MD)  │     │  (PDF/DOCX/MD)   │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│     Parser      │     │     Parser      │
│  (Chunk + Index)│     │  (Chunk + Index)│
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       │
┌─────────────────┐              │
│  Requirement    │              │
│   Extraction    │              │
└────────┬────────┘              │
         │                       │
         ▼                       ▼
┌────────────────────────────────────────┐
│          Compliance Checker            │
│  (Retrieve → Match → Evaluate → Score) │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│          Report Generator              │
│  (CSV Matrix + MD Report + JSON)       │
└────────────────────────────────────────┘
```

## Requirement Model

Each extracted requirement includes:

| Field | Description | Example |
|-------|-------------|---------|
| `id` | Deterministic hash ID | `REQ-A1B2C3D4E5F6` |
| `clause` | Section reference | `5.2.1` |
| `description` | Full requirement text | `The system shall...` |
| `acceptance_criteria` | Testable criteria | `Verify that...` |
| `type` | RFC 2119 keyword | `shall`, `should`, `must`, `may` |
| `priority` | Inferred priority | `high`, `medium`, `low`, `unknown` |
| `keywords` | Retrieval keywords | `["auth", "login", "user"]` |

## Verdict Model

Each compliance result includes:

| Field | Description | Options |
|-------|-------------|---------|
| `verdict` | Compliance status | `Compliant`, `Partial`, `Missing`, `Unknown` |
| `rationale` | Engineering justification | Free text |
| `evidence` | Supporting quotes with locations | List of evidence objects |
| `gaps` | Identified gaps | List of gap objects |
| `risk` | Risk assessment | `high`, `medium`, `low` |
| `risk_reason` | Risk explanation | Free text |

## Supported File Formats

| Format | Extension | Parser |
|--------|-----------|--------|
| PDF | `.pdf` | PyMuPDF (fitz) |
| Word | `.docx` | python-docx |
| Markdown | `.md` | Plain text |
| Text | `.txt` | Plain text |

## Dependencies

### Core (Required)
- `fastapi>=0.104.0` - Web framework
- `uvicorn>=0.24.0` - ASGI server
- `pydantic>=2.5.0` - Data validation
- `python-multipart>=0.0.6` - File uploads

### Document Parsing
- `pymupdf>=1.23.0` - PDF parsing
- `python-docx>=1.0.0` - DOCX parsing

### Optional
- `openpyxl>=3.1.0` - XLSX generation

## Configuration

Environment variables (optional):

```bash
# Logging level
LOG_LEVEL=INFO

# Output directory
OUTPUT_DIR=./outputs

# Chunk settings
CHUNK_SIZE=1500
CHUNK_OVERLAP=200
```

## Extending with LLM

The MVP includes prompts for LLM-assisted extraction and evaluation. To enable:

1. Implement an LLM adapter function:
```python
def llm_adapter(prompt: str) -> str:
    # Call your local LLM (e.g., Ollama, llama.cpp)
    return response
```

2. Pass to extraction/checking:
```python
requirements = extract_requirements(doc, use_llm=True, llm_adapter=llm_adapter)
results = check_compliance(reqs, design, retriever, use_llm=True, llm_adapter=llm_adapter)
```

## Next Iterations

### Phase 2: Enhanced Extraction
- [ ] Semantic embeddings for better retrieval (local models)
- [ ] Table and figure reference extraction
- [ ] Cross-reference resolution

### Phase 3: LLM Integration
- [ ] Local LLM support (Ollama, llama.cpp)
- [ ] Configurable model selection
- [ ] Confidence scoring

### Phase 4: UI & Workflow
- [ ] Streamlit web interface
- [ ] Interactive requirement editing
- [ ] Version comparison

### Phase 5: Enterprise Features
- [ ] Multi-document projects
- [ ] Requirement lifecycle tracking
- [ ] Integration with issue trackers

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

---

*SpecGuard - Making compliance verification systematic and auditable.*
