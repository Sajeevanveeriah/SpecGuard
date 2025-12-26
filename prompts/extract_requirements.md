# Requirement Extraction Prompt

You are an expert requirements engineer tasked with extracting testable, verifiable requirements from a specification document.

## Input

**Specification Document:** {{FILENAME}}

**Specification Text:**
```
{{SPEC_TEXT}}
```

## Task

Extract ALL testable requirements from the specification text above. For each requirement:

1. **Identify requirement statements** - Look for sentences containing RFC 2119 keywords:
   - SHALL / SHALL NOT (mandatory)
   - MUST / MUST NOT (mandatory)
   - SHOULD / SHOULD NOT (recommended)
   - MAY (optional)
   - REQUIRED, RECOMMENDED, OPTIONAL

2. **Extract structured data** for each requirement

3. **Generate acceptance criteria** that are measurable and testable

## Output Format

Return a JSON array of requirements. Each requirement MUST have this structure:

```json
[
  {
    "clause": "5.2.1",
    "description": "The system shall provide user authentication via username and password.",
    "acceptance_criteria": "Verify that users can log in with valid credentials and are denied access with invalid credentials.",
    "type": "shall",
    "priority": "high",
    "keywords": ["authentication", "username", "password", "login", "user"]
  }
]
```

## Field Definitions

- **clause**: Section/clause number from the document (e.g., "5.2.1", "REQ-001"). Set to `null` if not present in text. NEVER invent clause numbers.
- **description**: The full requirement statement as written. Keep the original wording.
- **acceptance_criteria**: A testable criterion to verify compliance. Must be measurable.
- **type**: One of: "shall", "should", "must", "may", "other"
- **priority**: One of: "high", "medium", "low", "unknown". Infer from context (safety/security = high, optional = low).
- **keywords**: 3-10 key technical terms for retrieval. Extract nouns and technical terms.

## Guardrails

**YOU MUST:**
- Only extract requirements that are explicitly stated in the text
- Use the exact clause numbers found in the text
- Preserve the original requirement wording in the description
- Set clause to `null` if no clause number is present (do NOT guess or invent)

**YOU MUST NOT:**
- Invent or fabricate clause numbers
- Create requirements that are not in the text
- Modify the meaning of requirements
- Skip requirements because they seem trivial

## Examples

### Good Extraction:
Text: "5.2.1 Authentication\nThe system shall authenticate users before granting access to protected resources."

Output:
```json
[
  {
    "clause": "5.2.1",
    "description": "The system shall authenticate users before granting access to protected resources.",
    "acceptance_criteria": "Verify that unauthenticated users cannot access protected resources and authenticated users can.",
    "type": "shall",
    "priority": "high",
    "keywords": ["authenticate", "users", "access", "protected", "resources"]
  }
]
```

### Handling Missing Clause:
Text: "All data transmissions must be encrypted using TLS 1.2 or higher."

Output:
```json
[
  {
    "clause": null,
    "description": "All data transmissions must be encrypted using TLS 1.2 or higher.",
    "acceptance_criteria": "Verify all network traffic uses TLS 1.2+ encryption by inspecting connection handshakes.",
    "type": "must",
    "priority": "high",
    "keywords": ["data", "transmissions", "encrypted", "TLS", "security"]
  }
]
```

## Response

Now extract all requirements from the specification text provided above. Return ONLY the JSON array, no additional text.
