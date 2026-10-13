# REST API Documentation

The **AI-Powered Email Writing Assistant** backend is built on FastAPI, providing interactive documentation via Swagger UI at `http://127.0.0.1:8000/docs` and OpenAPI JSON at `http://127.0.0.1:8000/openapi.json`.

---

## Base URL & Endpoints Overview

- **Base URL**: `http://127.0.0.1:8000/api`
- **Root URL**: `http://127.0.0.1:8000/`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` / `/api/health` | Service health status and active embedding engine |
| `POST` | `/api/analyze` | Parse natural language requirements into dynamic structured representation |
| `POST` | `/api/generate` | Generate email draft based on confirmed representation & constraints |
| `POST` | `/api/validate` | Evaluate generated draft against semantic goals, tone, and facts |
| `POST` | `/api/evaluate` | Execute research benchmark comparing Baselines A, B, and Proposed System |

---

## 1. Health Check

### `GET /api/health` or `GET /health`

**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "llm_provider": "local",
  "database_status": "connected",
  "embedding_engine": "Scikit-Learn Subword/Word TF-IDF Cosine Engine"
}
```

---

## 2. Stage 1: Semantic Analysis

### `POST /api/analyze`

Parses natural language requirements into an extensible structured communication representation.

**Request Body (`AnalyzeRequest`)**:
```json
{
  "requirement": "Tell Professor Jones I will miss tomorrow class due to fever and ask for lecture notes.",
  "context": "Course: Senior Thesis (CS 499)",
  "recipient_hint": "Professor Jones",
  "tone_hint": "formal",
  "desired_length": "medium",
  "language": "English",
  "explicit_restrictions": []
}
```

**Response (`200 OK` - `AnalyzeResponse`)**:
```json
{
  "request_id": "7da5b932-a5fd-4d00-add6-1112bf6155a0",
  "original_requirement": "Tell Professor Jones I will miss tomorrow class due to fever and ask for lecture notes.",
  "representation": {
    "intent": "absence_notification",
    "tone": "polite and professional",
    "recipient": "Professor Jones",
    "topic": "miss tomorrow class due to fever",
    "required_action": "ask for lecture notes",
    "reason": "fever",
    "deadline": "tomorrow",
    "requested_duration": null,
    "required_facts": [
      "Reason: fever",
      "Action: ask for lecture notes",
      "Temporal reference: tomorrow"
    ],
    "constraints": [
      "Do not invent facts, names, or promises not specified in the requirements.",
      "Maintain professional boundaries and do not commit to unverified timelines."
    ],
    "uncertainties": [],
    "inferred_details": {
      "has_causal_justification": true,
      "has_temporal_anchor": true,
      "linguistic_backend": "spaCy"
    },
    "clarification_questions": [],
    "user_confirmed": false,
    "custom_attributes": {}
  },
  "model_identifier": "semantic-pipeline-sbert-nlp-v1",
  "confidence": 0.88,
  "analysis_time_ms": 115.4,
  "warnings": []
}
```

---

## 3. Stage 2: Email Generation

### `POST /api/generate`

Drafts an email based on confirmed structured representation and constraint-guided prompt construction.

**Request Body (`GenerateRequest`)**:
```json
{
  "requirement": "Tell Professor Jones I will miss tomorrow class due to fever and ask for lecture notes.",
  "representation": {
    "intent": "absence_notification",
    "tone": "formal",
    "recipient": "Professor Jones",
    "topic": "miss tomorrow class due to fever",
    "required_action": "ask for lecture notes",
    "reason": "fever",
    "deadline": "tomorrow",
    "requested_duration": null,
    "required_facts": ["absent tomorrow", "fever", "request notes"],
    "constraints": ["Keep concise", "Formal tone"],
    "uncertainties": [],
    "inferred_details": {},
    "clarification_questions": [],
    "user_confirmed": true,
    "custom_attributes": {}
  },
  "context": "Course: Senior Thesis (CS 499)",
  "desired_length": "medium",
  "language": "English",
  "system_version": "proposed"
}
```

**Response (`200 OK` - `GenerateResponse`)**:
```json
{
  "subject": "Absence Notification and Notes Request - Tomorrow's Class",
  "body": "Dear Professor Jones,\n\nI am writing to inform you that I will be unable to attend tomorrow's class due to fever.\n\nCould you please let me know if lecture notes or materials will be available so I can stay caught up?\n\nThank you for your understanding.\n\nSincerely,\n[Your Name]",
  "system_version": "proposed",
  "model_identifier": "local-fallback-generator",
  "generation_time_ms": 95.2,
  "raw_prompt_used": "...",
  "email_id": "42d2a45a-c6b7-4fc9-b632-15f532a83852"
}
```

---

## 4. Stage 3: Output Validation

### `POST /api/validate`

Evaluates the generated email against semantic goals, tone suitability, completeness, and unsupported statements.

**Request Body (`ValidateRequest`)**:
```json
{
  "requirement": "Tell Professor Jones I will miss tomorrow class due to fever and ask for lecture notes.",
  "representation": { ... },
  "email_subject": "Absence Notification and Notes Request",
  "email_body": "Dear Professor Jones, I will be unable to attend tomorrow's class due to fever...",
  "generated_email_id": "42d2a45a-c6b7-4fc9-b632-15f532a83852"
}
```

**Response (`200 OK` - `OutputValidationReport`)**:
```json
{
  "overall_status": "Valid",
  "intent_preserved": true,
  "intent_explanation": "Draft accurately conveys the absence notification and request for notes.",
  "semantic_similarity": 0.88,
  "similarity_model": "Scikit-Learn Subword/Word TF-IDF Cosine Engine",
  "tone_suitable": true,
  "tone_detected": "formal",
  "tone_explanation": "Tone aligns with the requested formal and polite register.",
  "completeness_score": 1.0,
  "fact_evaluations": [],
  "missing_details": [],
  "contradictions": [],
  "unsupported_details": [],
  "language_quality_issues": [],
  "actionable_warnings": [],
  "metric_definitions": {},
  "validation_time_ms": 42.1
}
```

---

## 5. Research Evaluation Benchmark

### `POST /api/evaluate`

Executes automated controlled experiments comparing Baselines A, B, and the Proposed System across test splits.

**Request Body (`ExperimentRunRequest`)**:
```json
{
  "experiment_name": "Comparative Benchmark",
  "methods_to_run": ["baseline_a", "baseline_b", "proposed_system"],
  "max_examples": 10,
  "split_filter": "test"
}
```

**Response (`200 OK` - `ExperimentRunResponse`)**:
```json
{
  "experiment_id": "exp_39c14a9",
  "experiment_name": "Comparative Benchmark",
  "total_examples_evaluated": 10,
  "methods_evaluated": ["baseline_a", "baseline_b", "proposed_system"],
  "aggregate_metrics": { ... },
  "detailed_results": [ ... ],
  "runtime_seconds": 1.45,
  "export_csv_filename": "data/experiment_results_exp_39c14a9.csv"
}
```
