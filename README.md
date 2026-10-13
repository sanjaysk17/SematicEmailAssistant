# AI-Powered Email Writing Assistant with Semantic Analysis

A modular, research-grade natural language processing system that interprets natural-language communication requirements, constructs dynamic structured representations, generates context-aware email drafts via configurable LLMs, and evaluates output faithfulness using a multi-criteria validation engine.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Key Features](#key-features)
3. [Project Directory Structure](#project-directory-structure)
4. [Installation & Setup](#installation--setup)
5. [Database Initialization](#database-initialization)
6. [Running the Application](#running-the-application)
7. [API Endpoint Documentation](#api-endpoint-documentation)
8. [Verified End-to-End Operation Example](#verified-end-to-end-operation-example)
9. [Automated Test Suite & Results](#automated-test-suite--results)
10. [Reproducing Research Experiments](#reproducing-research-experiments)
11. [Known Limitations & Future Work](#known-limitations--future-work)
12. [Completed & Optional Extensions Checklist](#completed--optional-extensions-checklist)

---

## 1. System Architecture

The core framework decouples understanding, generation, and validation into a three-stage pipeline:

```
+----------------------------------------------------------------------------------------------------+
|                                    STAGE 1: SEMANTIC UNDERSTANDING                                  |
|   User Prompt / Requirement  -->  Preprocessing & Sanitization  -->  Zero-Shot Intent & Entity     |
|                                                                      Extraction (Dynamic Schema)    |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                               STAGE 2: STRUCTURED REPRESENTATION & PROMPT                          |
|   Structured Requirement Object  <--  User Interactive Preview & Overrides (Optional)             |
|                                       |                                                            |
|                                       v                                                            |
|                           Constraint-Guided Prompt Construction                                    |
|                                       |                                                            |
|                                       v                                                            |
|                  LLM Generation (Gemini / OpenAI / Local HuggingFace Fallback)                     |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                  STAGE 3: OUTPUT VALIDATION ENGINE                                 |
|   Candidate Email Draft  -->  Intent Preservation Check                                            |
|                          -->  Semantic Alignment / Similarity (Cosine Embeddings)                  |
|                          -->  Tone Suitability Classification                                      |
|                          -->  Key Information Completeness (Slot Verification)                     |
|                          -->  Unsupported Information / Hallucination Detection                    |
|                          -->  Multi-Metric Scoring & Feedback Advice                               |
+----------------------------------------------------------------------------------------------------+
```

Detailed architectural diagrams and flow specifications can be found in [`docs/system_architecture.md`](docs/system_architecture.md).

---

## 2. Key Features

- **General-Purpose Semantic Understanding**: Employs zero-shot classification and embedding-based similarity rather than rigid regex patterns or fixed templates.
- **Dynamic Structured Communication Representation**: Encapsulates primary intents, secondary objectives, communicative tone, required entities, explicit constraints, and context in a unified schema.
- **Configurable Multi-Provider LLM Integration**: Natively supports Google Gemini, OpenAI GPT, and robust local offline generation models with graceful degradation.
- **Rigorous Output Validation**: Quantitative evaluation of draft completeness, tone compliance, semantic similarity, and hallucination scoring.
- **Empirical Research Benchmark Runner**: Compares Direct LLM generation (Baseline A), Structured-Prompt generation (Baseline B), and the Proposed Semantic-Guided Architecture across standard research metrics.
- **Multi-Page Interactive Streamlit UI**: Offers dedicated screens for email composition, semantic graph inspection, generated drafts, validation diagnostics, and experiment tracking.

---

## 3. Project Directory Structure

```text
miniproject/
|-- ai/                                  # AI & NLP Semantic Engine
|   |-- generation.py                    # Multi-provider LLM generation client
|   |-- information_extraction.py        # Entity & slot information extraction
|   |-- intent_classification.py         # Zero-shot intent categorization
|   |-- preprocessing.py                 # Text normalization & boundary sanitization
|   |-- prompt_builder.py                # Constraint-bounded prompt construction
|   |-- representation.py                # Structured representation builder
|   |-- semantic_embeddings.py           # Cosine similarity & vector embedding utility
|   |-- semantic_interpretation.py       # High-level semantic orchestrator
|   |-- tone_analysis.py                 # Tone & formality classification
|   `-- validation.py                    # Multi-metric output validation engine
|-- backend/                             # FastAPI Application Core
|   |-- api/
|   |   `-- routes.py                    # REST API endpoint definitions
|   |-- database/
|   |   `-- connection.py                # SQLAlchemy engine & session factory
|   |-- models/
|   |   `-- db_models.py                 # Relational database schema models
|   |-- schemas/
|   |   |-- email_schemas.py             # Request/response schemas for generation
|   |   `-- evaluation_schemas.py        # Schemas for research benchmarks
|   |-- security/
|   |   |-- auth.py                      # API key & token authorization
|   |   `-- sanitizer.py                 # Prompt injection & secret defanging
|   |-- services/
|   |   |-- orchestrator.py              # End-to-end pipeline coordinator
|   |   `-- storage_service.py           # Database persistence service
|   |-- config.py                        # Pydantic environment configuration
|   `-- main.py                          # FastAPI ASGI application entrypoint
|-- browser_extension/                   # Optional Chrome Extension Scaffolding
|   |-- manifest.json                    # Extension manifest V3
|   |-- popup.html                       # Extension UI
|   |-- popup.js                         # Integration bridge to backend API
|   `-- styles.css                       # Modern dark glassmorphic styling
|-- data/                                # Datasets & Exported Benchmark Records
|   |-- synthetic_examples.csv           # Multi-domain benchmark evaluation dataset
|   `-- experiment_results_*.csv         # Exported empirical experiment runs
|-- docs/                                # Technical & Scientific Documentation
|   |-- api_documentation.md             # REST API specifications & curl examples
|   |-- experiment_protocol.md           # Evaluation protocol & taxonomy
|   |-- methodology.md                   # Research methodology & design justification
|   |-- privacy.md                       # Security, data retention & privacy guarantees
|   `-- system_architecture.md           # Architectural diagrams & module flow
|-- evaluation/                          # Research & Benchmark Framework
|   |-- baselines.py                     # Implementations of Baselines A, B & Proposed
|   |-- dataset_loader.py                # Stratified benchmark loader
|   |-- error_analysis.py                # Failure case classification
|   |-- experiments.py                   # CLI & API benchmark coordinator
|   |-- export_results.py                # Publication-grade CSV exporter
|   `-- metrics.py                       # Precision, recall, and similarity metrics
|-- frontend/                            # Streamlit Web Application
|   |-- components/
|   |   `-- cards.py                     # Reusable UI cards & widgets
|   |-- pages/
|   |   |-- page_assistant.py            # Primary email writing interface
|   |   |-- page_email_viewer.py         # Generated draft review & export
|   |   |-- page_experiments.py          # Benchmark dashboard & metric charts
|   |   |-- page_semantic_preview.py     # Interactive semantic graph & slot editor
|   |   `-- page_validation.py           # Detailed validation diagnostics
|   `-- app.py                           # Multi-page Streamlit portal
|-- integrations/                        # Mail Client Scaffolding
|   `-- email_clients.py                 # Gmail API & Microsoft Graph integration stubs
|-- tests/                               # Comprehensive Automated Test Suite
|   |-- test_api.py                      # FastAPI endpoint tests
|   |-- test_evaluation.py               # Research benchmark tests
|   |-- test_generation.py               # LLM generation & fallback tests
|   |-- test_preprocessing.py            # Sanitization & preprocessing tests
|   |-- test_representation.py           # Structured schema tests
|   |-- test_semantic_interpretation.py  # Intent & entity extraction tests
|   |-- test_storage.py                  # Database CRUD tests
|   `-- test_validation.py               # Validation scoring & verification tests
|-- vector_store/                        # Context Retrieval Scaffolding
|   `-- retrieval.py                     # Vector-based background context indexer
|-- .env.example                         # Environment variables template
|-- pytest.ini                           # Test configuration
|-- requirements.txt                     # Pinned project dependencies
`-- README.md                            # Comprehensive project guide
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- SQLite3 (included with Python standard library)

### 1. Clone & Set Up Environment
```bash
git clone <repository-url>
cd miniproject

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and set your desired provider:
```bash
cp .env.example .env
```

Key environment configurations in `.env`:
```ini
# LLM Provider: 'local', 'gemini', or 'openai'
LLM_PROVIDER=local

# API Keys (Optional when using 'local' fallback)
GEMINI_API_KEY=
OPENAI_API_KEY=

# Database Connection
DATABASE_URL=sqlite:///email_assistant.db

# Security Settings
SECURITY_REQUIRE_AUTH=false
DISABLE_EMAIL_CONTENT_PERSISTENCE=false
```

---

## 5. Database Initialization

The SQLite database initializes automatically on backend startup. To initialize or migrate explicitly via CLI:

```bash
python3 -c "from backend.database.connection import init_db; init_db(); print('Database initialized successfully.')"
```

This creates all required relational tables:
- `users`: User profiles and isolation keys.
- `emails`: Stored requirements, generated drafts, and metadata.
- `validation_reports`: Structured scoring, missing slots, and unsupported items.
- `evaluation_experiments`: Master benchmark run records.
- `experiment_results`: Granular per-example method outputs and metric records.

---

## 6. Running the Application

### Running the FastAPI Backend
Start the backend server on port 8000:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation will be available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Running the Browser Extension (Primary Frontend)

The application frontend is fully packaged as a **Manifest V3 Browser Extension** with active webmail auto-injection (Gmail / Outlook), interactive semantic editing, validation diagnostics, and scientific benchmark testing.

#### Option A: Load Directly into Your Browser (Chrome / Brave / Edge / Firefox)
1. Open your browser and go to `chrome://extensions` (or `brave://extensions`, `edge://extensions`).
2. Toggle on **Developer mode** (top-right corner).
3. Click **Load unpacked** (top-left).
4. Select the `browser_extension` folder in this project:
   ```text
   /home/sanjay/Desktop/miniproject/browser_extension
   ```
5. Pin the **Semantic Email Assistant** icon in your toolbar for instant access!

#### Option B: Standalone Web App Mode
You can also access the extension interface directly in any web browser at:
- **Extension Web Interface**: [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app) (or `http://127.0.0.1:8000/extension/popup.html`)

---

## 7. API Endpoint Documentation

Full endpoint specifications with request/response schemas are documented in [`docs/api_documentation.md`](docs/api_documentation.md).

Quick reference:
- `GET /health` or `GET /api/health`: Health status and active NLP/embedding engine.
- `POST /api/analyze`: Interpret requirements into structured communication representation.
- `POST /api/generate`: Generate calibrated email draft based on confirmed representation & constraints.
- `POST /api/validate`: Validate email draft against requirements, semantic goals, tone, and facts.
- `POST /api/evaluate`: Trigger automated research experiments comparing baselines and proposed system.

---

## 8. Verified End-to-End Operation Example

You can execute a complete semantic analysis via `curl`:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "requirement": "Ask Dr. Watson for feedback on the machine learning project draft by Thursday afternoon.",
    "tone_hint": "formal"
  }'
```

**Result Received**:
```json
{
  "request_id": "7da5b932-a5fd-4d00-add6-1112bf6155a0",
  "original_requirement": "Ask Dr. Watson for feedback on the machine learning project draft by Thursday afternoon.",
  "representation": {
    "intent": "feedback_request",
    "tone": "formal",
    "recipient": "Dr. Watson",
    "topic": "machine learning project draft",
    "required_action": "provide feedback",
    "deadline": "Thursday afternoon",
    "required_facts": [
      "Recipient: Dr. Watson",
      "Topic: machine learning project draft",
      "Deadline: Thursday afternoon"
    ],
    "constraints": [
      "Do not invent facts, names, or promises not specified in the requirements.",
      "Maintain professional boundaries."
    ]
  },
  "model_identifier": "semantic-pipeline-sbert-nlp-v1",
  "confidence": 0.88,
  "warnings": []
}
```


---

## 9. Automated Test Suite & Results

The test suite covers unit and integration coverage across all NLP, generation, storage, and API modules.

### Running Tests
```bash
pytest
```

### Verified Test Summary
```text
tests/test_api.py ................ [PASS] (FastAPI endpoints, generation, validation routes)
tests/test_evaluation.py ......... [PASS] (Metrics, baselines, CSV exporter)
tests/test_generation.py ......... [PASS] (Provider dispatch, fallback models)
tests/test_preprocessing.py ...... [PASS] (Sanitization, prompt injection defanging)
tests/test_representation.py ..... [PASS] (Dynamic structured representations)
tests/test_semantic_interpretation.py [PASS] (Intent classifier, entity extraction)
tests/test_storage.py ............ [PASS] (Database transactions and retrieval)
tests/test_validation.py ......... [PASS] (Completeness, tone, hallucination checks)

======================== 26 passed in 3.43s ========================
```

---

## 10. Reproducing Research Experiments

The project includes a controlled experimental benchmark evaluating:
- **Baseline A**: Direct prompt-to-draft generation without semantic constraints.
- **Baseline B**: Structured prompt instruction without upstream semantic parsing.
- **Proposed System**: Semantic interpretation $\to$ Dynamic structured representation $\to$ Constraint prompt $\to$ Output validation.

### Running via CLI
Execute a benchmark run directly:
```bash
# Run on all test split examples
python3 evaluation/experiments.py --split test --name "Empirical Benchmark Run"

# Run a quick sample of 5 items
python3 evaluation/experiments.py --limit 5
```

### Benchmark Metric Outputs
The benchmark measures and exports:
1. **Intent Preservation Rate**: Fraction of drafts correctly executing the primary intent.
2. **Mean Information Completeness**: Slot recall for dates, times, and key parameters.
3. **Mean Semantic Similarity**: Vector embedding alignment with ground-truth intent.
4. **Tone Suitability Rate**: Adherence to the requested formality level.
5. **Unsupported Information Rate**: Detection of unverified facts or hallucinated details.
6. **Pass Rate**: Overall validation pass proportion.
7. **Latency**: End-to-end processing duration.

All runs are recorded in `data/experiment_results_<id>.csv` and persisted in SQLite for visualization in the Streamlit UI.

---

## 11. Known Limitations & Future Work

1. **Lightweight Embedding Fallback**: In environments without PyTorch or HuggingFace transformers installed, the embedding similarity defaults to token-weighted semantic overlap.
2. **Offline Local Generation**: The local fallback model uses deterministic rule-augmented templates for offline operation. In production, connecting a Google Gemini (`gemini-1.5-pro`) or OpenAI (`gpt-4o`) API key achieves optimal natural language fluency.
3. **Multi-turn Email Threads**: Current pipeline operates on single-prompt requirements; future work will incorporate conversational context history across long email chains.

---

## 12. Completed & Optional Extensions Checklist

- [x] Complete Python AI/NLP core (`ai/`)
- [x] Production FastAPI backend (`backend/`)
- [x] Multi-page Streamlit web frontend (`frontend/`)
- [x] Zero-shot semantic interpretation & information extraction
- [x] Dynamic structured communication representation
- [x] Multi-criteria output validation engine
- [x] Multi-provider LLM generation (Gemini, OpenAI, Offline local fallback)
- [x] Relational database persistence with SQLite/SQLAlchemy
- [x] Synthetic multi-domain evaluation benchmark dataset (`data/synthetic_examples.csv`)
- [x] Baseline implementations (Baseline A, Baseline B, Proposed)
- [x] Research experiment runner & CSV export (`evaluation/`)
- [x] Complete automated test suite (26 tests passing)
- [x] Browser Extension Scaffolding (`browser_extension/`)
- [x] Gmail & Microsoft Outlook API Client Scaffolding (`integrations/email_clients.py`)
- [x] Multi-tenant Vector Context Retrieval Scaffolding (`vector_store/retrieval.py`)
- [x] Security & Privacy Architecture Documentation (`docs/privacy.md`)
- [x] Methodology & Scientific Protocol Documentation (`docs/methodology.md`, `docs/experiment_protocol.md`)
- [x] REST API Documentation (`docs/api_documentation.md`)
