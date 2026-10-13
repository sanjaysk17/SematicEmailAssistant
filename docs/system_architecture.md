# System Architecture & Technical Specifications

## 1. High-Level System Architecture

The **AI-Powered Email Writing Assistant with Semantic Analysis** implements a research-oriented, three-stage pipeline: **UNDERSTAND**, **GENERATE**, and **VERIFY**.

```
                           +-------------------------------------+
                           |            USER INTERFACE           |
                           |   Streamlit (Multi-Page Modular)    |
                           |   Optional Chrome Extension (MV3)   |
                           +------------------+------------------+
                                              | HTTP REST / JSON
                                              v
                           +-------------------------------------+
                           |         BACKEND API (FastAPI)       |
                           |  POST /api/analyze                  |
                           |  POST /api/generate                 |
                           |  POST /api/validate                 |
                           |  POST /api/evaluate                 |
                           |  GET  /api/health                   |
                           +------------------+------------------+
                                              |
      +---------------------------------------+---------------------------------------+
      |                                       |                                       |
      v                                       v                                       v
[1. UNDERSTAND]                         [2. GENERATE]                           [3. VERIFY]
Text Preprocessor                       Prompt Construction                     Output Validation Engine
  |                                       - Direct Prompt (Base A)                - Intent Preservation Check
Information Extractor                     - Structured Prompt (Base B)            - SBERT Semantic Cosine
  - Linguistic Entity Parsing             - Semantic-Guided Prompt (Proposed)     - Tone Suitability Check
  - Causal Reason Extractor               |                                       - Completeness Matrix
  - Temporal & Duration Extractor       LLM Provider Abstraction                  - Hallucination / Unsupported
  |                                       - Mock Provider (Deterministic)         - Language Quality & Formats
Semantic Intent Classifier                - OpenAI Provider (gpt-4o)              |
  - Exemplar Vector Clusters              - Local Provider (Ollama / vLLM)      Validation Verdict:
  - Open-Ended Intent Fallback            - Gemini Provider (gemini-1.5)          - PASS
  |                                                                               - NEEDS_REVIEW
Tone & Formality Analyzer                                                         - VALIDATION_ERROR
  |
Structured Communication Representation
  (Extensible Pydantic Schema)
      |                                       |                                       |
      +---------------------------------------+---------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           |       PERSISTENCE & SECURITY        |
                           | - Relational DB: SQLite / Postgres  |
                           | - Security Sanitizer (Prompt Armor) |
                           | - Privacy: Masking & No-Store Mode  |
                           | - Optional Vector Context Store     |
                           +-------------------------------------+
```

---

## 2. Core Architectural Pipeline

### Stage 1: UNDERSTAND (Semantic Interpretation)
1. **Input Normalization**: NFKC Unicode normalization, safe whitespace condensation, and non-destructive retention of negation markers (`not`, `never`), numbers, and dates.
2. **Information Extraction**:
   - Locates recipient and audience entities from prepositions and honorific titles.
   - Extracts causal justifications (`because of`, `due to`, medical conditions, technical reasons).
   - Extracts temporal references, explicit deadlines, and requested durations (`two additional days`, `Thursday 3 PM`).
   - Gathers explicit facts and negative constraints.
   - Flags ambiguities and proposes clarification questions when vital information is absent.
3. **Intent Interpretation**:
   - Matches semantic clusters using embedding space cosine similarities.
   - Supports open-ended intents without forcing closed categories.
   - Identifies high ambiguity requests as `uncertain_open_ended`.
4. **Tone Analysis**:
   - Evaluates requested register across formality, politeness, assertiveness, and urgency.
5. **Output**: `StructuredRequirement` Pydantic model.

### Stage 2: GENERATE (Constraint-Guided Generation)
1. **Prompt Construction**:
   - `PromptBuilder` formats the original requirement, verified representation, constraints, and instructions not to invent facts.
   - Treats user text strictly as read-only data to defend against prompt injection.
2. **Provider Abstraction**:
   - Decouples generation backend via `BaseLLMProvider`.
   - Supports `MockLLMProvider` (offline reproducible testing), `OpenAILLMProvider`, `LocalLLMProvider`, and `GeminiLLMProvider`.
   - Returns structured subject, body, generation metadata, and elapsed latency.

### Stage 3: VERIFY (Output Validation Engine)
1. **Intent Preservation**: Confirms communication purpose using semantic signal matching.
2. **Semantic Alignment**: Measures cosine similarity using SBERT or calibrated subword N-gram embeddings.
3. **Tone Suitability**: Analyzes politeness and formality markers in generated draft vs. requested register.
4. **Key-Information Completeness**: Evaluates each structured parameter as `Present`, `Missing`, `Contradicted`, `Not applicable`, or `Uncertain`.
5. **Unsupported Information**: Identifies unprompted commitments (e.g. unverified attachments, unprompted overtime guarantees).
6. **Overall Decision**: Emits `PASS`, `NEEDS_REVIEW`, or `VALIDATION_ERROR`.

---

## 3. Database Schema Overview

The relational database (`SQLite` default, `PostgreSQL` compatible) comprises:
- `users`: User identity and account records.
- `user_preferences`: Stored writing style, default tone, and language preferences.
- `email_requests`: Raw requirements, contextual threads, and input timestamps.
- `semantic_analyses`: Structured representations, model identifiers, and confidence values.
- `generated_emails`: Subject, body, generation settings, and model metadata.
- `validation_results`: Metric values, missing details, contradictions, and validation status.
- `experiments`: Master experiment configurations and baseline parameters.
- `experiment_results`: Per-example outputs, latency measurements, and metric scores.

*Privacy Compliance Note*: Setting `DISABLE_EMAIL_CONTENT_PERSISTENCE=true` replaces raw requirements and generated bodies with `[PERSISTENCE_DISABLED_FOR_PRIVACY]` in database records.
