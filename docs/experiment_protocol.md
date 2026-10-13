# Experimental Evaluation Protocol & Benchmark Design

## 1. Experimental Systems & Baselines

To rigorously evaluate the research question, three systems are tested under identical inputs:

### Baseline A: Direct LLM Generation
- **Pipeline**: Requirement $\to$ Direct Prompt $\to$ LLM $\to$ Email Draft.
- **Hypothesis**: Direct generation will produce fluent text but omit specific constraints (such as explicit durations or dates) and add generic conversational filler.

### Baseline B: Structured-Prompt Generation
- **Pipeline**: Requirement $\to$ Templatized Prompt with Headers $\to$ LLM $\to$ Email Draft.
- **Hypothesis**: Adding structured output instructions will improve formatting and reduce conversational filler, but lacking upstream semantic parsing will still permit omissions when requirements are unstructured or complex.

### Proposed System: Semantic-Guided Architecture
- **Pipeline**: Requirement $\to$ Semantic Interpretation $\to$ Dynamic Structured Representation $\to$ Constraint-Guided Prompt Construction $\to$ LLM $\to$ Output Validation Engine.
- **Hypothesis**: Explicit semantic interpretation extracts exact parameters, prompt boundaries prevent unverified commitments, and the validator flags discrepancies before user delivery.

---

## 2. Dataset Design & Inclusion Criteria

The benchmark dataset (`data/synthetic_examples.csv`) incorporates varied domains:
- **Academic**: Project extension requests, recommendation letters, conference grant inquiries.
- **Workplace**: Sick leave notices, equipment requests, meeting reschedules, status reports.
- **Customer Service**: Billing disputes, duplicate charges, refund requests.
- **Employment**: Interview follow-ups, recruiter inquiries.
- **Uncertain / Open-Ended**: Ambiguous or poorly specified requests designed to test ambiguity detection.

All synthetic examples are explicitly labeled `is_synthetic = True` to maintain scientific integrity.

---

## 3. Evaluation Procedure

1. **Held-Out Test Set**: Benchmarking executes on the designated `test` split.
2. **Zero Ingestion Bias**: Generation prompts do not receive the ground truth labels or validator answers.
3. **Double-Blind Scoring**: The Output Validation Engine evaluates all three systems using identical metric calculations without favoring the proposed system.
4. **Export Artifacts**: Outputs, metric values, and error diagnostic logs are saved to `data/experiment_results_<ID>.csv`.

---

## 4. Error Taxonomy for Failure Analysis

Failed outputs are categorized into four diagnostic classes:
1. **Omission Failure**: Required action, duration, deadline, or fact is missing from the output.
2. **Hallucination / Unsupported Addition**: Email promises attachments, overtime, or claims not justified by input.
3. **Tone Divergence**: Formality or politeness markers fail requested criteria.
4. **Intent Drift**: Core communication purpose is altered or misinterpreted.
