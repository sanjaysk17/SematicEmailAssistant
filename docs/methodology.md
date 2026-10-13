# Research Methodology: Semantic-Guided Email Synthesis

## 1. Research Motivation & Theoretical Framework

Large Language Models (LLMs) demonstrate high fluency and grammatical competence when producing natural language text. However, fluency does not guarantee fidelity:
1. **Omission**: Models frequently omit essential constraints (e.g., exact deadlines or requested durations).
2. **Hallucination / Drift**: Models frequently inject polite but unsupported commitments (e.g., promising unmentioned attachments or unverified delivery dates).
3. **Tone Mismatch**: Models may revert to generic cheerfulness when a firm, apologetic, or respectful academic tone is demanded.

This research investigates whether an explicit **UNDERSTAND -> GENERATE -> VERIFY** architecture measurably improves the preservation of communication requirements over unguided direct generation and template-guided generation.

---

## 2. Research Question

> **Does explicit semantic interpretation, structured representation, and post-generation validation improve the preservation of user communication requirements compared with direct LLM generation and structured-prompt-based generation?**

---

## 3. Mathematical Metric Formulations

### 1. Intent Preservation Rate ($R_{\text{intent}}$)
Measures the proportion of generated emails that fulfill the primary communication action:
$$R_{\text{intent}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{Intent Preserved}_i)$$
Where $\mathbb{I}(\cdot) \in \{0, 1\}$.

### 2. Key-Information Completeness ($C_{\text{mean}}$)
Given a set of $M_i$ required communication elements $\{f_1, f_2, \dots, f_{M_i}\}$ for scenario $i$:
$$C_i = \frac{1}{M_i} \sum_{j=1}^{M_i} \mathbb{I}(f_j == \text{"Present"})$$
$$C_{\text{mean}} = \frac{1}{N} \sum_{i=1}^{N} C_i$$

### 3. Semantic Alignment ($\text{Sim}_{\text{cosine}}$)
Cosine similarity between normalized embedding vectors of requirement $\mathbf{r}$ and generated email $\mathbf{e}$:
$$\text{Sim}_{\text{cosine}}(\mathbf{r}, \mathbf{e}) = \frac{\mathbf{r} \cdot \mathbf{e}}{\|\mathbf{r}\|_2 \|\mathbf{e}\|_2}$$

### 4. Tone Suitability Rate ($R_{\text{tone}}$)
Measures adherence to requested formality and politeness criteria:
$$R_{\text{tone}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{Tone Matches}_i)$$

### 5. Unsupported-Information Rate ($R_{\text{unsupported}}$)
Measures the fraction of outputs containing at least one unverified claim, commitment, or attachment:
$$R_{\text{unsupported}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{Count}(\text{Unsupported}_i) > 0)$$

---

## 4. NLP Components & Fallback Architecture

| Module | Primary NLP Engine | Operational Fallback | Academic Rationale |
| :--- | :--- | :--- | :--- |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) | Scikit-Learn Hybrid Subword N-Gram Cosine | Guarantees deterministic, 100% offline evaluation without external network dependencies. |
| **Linguistic Extraction** | spaCy dependency parser & entities | Regex nominal & causal clause parsing | Prevents reliance on hardcoded single-situation templates while ensuring fast execution. |
| **Tone Analysis** | Multidimensional linguistic marker matrix | Formality / politeness ratio analyzer | Distinguishes politeness, urgency, assertiveness, and formality as distinct concepts. |
