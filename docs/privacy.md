# Security, Privacy & Data Governance Architecture

## 1. Security Architecture Principles

The **AI-Powered Email Writing Assistant** adheres to privacy-by-design standards:

### 1. Zero Automatic Dispatch Guarantee
- Under no circumstance does the system send an email automatically.
- Drafts are created exclusively for user review, copy, or manual mailbox dispatch.

### 2. Prompt Injection Defense
- User-supplied requirements and retrieved background contexts are treated strictly as **untrusted data**, not system instructions.
- System prompt boundaries explicitly instruct LLMs to reject prompt injection attacks (`"Ignore previous instructions"`, `"Reveal system keys"`).
- `SecuritySanitizer` defangs adversarial escape directives prior to generation.

### 3. Credential Protection
- API keys (OpenAI, Gemini) are stored strictly in environment variables (`.env`).
- Credentials are never exposed in frontend code, client responses, or committed files.
- `SecuritySanitizer.mask_credentials` scrubs secret tokens from log outputs.

### 4. Selective Data Retention (No-Store Mode)
- In sensitive operating environments, setting `DISABLE_EMAIL_CONTENT_PERSISTENCE=true` masks email requirements and draft bodies before storing in SQLite/PostgreSQL.
- Database records store only metadata (timestamps, token usage, metric scores), preventing unauthorized access to confidential corporate or personal correspondence.

### 5. Multi-Tenant Vector Isolation
- The `ContextRetrievalService` isolates context records by `user_id`.
- Retrieval operations strictly query records authorized for the requesting user, preventing cross-user data leakage.

### 6. Third-Party Provider Data Policies
- When using commercial LLMs (OpenAI, Gemini), user prompts are transmitted to provider endpoints under the respective vendor's API terms.
- For complete air-gapped data confidentiality, the application natively supports local models via Ollama or vLLM (`LLM_PROVIDER=local`).
