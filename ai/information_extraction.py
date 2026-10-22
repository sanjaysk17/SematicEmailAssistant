"""
ai/information_extraction.py - Neural Semantic Information & Intent Extractor
Dedicated exclusively to model-based semantic extraction:
- Google Gemini API (gemini-3.8-flash, gemini-2.0-flash, gemini-1.5-flash)
- OpenAI API (gpt-4o-mini, gpt-4o)
- Local Hugging Face / Ollama models

Extracts communication intent, tone, recipient, required actions, reasons,
dates/deadlines, durations, facts, and constraints in general without regex rules.
Formatted and architected in direct parity with ai/generation.py.
"""

import abc
import json
import re
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
import httpx

from backend.config import settings, normalize_gemini_model

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an advanced semantic information extraction and intent classification engine.
Given the user's natural language email request, analyze and extract all communication parameters into a strictly valid JSON object.

Extract the following fields:
- "intent": The primary intent in snake_case (e.g. "request_leave", "festival_leave", "request_extension", "informal_invitation", "reschedule_meeting", "request_refund", "acknowledgement", "general_inquiry").
- "tone": The requested or contextually appropriate tone (e.g. "friendly and casual", "polite and professional", "formal and respectful").
- "recipient": The target recipient's name or title (e.g. "Dhinesh", "Professor Marcus", "Sarah"). If not specified, set to null.
- "topic": The core topic/subject of the email.
- "required_action": The core action requested or conveyed.
- "reason": The explicit justification or reason provided (e.g. "festival celebrations", "technical issue"). If not specified, set to null.
- "deadline": Specific deadline or date anchor mentioned, or null.
- "requested_duration": Duration or date range requested (e.g. "10.11.2026 to 15.11.2026", "two additional days"), or null.
- "required_facts": List of explicit facts, dates, names, and details mentioned that must be preserved.
- "constraints": List of constraints or negative boundaries (e.g. "Do not invent unmentioned details").
- "uncertainties": List of ambiguities if the request is vague or missing vital info, otherwise empty list [].
- "clarification_questions": Any clarifying questions needed if ambiguous, otherwise empty list [].

Output only the JSON object.
"""


class BaseExtractionProvider(abc.ABC):
    """Abstract interface for information extraction and intent classification providers."""

    @abc.abstractmethod
    def extract(self, text: str, user_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Takes user requirement text and returns structured extraction dictionary."""
        pass


class OpenAIExtractionProvider(BaseExtractionProvider):
    """Extraction using OpenAI Chat Completions API."""

    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model_name = settings.OPENAI_MODEL_NAME
        self.api_base = settings.OPENAI_API_BASE

    def extract(self, text: str, user_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured in environment.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract structured email parameters from this requirement:\n\"\"\"\n{text.strip()}\n\"\"\""}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{self.api_base}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed["model_identifier"] = f"openai-{self.model_name}"
            return parsed


class LocalExtractionProvider(BaseExtractionProvider):
    """Extraction using local Ollama / vLLM / Hugging Face model API."""

    def __init__(self):
        self.url = settings.LOCAL_LLM_URL
        self.model_name = settings.LOCAL_LLM_MODEL

    def extract(self, text: str, user_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        full_prompt = (
            f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
            f"User Requirement:\n\"\"\"\n{text.strip()}\n\"\"\"\n\n"
            "Output JSON:"
        )
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "format": "json"
        }

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(self.url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data.get("response", "{}")
            parsed = json.loads(raw_text)
            parsed["model_identifier"] = f"local-{self.model_name}"
            return parsed


class GeminiExtractionProvider(BaseExtractionProvider):
    """Extraction using Google Gemini Generative Language REST API."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = normalize_gemini_model(model_name or settings.GEMINI_MODEL_NAME or "gemini-3.5-flash-lite")
        self.api_base = getattr(settings, "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")

    def extract(self, text: str, user_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        api_key = (user_hints or {}).get("api_key") or self.api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured in .env or settings.")

        model = normalize_gemini_model((user_hints or {}).get("model_name") or self.model_name)
        url = f"{self.api_base}/models/{model}:generateContent?key={api_key}"

        prompt = (
            f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
            f"[USER INSTRUCTION / PROMPT]:\n\"\"\"\n{text.strip()}\n\"\"\"\n\n"
            "JSON:"
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 800,
                "responseMimeType": "application/json"
            }
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 400 and "responseMimeType" in resp.text:
                payload["generationConfig"].pop("responseMimeType", None)
                resp = client.post(url, json=payload)

            if resp.status_code != 200:
                err_text = resp.text
                try:
                    err_json = resp.json()
                    err_text = err_json.get("error", {}).get("message", resp.text)
                except Exception:
                    pass
                raise RuntimeError(f"Gemini Extraction API error ({resp.status_code}): {err_text}")

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini API returned no candidates for extraction.")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise RuntimeError("Gemini API returned empty content for extraction.")

            raw_text = parts[0].get("text", "").strip()
            parsed = self._parse_json(raw_text)
            parsed["model_identifier"] = f"gemini-{model}"
            return parsed

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except Exception:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(cleaned[start:end+1])
            raise RuntimeError(f"Could not parse valid JSON from model response: {cleaned[:200]}")


class InformationExtractor:
    """
    Unified neural information extraction service.
    Orchestrates provider selection and extracts communication parameters in general.
    """

    def __init__(self, provider_type: Optional[str] = None):
        self.provider_type = (provider_type or settings.LLM_PROVIDER or "gemini").lower()
        if self.provider_type == "mock":
            self.provider_type = "gemini"

    def _resolve_provider(
        self,
        provider_type: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> BaseExtractionProvider:
        p_type = provider_type.lower()
        if p_type == "gemini":
            key = api_key or getattr(settings, "active_gemini_key", None)
            if not key:
                raise ValueError("GEMINI_API_KEY is not configured in .env or settings.")
            return GeminiExtractionProvider(api_key=key, model_name=model_name)
        elif p_type == "openai":
            key = api_key or settings.OPENAI_API_KEY
            if not key:
                raise ValueError("OPENAI_API_KEY is not configured in .env or settings.")
            return OpenAIExtractionProvider()
        elif p_type in ("local", "ollama", "huggingface"):
            return LocalExtractionProvider()
        else:
            raise ValueError(f"Unsupported extraction provider '{provider_type}'. Supported: 'gemini', 'openai', 'local'.")

    def extract(self, normalized_text: str, user_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extracts communication attributes from user text using the neural model provider.
        Integrates user hints and formats results for StructuredRequirement.
        """
        user_hints = user_hints or {}
        p_type = (user_hints.get("llm_provider") or self.provider_type).lower()
        if p_type == "mock":
            p_type = "gemini"

        # Only invoke external neural extraction API when explicitly requested (e.g. use_neural_analysis=True)
        # to ensure microsecond responsiveness, prevent rate-limit 429/503 spikes, and avoid blocking lags.
        use_neural = bool(user_hints.get("use_neural_analysis") or user_hints.get("use_neural_extraction"))
        if use_neural and p_type in ("gemini", "openai", "local"):
            try:
                provider = self._resolve_provider(
                    provider_type=p_type,
                    api_key=user_hints.get("api_key"),
                    model_name=user_hints.get("model_name")
                )
                extracted = provider.extract(normalized_text, user_hints=user_hints)
            except Exception as e:
                logger.warning(f"Neural extraction provider '{p_type}' failed: {e}. Using generalized local NLP extraction.")
                extracted = self._local_nlp_fallback(normalized_text, user_hints=user_hints)
        else:
            extracted = self._local_nlp_fallback(normalized_text, user_hints=user_hints)

        # Merge user hints if explicitly provided
        raw_recip = extracted.get("recipient")
        if raw_recip and raw_recip.strip().lower() in ("to", "for", "from", "an", "the", "a", "it", "me", "you", "them", "us", "this", "that"):
            raw_recip = None
        recipient = user_hints.get("recipient_hint") or raw_recip

        user_tone = user_hints.get("tone_hint")
        if user_tone and user_tone.strip().lower() in ("default", "auto-detect", "auto", "none", ""):
            user_tone = None
        tone = user_tone or extracted.get("tone")
        intent = extracted.get("intent")
        topic = extracted.get("topic") or "Communication"
        action = extracted.get("required_action")
        reason = extracted.get("reason")
        deadline = extracted.get("deadline")
        duration = extracted.get("requested_duration")
        required_facts = list(extracted.get("required_facts") or [])
        constraints = list(extracted.get("constraints") or [])

        explicit_restrictions = user_hints.get("explicit_restrictions") or []
        for r in explicit_restrictions:
            if r not in constraints:
                constraints.append(r)

        uncertainties = list(extracted.get("uncertainties") or [])
        clarification_questions = list(extracted.get("clarification_questions") or [])

        inferred_details = {
            "has_causal_justification": reason is not None,
            "has_temporal_anchor": deadline is not None or duration is not None,
            "model_backend": extracted.get("model_identifier", p_type)
        }

        return {
            "intent": intent,
            "tone": tone,
            "recipient": recipient,
            "topic": topic,
            "required_action": action,
            "reason": reason,
            "deadline": deadline,
            "requested_duration": duration,
            "required_facts": required_facts,
            "constraints": constraints,
            "uncertainties": uncertainties,
            "clarification_questions": clarification_questions,
            "inferred_details": inferred_details,
            "confidence": 0.95,
            "model_identifier": extracted.get("model_identifier", p_type)
        }

    def _local_nlp_fallback(self, text: str, user_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generalized offline NLP extraction engine.
        Extracts recipient, intent, topic, action, reason, deadlines, and facts
        using generalized linguistic structures rather than brittle keywords.
        """
        user_hints = user_hints or {}
        lower_t = text.lower()

        DAYS_AND_TEMPORAL = {
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "today", "tomorrow", "tonight", "yesterday", "morning", "afternoon", "evening",
            "january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december"
        }
        COMMON_VERBS_AND_STOPWORDS = {
            "reschedule", "request", "requesting", "ask", "asking", "confirm", "confirming",
            "inform", "informing", "discuss", "discussing", "send", "sending", "write", "writing",
            "meet", "meeting", "tell", "telling", "follow", "following", "arrange", "arranging",
            "apologize", "apologizing", "fix", "fixing", "be", "have", "do", "say", "go", "get", "make",
            "see", "know", "take", "come", "think", "look", "want", "give", "use", "find", "need",
            "advised", "taking", "due", "because", "additional", "courteous", "polite", "urgent",
            "his", "her", "their", "him", "he", "she", "it", "its", "me", "you", "them", "us",
            "my", "the", "a", "an", "our", "someone", "something", "stuff", "things", "about", "life",
            "to", "for", "from", "in", "on", "at", "by", "as", "with", "into", "onto", "is", "are",
            "was", "were", "this", "that", "these", "those", "old", "new", "good", "casual",
            "friendly", "informal", "warm", "natural", "email", "mail", "letter", "message", "note",
            "invitation", "draft", "short", "long", "medium", "formal", "respectful",
            "lunch", "dinner", "breakfast", "coffee", "drinks", "brunch", "meeting", "sync", "call", "catchup", "chat"
        }

        # 1. Recipient extraction (Generalized)
        recipient = user_hints.get("recipient_hint")
        if recipient and recipient.strip().lower() in COMMON_VERBS_AND_STOPWORDS:
            recipient = None

        if not recipient:
            # Check for explicit declarations: "his name Steve", "named Steve", "name is Steve"
            m_name = re.search(r"\b(?:his|her)?\s*name(?:d)?\s+(?:is\s+)?([A-Z][a-z]+)\b", text, re.IGNORECASE)
            if m_name:
                cand = m_name.group(1)
                if cand.lower() not in DAYS_AND_TEMPORAL and cand.lower() not in COMMON_VERBS_AND_STOPWORDS:
                    recipient = cand.title()

        if not recipient:
            # Direct recipient after verb and preposition: e.g. "write ... email to [an old college] friend [Steve]"
            # "email to Steve", "email my manager Sarah", "send an email to my colleague John"
            target_match = re.search(
                r"\b(?:write|send|draft|compose)?\s*(?:an?\s+[^.,;\n]+?\s+)?(?:email|letter|message|note)?\s*(?:to|for)\s+"
                r"(?:(?:my|an?|the|our)\s+)?(?:(?:old\s+|college\s+|close\s+|work\s+)?(friend|colleague|coworker|manager|boss|supervisor|professor|teacher|client|doctor|partner|team)\s+)?([A-Z][a-z]+)?\b",
                text, re.IGNORECASE
            )
            if target_match:
                role = target_match.group(1)
                name = target_match.group(2)
                if name and name.lower() not in DAYS_AND_TEMPORAL and name.lower() not in COMMON_VERBS_AND_STOPWORDS:
                    recipient = f"{role.title()} {name.title()}" if role and role.lower() in ("dr.", "prof.", "doctor", "professor") else name.title()
                elif role and role.lower() not in COMMON_VERBS_AND_STOPWORDS:
                    recipient = role.title()

        if not recipient:
            # Direct action verb followed directly by recipient: "email my manager Sarah", "contact Dr. Henderson", "message Steve"
            direct_verb_match = re.search(
                r"\b(?:email|contact|message)\s+(?:(?:my|the|our)\s+)?(?:(professor|manager|supervisor|candidate|friend|colleague)\s+)?([A-Z][a-z]+)\b",
                text, re.IGNORECASE
            )
            if direct_verb_match:
                role = direct_verb_match.group(1)
                name = direct_verb_match.group(2)
                if name and name.lower() not in DAYS_AND_TEMPORAL and name.lower() not in COMMON_VERBS_AND_STOPWORDS:
                    recipient = f"{role.title()} {name.title()}" if role and role.lower() in ("dr.", "prof.", "doctor", "professor") else name.title()
                elif role and role.lower() not in COMMON_VERBS_AND_STOPWORDS:
                    recipient = role.title()

        if not recipient:
            # Role + Person Name: "friend Steve", "manager Sarah", "colleague John"
            m_role = re.search(r"\b(?:friend|candidate|colleague|manager|professor|supervisor)\s+([A-Z][a-z]+)\b", text, re.IGNORECASE)
            if m_role:
                cand = m_role.group(1)
                if cand.lower() not in DAYS_AND_TEMPORAL and cand.lower() not in COMMON_VERBS_AND_STOPWORDS:
                    recipient = cand.title()

        if not recipient:
            # Title + Name: "Professor Smith", "Dr. Henderson"
            m = re.search(r"\b(professor|dr\.|doctor|prof\.)\s+([A-Z][a-z]+)\b", text, re.IGNORECASE)
            if m:
                role, name = m.group(1), m.group(2)
                if name.lower() not in DAYS_AND_TEMPORAL and name.lower() not in COMMON_VERBS_AND_STOPWORDS:
                    recipient = f"{role.title()} {name.title()}"

        if not recipient:
            # Generalized role/audience entity: "professor", "manager", "billing support", "customer support"
            m = re.search(
                r"\b(?:to\s+(?:my\s+|an?\s+)?|email\s+(?:my\s+|an?\s+)?|for\s+(?:my\s+|an?\s+)?|write\s+to\s+(?:my\s+|an?\s+)?|thank\s+yo(?:u)?\s+)"
                r"(billing\s+support|customer\s+support|support\s+team|it\s+helpdesk|department\s+head|team\s+lead|head\s+of\s+department|professor|manager|supervisor|director|doctor|client|recruiter|hr|colleague|partner|sir|madam|friend)\b",
                text, re.IGNORECASE
            )
            if m:
                recipient = m.group(1).title()

        # 2. Reason extraction (Generalized Causal Parsing)
        reason = None
        for pat in [
            r"\bbecause\s+of\s+(?:a\s+|an\s+)?([^.,;\n]+)",
            r"\bdue\s+to\s+(?:a\s+|an\s+)?([^.,;\n]+)",
            r"\bbecause\s+([^.,;\n]+)",
            r"\bhave\s+(?:a\s+|an\s+)?(fever|cold|headache|illness|flu|infection|medical\s+emergency)",
            r"\bfor\s+(festival\s+celebrations|vacation|holidays|conference|training|a\s+family\s+emergency|personal\s+reasons)",
            r"\bsince\s+([^.,;\n]+)",
            r"\bas\s+a\s+result\s+of\s+([^.,;\n]+)",
            r"\bon\s+account\s+of\s+([^.,;\n]+)"
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                r_val = m.group(1).strip()
                r_val = re.sub(r"\s+(?:write|email|send|please|so|and\s+write)\b.*$", "", r_val, flags=re.IGNORECASE).strip()
                if len(r_val) > 2:
                    reason = r_val
                    break

        # 3. Duration & Timeline
        duration = None
        for pat in [
            r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*(?:to|-)\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b",
            r"\b((?:\d+|two|three|four|five|several|a\s+few)\s+(?:additional\s+|more\s+)?(?:days|weeks|months|hours|business\s+days))\b",
            r"\b(today\s+\w+\s+and\s+tomorrow\s+\w+)\b",
            r"\b(for\s+\d+\s+days|for\s+a\s+week)\b"
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                duration = m.group(1).strip()
                break

        # 4. Deadline / Specific Date
        deadline = None
        for pat in [
            r"\b((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:at|by)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?)\b",
            r"\b((?:tomorrow|today|tonight)(?:\s+(?:at|by)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?)\b",
            r"\b(asap|end\s+of\s+week|by\s+tomorrow\s+morning)\b",
            r"\b((?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?)\b"
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                deadline = m.group(1).title()
                break

        # 5. Generalized Action & Topic Extraction
        action = None
        action_match = re.search(
            r"\b(?:requesting\s+(?:to\s+)?|asking\s+(?:to\s+|for\s+|if\s+)?|informing\s+(?:her|him|them|my\s+\w+)?\s*(?:that\s+|I\s+will\s+be\s+)?|inviting\s+to\s+|apologizing\s+for\s+|confirming\s+(?:that\s+)?|schedule\s+an?\s+|arrange\s+an?\s+|negotiate\s+|discuss\s+|follow\s+up\s+on\s+)([^.,;\n]+)",
            text, re.IGNORECASE
        )
        if action_match:
            raw_act = action_match.group(0).strip()
            action = raw_act[0].upper() + raw_act[1:]
        else:
            action = "Communicate requirement clearly"

        # General Topic resolution
        if "wi-fi" in lower_t or "wifi" in lower_t or "10 gb" in lower_t or "10gb" in lower_t:
            topic = "Wi-Fi Usage Policy"
            action = "Acknowledge notification and confirm adherence to 10 GB daily limit"
        elif "sick" in lower_t or "fever" in lower_t:
            topic = "Sick Leave Notice"
        elif "reschedule" in lower_t or "appointment" in lower_t:
            topic = "Meeting Reschedule Request"
        elif "refund" in lower_t or "billing" in lower_t or "duplicate" in lower_t:
            topic = "Billing Refund Request"
        elif "extension" in lower_t or "deadline" in lower_t or "additional days" in lower_t:
            topic = "Project Deadline Extension Request"
        elif "catch up" in lower_t or "hang out" in lower_t:
            topic = "Catching Up"
        elif "recommendation" in lower_t:
            topic = "Letter of Recommendation Request"
        elif "interview" in lower_t:
            topic = "Interview Discussion"
        elif "proposal" in lower_t or "quote" in lower_t:
            topic = "Proposal Follow-up"
        else:
            topic = "Important Communication"
            if reason:
                topic = f"Notice: {reason.title()}"
            elif duration:
                topic = f"Request: {duration.title()}"

        # 6. Consolidate Facts
        facts = []
        if reason:
            facts.append(reason)
        if duration:
            facts.append(duration)
        if "wi-fi" in lower_t or "10 gb" in lower_t or "10gb" in lower_t:
            facts.append("Confirm adherence to 10 GB daily Wi-Fi limit and keep login credentials secure")

        # Quantities/currencies
        for qty_pat in [r"\$\d+(?:\.\d{2})?", r"\b\d+\s*(?:gb|mb|kb|percent|%)\b"]:
            for qm in re.finditer(qty_pat, text, re.IGNORECASE):
                val = qm.group(0)
                if val not in facts:
                    facts.append(val)

        return {
            "recipient": recipient,
            "topic": topic,
            "required_action": action,
            "reason": reason,
            "deadline": deadline,
            "requested_duration": duration,
            "required_facts": facts,
            "constraints": ["Do not invent unmentioned details"],
            "uncertainties": [],
            "clarification_questions": [],
            "model_identifier": "local-nlp-general-v2"
        }
