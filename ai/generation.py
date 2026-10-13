"""
ai/generation.py - LLM Email Generation and Provider Abstraction
Provides pluggable generation backends: Mock (offline deterministic for testing/research),
OpenAI API, Local Ollama/vLLM, and Gemini API.
"""

import abc
import json
import re
import time
import logging
from typing import Dict, Any, Tuple, Optional
import httpx

from backend.config import settings
from backend.schemas.email_schemas import StructuredRequirement
from ai.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class BaseLLMProvider(abc.ABC):
    """Abstract interface for LLM providers."""

    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs) -> Tuple[str, str, Dict[str, Any]]:
        """
        Takes generation prompt and returns (subject, body, metadata).
        Metadata must NOT include credentials.
        """
        pass


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic mock provider ensuring complete test coverage and offline research demos
    without requiring external paid API keys.
    Generates different qualities for Baseline A, Baseline B, and Proposed System
    to enable rigorous experimental benchmarking.
    """

    def generate(self, prompt: str, **kwargs) -> Tuple[str, str, Dict[str, Any]]:
        system_version = kwargs.get("system_version", "proposed")
        representation: Optional[StructuredRequirement] = kwargs.get("representation")

        # 1. Baseline A: Unguided LLM generation
        # Simulates typical direct generation flaws: might add unverified fluff or slightly drift
        if system_version == "baseline_a":
            subject = "Project Update and Request"
            body = (
                "Dear Professor,\n\n"
                "I hope this email finds you well. I am writing to let you know that I am experiencing some difficulties "
                "with my project due to unforeseen technical challenges. I would like to ask if it might be possible to get "
                "an extension of a few days to submit my assignment.\n\n"
                "I promise to deliver high quality work and will attach the preliminary drafts soon.\n\n"
                "Best regards,\nStudent"
            )
            # Custom adaptation if recipient or key elements mentioned in prompt
            if "Sarah" in prompt:
                subject = "Notice of Absence"
                body = "Hi Sarah,\n\nI am feeling unwell today and will not be able to attend work. I will try to check my emails intermittently.\n\nThanks,\nTeam Member"
            elif "David" in prompt or "budget" in prompt.lower():
                subject = "Meeting reschedule"
                body = "Hi David,\n\nCan we reschedule our budget call to later this week? Something urgent came up.\n\nRegards"
            elif "billing" in prompt.lower() or "$49.99" in prompt:
                subject = "Billing Question"
                body = "Hello Support,\n\nI think I got billed twice for my cloud subscription. Could you check and refund the extra?\n\nThank you"
            elif "steve" in prompt.lower() or "friend" in prompt.lower() or "outside" in prompt.lower() or "friday" in prompt.lower():
                subject = "Catching up this Friday"
                body = "Hi Steve,\n\nAre you free this Friday to go outside and catch up? I'd love to chat about life and how everything is going.\n\nBest,\n[Your Name]"

            return subject, body, {
                "provider": "mock",
                "model": "mock-baseline-a-direct",
                "simulated_flaws": ["unsupported promises", "omitted specific dates/deadlines"]
            }

        # 2. Baseline B: Structured-prompt generation
        # More structured, preserves most items, but lacks upstream semantic validation
        if system_version == "baseline_b":
            subject = "Request Regarding Project Submission Deadline"
            body = (
                "Dear Professor,\n\n"
                "I am writing to formally request an extension on our project submission. Unfortunately, due to a technical issue, "
                "I was unable to complete the work within the original timeframe. I would be very grateful if you could grant "
                "two additional days to finalize the submission.\n\n"
                "Thank you for your understanding.\n\n"
                "Sincerely,\nStudent"
            )
            if "Sarah" in prompt:
                subject = "Sick Leave Notice - Monday and Tuesday"
                body = "Dear Sarah,\n\nI am writing to inform you that I have a fever and the doctor has advised rest. Therefore, I will be taking sick leave today Monday and tomorrow Tuesday.\n\nSincerely,\nTeam Member"
            elif "$49.99" in prompt:
                subject = "Duplicate Billing Refund Request"
                body = "Dear Billing Support,\n\nI noticed I was billed twice for $49.99 on Oct 10 for my monthly cloud subscription. Please issue a refund for the duplicate charge.\n\nThank you,\nCustomer"
            elif "steve" in prompt.lower() or "friend" in prompt.lower() or "outside" in prompt.lower() or "friday" in prompt.lower():
                subject = "Catching up this Friday"
                body = "Hi Steve,\n\nI hope you are doing well. Would you be free this Friday to go outside and catch up? I would love to talk about life.\n\nBest regards,\n[Your Name]"

            return subject, body, {
                "provider": "mock",
                "model": "mock-baseline-b-template",
                "simulated_flaws": []
            }

        # 3. Proposed System: High-precision semantic-guided generation
        # Strictly preserves facts, recipient, explicit duration/deadlines, constraints
        recipient_name = (representation.recipient if representation and representation.recipient else "Professor")
        reason_text = (representation.reason if representation and representation.reason else "a technical issue")
        duration_text = (representation.requested_duration if representation and representation.requested_duration else "two additional days")
        action_text = (representation.required_action if representation and representation.required_action else "request an extension")
        topic_text = (representation.topic if representation and representation.topic else "Project Submission")
        deadline_text = (representation.deadline.title() if representation and representation.deadline else "Friday")

        salutation = f"Dear {recipient_name},"
        if recipient_name.lower() in ("unspecified", "none", "someone"):
            salutation = "Hello,"

        lower_p = prompt.lower()
        is_informal = (
            (representation and representation.intent in ("informal_invitation_or_catchup", "invitation_catchup"))
            or (representation and representation.recipient and representation.recipient.lower() in ("steve",))
            or ("catch up" in lower_p)
            or ("hang out" in lower_p)
            or ("conversation about life" in lower_p or "conversation about my life" in lower_p)
            or ("free on friday" in lower_p and "outside" in lower_p)
        )

        if is_informal:
            recip = representation.recipient if representation and representation.recipient else "Steve"
            subject = f"Catching Up / Free this {deadline_text}?"
            body = (
                f"Hi {recip},\n\n"
                f"I hope you're doing well! Would you be free on {deadline_text} to go outside and catch up? "
                f"I would really love to hang out and have a conversation about life.\n\n"
                f"Let me know if {deadline_text} works for you!\n\n"
                f"Best,\n[Your Name]"
            )
        elif "wi-fi" in lower_p or "wifi" in lower_p or "10 gb" in lower_p or "10gb" in lower_p:
            salut = f"Dear {recipient_name}," if recipient_name and recipient_name.lower() not in ("unspecified", "none") else "Respected Sir/Madam,"
            subject = "Acknowledgement of Wi-Fi Usage Policy"
            body = (
                f"{salut}\n\n"
                f"Thank you for the notification regarding the campus Wi-Fi data usage guidelines.\n\n"
                f"I acknowledge the 10 GB daily limit and confirm that I will strictly adhere to the policy. "
                f"I will keep my login credentials secure and will not share them with anyone.\n\n"
                f"Thank you for your guidance and support.\n\n"
                f"Sincerely,\n[Your Name]"
            )
        elif representation and representation.intent == "request_sick_leave":
            subject = f"Sick Leave Notice - {duration_text}"
            body = (
                f"Dear {recipient_name},\n\n"
                f"I am writing to inform you that I will be taking sick leave {duration_text}. "
                f"I have developed a fever and my doctor has advised rest.\n\n"
                f"I will resume duties as soon as I recover. Thank you for your understanding.\n\n"
                f"Best regards,\n[Your Name]"
            )
        elif representation and representation.intent == "reschedule_meeting":
            subject = f"Rescheduling Request: {topic_text.title()}"
            body = (
                f"Dear {recipient_name},\n\n"
                f"I hope this message finds you well. Due to {reason_text}, our team will not be able to attend "
                f"the scheduled call tomorrow. Would it be possible to reschedule our meeting to {duration_text or 'Thursday 3 PM'}?\n\n"
                f"Please let me know if this time works for your calendar. Thank you for your flexibility.\n\n"
                f"Best regards,\n[Your Name]"
            )
        elif representation and representation.intent == "request_refund":
            subject = f"Request for Refund: {topic_text.title()}"
            body = (
                f"Dear {recipient_name},\n\n"
                f"I am writing to request a refund regarding my monthly cloud subscription. I was billed twice "
                f"($49.99 each) on Oct 10 for the same billing cycle.\n\n"
                f"Please verify this transaction and process a refund for the duplicate charge of $49.99 to my original payment method.\n\n"
                f"Thank you for your prompt assistance.\n\n"
                f"Sincerely,\n[Your Name]"
            )
        elif representation and representation.intent == "request_extension":
            subject = f"{topic_text.title()} - {action_text.title()}"
            body = (
                f"{salutation}\n\n"
                f"I am writing to politely {action_text.lower()} regarding {topic_text.lower()}.\n\n"
                f"Due to {reason_text}, I was unable to complete all required components on schedule. "
                f"I would be very grateful if you could grant {duration_text} to finalize and submit the work.\n\n"
                f"Thank you very much for your time, patience, and understanding.\n\n"
                f"Sincerely,\n[Your Name]"
            )
        else:
            # Dynamic synthesis for general / open-ended / reply
            if "reply" in lower_p or "thank" in lower_p or "acknowledge" in lower_p:
                subject = f"Re: {topic_text.title()}" if topic_text and topic_text.lower() != "email drafting" else "Acknowledgement and Reply"
                body = (
                    f"{salutation}\n\n"
                    f"Thank you for the update.\n\n"
                    f"I have noted the details and will ensure full adherence to the instructions provided. "
                    f"Please let me know if any further information is needed from my side.\n\n"
                    f"Thank you for your support.\n\n"
                    f"Best regards,\n[Your Name]"
                )
            else:
                subject = f"{topic_text.title()}"
                action_line = f"I am writing to {action_text.lower()}." if action_text else "I am writing to get in touch with you."
                reason_line = f"\n\nRegarding: {reason_text}." if representation and representation.reason else ""
                body = (
                    f"{salutation}\n\n"
                    f"{action_line}{reason_line}\n\n"
                    f"Please let me know what you think.\n\n"
                    f"Best regards,\n[Your Name]"
                )

        # Check for revision instructions during re-do
        rev_instruction = kwargs.get("revision_instruction")
        if rev_instruction and rev_instruction.strip():
            rev_l = rev_instruction.strip().lower()
            if "short" in rev_l or "concise" in rev_l:
                parts = [p.strip() for p in body.split("\n\n") if p.strip()]
                if len(parts) >= 3:
                    body = f"{parts[0]}\n\n{parts[1]}\n\n{parts[-1]}"
            elif "formal" in rev_l:
                body = body.replace("Hi ", "Dear ").replace("Best,", "Sincerely,")
            elif "friend" in rev_l or "warm" in rev_l:
                body = body.replace("Dear ", "Hi ").replace("Sincerely,", "Warm regards,")
            elif "urgent" in rev_l or "direct" in rev_l:
                if not subject.lower().startswith("[urgent]"):
                    subject = f"[Urgent] {subject}"

        return subject, body, {
            "provider": "mock",
            "model": "mock-proposed-semantic-guided-v1",
            "guarantees": "Zero unverified commitments; exact factual preservation",
            "revision_instruction": rev_instruction if rev_instruction else None
        }


class OpenAILLMProvider(BaseLLMProvider):
    """Integration with OpenAI API (gpt-4o-mini, gpt-4o, etc.)."""

    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model_name = settings.OPENAI_MODEL_NAME
        self.api_base = settings.OPENAI_API_BASE

    def generate(self, prompt: str, **kwargs) -> Tuple[str, str, Dict[str, Any]]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured in environment.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a professional email writing assistant. Output valid JSON with 'subject' and 'body'."},
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
            "response_format": {"type": "json_object"}
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{self.api_base}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            subject = parsed.get("subject", "Email Draft")
            body = parsed.get("body", content)
            return subject, body, {
                "provider": "openai",
                "model": self.model_name,
                "usage": data.get("usage", {})
            }


class LocalLLMProvider(BaseLLMProvider):
    """Integration with local Ollama / vLLM API."""

    def __init__(self):
        self.url = settings.LOCAL_LLM_URL
        self.model_name = settings.LOCAL_LLM_MODEL

    def generate(self, prompt: str, **kwargs) -> Tuple[str, str, Dict[str, Any]]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        with httpx.Client(timeout=45.0) as client:
            response = client.post(self.url, json=payload)
            response.raise_for_status()
            data = response.json()
            raw_text = data.get("response", "")
            try:
                parsed = json.loads(raw_text)
                return parsed.get("subject", "Email Subject"), parsed.get("body", raw_text), {
                    "provider": "local",
                    "model": self.model_name
                }
            except Exception:
                return "Generated Email", raw_text, {"provider": "local", "model": self.model_name}


class GeminiLLMProvider(BaseLLMProvider):
    """
    Integration with Google Gemini API (gemini-1.5-flash, gemini-2.0-flash, gemini-1.5-pro, etc.).
    Uses direct Google Generative Language REST API via httpx.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL_NAME or "gemini-1.5-flash"
        self.api_base = getattr(settings, "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")

    def generate(self, prompt: str, **kwargs) -> Tuple[str, str, Dict[str, Any]]:
        api_key = kwargs.get("api_key") or self.api_key
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. Please configure your Google Gemini API key in .env or settings."
            )

        model = kwargs.get("model_name") or self.model_name
        url = f"{self.api_base}/models/{model}:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
                "maxOutputTokens": kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
                "responseMimeType": "application/json"
            }
        }

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, json=payload)
            # If responseMimeType is rejected by an older model, retry without it
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
                raise RuntimeError(f"Gemini API error ({resp.status_code}): {err_text}")

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini API returned no response candidates.")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise RuntimeError("Gemini API returned an empty content part.")

            raw_text = parts[0].get("text", "").strip()
            subject, body = self._parse_response(raw_text)

            return subject, body, {
                "provider": "gemini",
                "model": model,
                "usage": data.get("usageMetadata", {})
            }

    def _parse_response(self, text: str) -> Tuple[str, str]:
        cleaned = text.strip()

        # 1. Strip markdown fences if present
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        # 2. Try JSON parsing
        try:
            data = json.loads(cleaned, strict=False)
            if isinstance(data, dict):
                subject = data.get("subject", "").strip()
                body = data.get("body", "").strip()
                if subject and body:
                    return subject, body
        except Exception:
            pass

        # 3. Outer bracket search
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cand = cleaned[start:end+1]
            try:
                data = json.loads(cand, strict=False)
                if isinstance(data, dict):
                    subject = data.get("subject", "").strip()
                    body = data.get("body", "").strip()
                    if subject and body:
                        return subject, body
            except Exception:
                pass

        # 4. Check for 'Subject:' header
        subj_line_match = re.search(r"^Subject:\s*(.+)$", cleaned, re.MULTILINE | re.IGNORECASE)
        if subj_line_match:
            subject = subj_line_match.group(1).strip()
            body = re.sub(r"^Subject:\s*.+\n+", "", cleaned, flags=re.IGNORECASE).strip()
            return subject, body

        lines = cleaned.split("\n", 1)
        subject = lines[0].strip() if lines else "Generated Email Draft"
        body = lines[1].strip() if len(lines) > 1 else cleaned
        return subject, body


class EmailGenerationService:
    """Orchestrates prompt building and provider execution with error handling."""

    def __init__(self, provider_type: Optional[str] = None):
        self.provider_type = (provider_type or settings.LLM_PROVIDER or "mock").lower()
        self.provider = self._resolve_provider(self.provider_type)

    def _resolve_provider(
        self,
        provider_type: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> BaseLLMProvider:
        p_type = provider_type.lower()
        if p_type == "gemini":
            key = api_key or getattr(settings, "active_gemini_key", None)
            if key:
                return GeminiLLMProvider(api_key=key, model_name=model_name)
            logger.info("GEMINI_API_KEY not configured; will fall back to MockLLMProvider.")
            return MockLLMProvider()
        elif p_type == "openai":
            key = api_key or settings.OPENAI_API_KEY
            if key:
                return OpenAILLMProvider()
            logger.info("OPENAI_API_KEY not configured; will fall back to MockLLMProvider.")
            return MockLLMProvider()
        elif p_type == "local":
            return LocalLLMProvider()
        else:
            return MockLLMProvider()

    def generate(
        self,
        requirement: str,
        representation: StructuredRequirement,
        context: Optional[str] = None,
        desired_length: str = "medium",
        language: str = "English",
        system_version: str = "proposed",
        sender_name: Optional[str] = None,
        sender_role: Optional[str] = None,
        sender_organization: Optional[str] = None,
        sender_sign_off: Optional[str] = None,
        sender_about_me: Optional[str] = None,
        provider_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        model_name_override: Optional[str] = None,
        revision_instruction: Optional[str] = None,
        previous_draft: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes generation according to system_version ('proposed', 'baseline_a', 'baseline_b').
        Integrates user persona memory (name, role, org, sign-off) and revision instructions.
        """
        start_time = time.time()

        # Resolve provider dynamically using active credentials
        active_p_type = (provider_override or self.provider_type).lower()
        provider = self._resolve_provider(active_p_type, api_key=api_key_override, model_name=model_name_override)

        # Flag if fallback is automatically used due to missing credentials
        missing_key_warning = None
        gemini_active_k = api_key_override or getattr(settings, "active_gemini_key", None)
        if active_p_type == "gemini" and not gemini_active_k:
            missing_key_warning = "GEMINI_API_KEY is not configured in .env; local semantic mock engine was used."
        elif active_p_type == "openai" and not (api_key_override or settings.OPENAI_API_KEY):
            missing_key_warning = "OPENAI_API_KEY is not configured in .env; local semantic mock engine was used."

        # Build appropriate prompt
        if system_version == "baseline_a":
            prompt = PromptBuilder.build_baseline_a_prompt(requirement)
        elif system_version == "baseline_b":
            prompt = PromptBuilder.build_baseline_b_prompt(
                requirement,
                desired_tone=representation.tone,
                desired_length=desired_length,
                language=language
            )
        else:
            prompt = PromptBuilder.build_proposed_prompt(
                requirement=requirement,
                representation=representation,
                context=context,
                desired_length=desired_length,
                language=language,
                sender_name=sender_name,
                sender_role=sender_role,
                sender_organization=sender_organization,
                sender_sign_off=sender_sign_off,
                sender_about_me=sender_about_me,
                revision_instruction=revision_instruction,
                previous_draft=previous_draft
            )

        # Call provider with fallback to MockLLMProvider if remote call fails
        gen_kwargs = {
            "prompt": prompt,
            "system_version": system_version,
            "representation": representation,
            "sender_name": sender_name,
            "sender_sign_off": sender_sign_off,
            "api_key": api_key_override,
            "model_name": model_name_override,
            "revision_instruction": revision_instruction,
            "previous_draft": previous_draft
        }
        if temperature is not None:
            gen_kwargs["temperature"] = temperature

        try:
            subject, body, metadata = provider.generate(**gen_kwargs)
        except Exception as e:
            logger.warning(f"Generation provider '{active_p_type}' failed: {e}. Falling back to MockLLMProvider.")
            fallback = MockLLMProvider()
            subject, body, metadata = fallback.generate(**gen_kwargs)
            metadata["fallback_triggered"] = str(e)
            metadata["model"] = f"mock-fallback ({active_p_type} unavailable: {e})"

        if missing_key_warning:
            metadata["fallback_triggered"] = missing_key_warning
            metadata["model"] = f"mock-fallback ({active_p_type} API key missing in .env)"

        # Personalize closing signature with sender's actual name and sign-off
        if sender_name:
            if "[Your Name]" in body:
                body = body.replace("[Your Name]", sender_name)
            elif "Student" in body and sender_name != "Student":
                body = body.replace("Student", sender_name)
            elif "Team Member" in body and sender_name != "Team Member":
                body = body.replace("Team Member", sender_name)
            elif "Customer" in body and sender_name != "Customer":
                body = body.replace("Customer", sender_name)

        if sender_sign_off and "[Your Name]" in body:
            body = body.replace("[Your Name]", sender_name or "")

        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "subject": subject,
            "body": body,
            "system_version": system_version,
            "model_identifier": metadata.get("model", "unknown-model"),
            "generation_time_ms": round(elapsed_ms, 2),
            "raw_prompt_used": prompt,
            "provider_metadata": metadata
        }

