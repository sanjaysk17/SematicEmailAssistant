"""
ai/semantic_interpretation.py - General-Purpose Semantic Understanding Pipeline
Orchestrates preprocessing, information extraction, intent interpretation,
and tone analysis into a validated CommunicationRepresentation.
"""

import time
import json
import re
import logging
from typing import Dict, Any, Optional, List
import httpx

from ai.preprocessing import TextPreprocessor
from ai.representation import CommunicationRepresentation
from ai.information_extraction import InformationExtractor
from ai.intent_classification import IntentClassifier
from ai.tone_analysis import ToneAnalyzer
from backend.config import settings

logger = logging.getLogger(__name__)


class SemanticInterpreter:
    """
    Core understanding engine that transforms raw natural language requirements
    into structured communication representations without hardcoded rules.
    Supports both Google Gemini API neural semantic analysis and local NLP pipeline.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.extractor = InformationExtractor()
        self.intent_classifier = IntentClassifier()
        self.tone_analyzer = ToneAnalyzer()

    def interpret(
        self,
        raw_requirement: str,
        context: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes full semantic interpretation pipeline.
        Prioritizes Gemini LLM for tone, intent, and factual extraction when key is active.
        """
        user_preferences = user_preferences or {}

        # 0. Check if Gemini AI semantic analysis is available
        gemini_key = user_preferences.get("api_key") or getattr(settings, "active_gemini_key", None)
        if gemini_key and (settings.LLM_PROVIDER == "gemini" or user_preferences.get("api_key")):
            gemini_res = self._interpret_with_gemini(
                raw_requirement=raw_requirement,
                context=context,
                api_key=gemini_key,
                user_preferences=user_preferences
            )
            if gemini_res:
                return gemini_res

        start_time = time.time()

        # 1. Preprocessing and normalization
        prep_result = self.preprocessor.process(raw_requirement)
        normalized_text = prep_result["normalized_text"]

        # Append optional context to normalized text for extraction if available
        combined_text = normalized_text
        if context and context.strip():
            combined_text += f"\nContext: {context.strip()}"

        # 2. Extract key information
        extracted = self.extractor.extract(combined_text, user_hints=user_preferences)

        # 3. Classify communication intent
        intent_result = self.intent_classifier.classify(normalized_text, user_hints=user_preferences)

        # 4. Analyze requested tone
        requested_tone = self.tone_analyzer.extract_requested_tone(
            raw_requirement,
            user_hint=user_preferences.get("tone_hint")
        )

        # 5. Compile structured representation
        recipient = extracted["recipient"] or user_preferences.get("recipient_hint")
        action = extracted["required_action"]
        reason = extracted["reason"]
        deadline = extracted["deadline"]
        duration = extracted["requested_duration"]
        required_facts = extracted["required_facts"]
        constraints = extracted["constraints"]
        uncertainties = extracted["uncertainties"]
        clarification_questions = extracted["clarification_questions"]

        # If user explicitly specified constraints, merge them
        explicit_restrictions = user_preferences.get("explicit_restrictions", [])
        for r in explicit_restrictions:
            if r not in constraints:
                constraints.append(r)

        # Additional domain attributes
        custom_attrs = {
            "has_negation": prep_result["has_negation"],
            "detected_quantities": prep_result["quantities_found"],
            "is_open_ended_intent": intent_result["is_open_ended"]
        }

        representation = CommunicationRepresentation(
            intent=intent_result["intent"],
            tone=requested_tone,
            recipient=recipient,
            topic=extracted["topic"],
            required_action=action,
            reason=reason,
            deadline=deadline,
            requested_duration=duration,
            required_facts=required_facts,
            constraints=constraints,
            uncertainties=uncertainties,
            inferred_details=extracted["inferred_details"],
            clarification_questions=clarification_questions,
            user_confirmed=False,
            custom_attributes=custom_attrs
        )

        warnings = []
        if uncertainties:
            warnings.extend(uncertainties)
        if prep_result["has_negation"]:
            warnings.append("Negation detected in requirement; preserved for prompt safety.")

        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "representation": representation,
            "confidence": intent_result["confidence"],
            "model_identifier": "semantic-pipeline-sbert-nlp-v1",
            "warnings": warnings,
            "elapsed_ms": round(elapsed_ms, 2)
        }

    def _interpret_with_gemini(
        self,
        raw_requirement: str,
        context: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Uses Google Gemini 1.5 Flash to perform neural semantic and tone analysis.
        Extracts intent, tone, recipient, required action, facts, and constraints with high contextual awareness.
        """
        start_time = time.time()
        user_preferences = user_preferences or {}
        model = model_name or settings.GEMINI_MODEL_NAME or "gemini-1.5-flash"
        api_base = getattr(settings, "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        url = f"{api_base}/models/{model}:generateContent?key={api_key}"

        prompt = (
            "You are an expert NLP communication analyst for an email assistant.\n"
            "Analyze the user's email drafting instruction and extract the structured communication intent, "
            "tone, recipient, core topic, required action, reason, deadline, facts, and constraints.\n\n"
            f"[USER INSTRUCTION / PROMPT]:\n{raw_requirement.strip()}\n\n"
        )
        if context and context.strip():
            prompt += f"[ADDITIONAL CONTEXT]:\n{context.strip()}\n\n"

        prompt += (
            "Return a strictly valid JSON object with the following fields (no markdown fences, just pure JSON):\n"
            "{\n"
            '  "intent": "The primary intent (e.g. reply_and_acknowledgement, request_extension, informal_invitation, reschedule_meeting, request_refund, general_inquiry)",\n'
            '  "tone": "The requested or appropriate tone (e.g. polite and professional, formal and respectful, friendly and casual, appreciative)",\n'
            '  "recipient": "The recipient name, title, or honorific (e.g. Sir, NMC Administration, Professor Marcus, Steve). Null if not specified",\n'
            '  "topic": "The core subject matter or topic of the email",\n'
            '  "required_action": "Specific action requested from the recipient, or confirmed by the sender",\n'
            '  "reason": "The reason or justification provided, or null",\n'
            '  "deadline": "Any deadline, day, or time mentioned, or null",\n'
            '  "requested_duration": "Any duration mentioned, or null",\n'
            '  "required_facts": ["List of explicit facts to preserve"],\n'
            '  "constraints": ["List of negative constraints or instructions"]\n'
            "}"
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 800,
                "responseMimeType": "application/json"
            }
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 400 and "responseMimeType" in resp.text:
                    payload["generationConfig"].pop("responseMimeType", None)
                    resp = client.post(url, json=payload)

                if resp.status_code != 200:
                    logger.warning(f"Gemini semantic interpretation HTTP error ({resp.status_code}): {resp.text}")
                    return None

                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return None
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    return None
                raw_text = parts[0].get("text", "").strip()

                # Clean markdown fences if any
                fence_m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
                if fence_m:
                    raw_text = fence_m.group(1).strip()

                parsed = json.loads(raw_text)

                recipient = user_preferences.get("recipient_hint") or parsed.get("recipient")
                tone = user_preferences.get("tone_hint") or parsed.get("tone") or "polite and professional"
                intent = parsed.get("intent") or "open_ended_communication"
                topic = parsed.get("topic") or "General Communication"
                action = parsed.get("required_action")
                reason = parsed.get("reason")
                deadline = parsed.get("deadline")
                duration = parsed.get("requested_duration")
                facts = parsed.get("required_facts") or []
                constraints = parsed.get("constraints") or []

                for r in user_preferences.get("explicit_restrictions", []):
                    if r not in constraints:
                        constraints.append(r)

                rep = CommunicationRepresentation(
                    intent=intent,
                    tone=tone,
                    recipient=recipient,
                    topic=topic,
                    required_action=action,
                    reason=reason,
                    deadline=deadline,
                    requested_duration=duration,
                    required_facts=facts,
                    constraints=constraints,
                    uncertainties=[],
                    inferred_details={"gemini_analyzed": True, "model": model},
                    clarification_questions=[],
                    user_confirmed=False,
                    custom_attributes={"gemini_driven": True}
                )

                elapsed_ms = (time.time() - start_time) * 1000.0
                return {
                    "representation": rep,
                    "confidence": 0.98,
                    "model_identifier": f"gemini-semantic-pipeline ({model})",
                    "warnings": [],
                    "elapsed_ms": round(elapsed_ms, 2)
                }
        except Exception as e:
            logger.warning(f"Gemini semantic interpretation failed: {e}. Falling back to local NLP pipeline.")
            return None
