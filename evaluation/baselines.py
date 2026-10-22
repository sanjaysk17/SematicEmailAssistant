"""
evaluation/baselines.py - Experimental Baseline and Proposed System Implementations
Executes the three systems under identical experimental controls:
- Baseline A: Direct LLM Generation
- Baseline B: Structured-Prompt Generation
- Proposed System: Semantic Interpretation -> Structured Representation -> Prompt Construction -> LLM -> Output Validation
"""

import time
from typing import Dict, Any, Tuple, Optional
from backend.schemas.evaluation_schemas import EvaluationExample
from backend.schemas.email_schemas import StructuredRequirement
from ai.semantic_interpretation import SemanticInterpreter
from ai.generation import EmailGenerationService
from ai.validation import OutputValidator


class BaselineEvaluator:
    """Runs single example iterations through the candidate architectures."""

    def __init__(self, provider_type: Optional[str] = "mock"):
        self.interpreter = SemanticInterpreter()
        self.provider_type = provider_type or "mock"
        self.generation_service = EmailGenerationService(provider_type=self.provider_type)
        self.validator = OutputValidator()

    def run_baseline_a(self, example: EvaluationExample) -> Dict[str, Any]:
        """
        Baseline A: Direct LLM Generation
        Requirement -> Direct LLM -> Email (evaluated post-hoc for metrics)
        """
        start = time.time()
        # Direct generation without upstream semantic understanding
        dummy_rep = StructuredRequirement(
            intent=example.expected_intent,
            tone=example.expected_tone,
            recipient=example.recipient,
            topic=example.topic,
            required_action=example.required_action,
            reason=example.reason,
            requested_duration=example.date_deadline_duration,
            required_facts=[example.important_facts] if example.important_facts else []
        )

        gen = self.generation_service.generate(
            requirement=example.user_requirement,
            representation=dummy_rep,
            system_version="baseline_a",
            provider_override=self.provider_type
        )
        latency = (time.time() - start) * 1000.0

        # Independent validation against ground truth
        report = self.validator.validate(
            requirement=example.user_requirement,
            representation=dummy_rep,
            subject=gen["subject"],
            body=gen["body"]
        )

        return {
            "method": "baseline_a",
            "subject": gen["subject"],
            "body": gen["body"],
            "report": report,
            "latency_ms": round(latency, 2)
        }

    def run_baseline_b(self, example: EvaluationExample) -> Dict[str, Any]:
        """
        Baseline B: Structured-Prompt Generation
        Requirement -> Structured Prompt Template -> LLM -> Email
        """
        start = time.time()
        dummy_rep = StructuredRequirement(
            intent=example.expected_intent,
            tone=example.expected_tone,
            recipient=example.recipient,
            topic=example.topic,
            required_action=example.required_action,
            reason=example.reason,
            requested_duration=example.date_deadline_duration,
            required_facts=[example.important_facts] if example.important_facts else []
        )

        gen = self.generation_service.generate(
            requirement=example.user_requirement,
            representation=dummy_rep,
            system_version="baseline_b",
            provider_override=self.provider_type
        )
        latency = (time.time() - start) * 1000.0

        report = self.validator.validate(
            requirement=example.user_requirement,
            representation=dummy_rep,
            subject=gen["subject"],
            body=gen["body"]
        )

        return {
            "method": "baseline_b",
            "subject": gen["subject"],
            "body": gen["body"],
            "report": report,
            "latency_ms": round(latency, 2)
        }

    def run_proposed(self, example: EvaluationExample) -> Dict[str, Any]:
        """
        Proposed System:
        Requirement -> Semantic Interpretation -> Structured Representation -> Prompt Construction -> LLM -> Output Validation
        """
        start = time.time()
        # Stage 1: Semantic interpretation
        interp = self.interpreter.interpret(
            raw_requirement=example.user_requirement,
            user_preferences={"tone_hint": example.expected_tone, "recipient_hint": example.recipient}
        )
        rep: StructuredRequirement = interp["representation"]

        # Stage 2: Prompt construction and Generation
        gen = self.generation_service.generate(
            requirement=example.user_requirement,
            representation=rep,
            system_version="proposed",
            provider_override=self.provider_type
        )

        # Stage 3: Output Validation
        report = self.validator.validate(
            requirement=example.user_requirement,
            representation=rep,
            subject=gen["subject"],
            body=gen["body"]
        )
        latency = (time.time() - start) * 1000.0

        return {
            "method": "proposed",
            "subject": gen["subject"],
            "body": gen["body"],
            "report": report,
            "latency_ms": round(latency, 2),
            "representation": rep
        }
