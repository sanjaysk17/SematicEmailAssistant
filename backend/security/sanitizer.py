"""
backend/security/sanitizer.py - Security, Privacy, and Input Sanitization
Handles prompt-injection sanitization, PII masking, and data persistence controls.
"""

import re
from typing import Dict, Any, List
from backend.config import settings


class SecuritySanitizer:
    """
    Sanitizes user input to prevent prompt injection and protects privacy
    by masking sensitive credentials and respecting persistence settings.
    """

    # Patterns attempting to override LLM system prompts
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+a\s+different\s+ai", re.IGNORECASE),
        re.compile(r"system\s*:\s*override", re.IGNORECASE),
        re.compile(r"format\s*:\s*drop\s+tables", re.IGNORECASE),
        re.compile(r"reveal\s+(api\s+key|system\s+prompt|password)", re.IGNORECASE)
    ]

    CREDENTIAL_PATTERNS = [
        re.compile(r"(?:api[_-]?key|secret|token|password|auth)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{16,})['\"]?", re.IGNORECASE),
        re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE),
    ]

    @classmethod
    def sanitize_input(cls, text: str) -> str:
        """
        Escapes and cleans user input to prevent jailbreak / prompt injection attacks
        before passing into LLM prompts.
        """
        if not text:
            return ""

        cleaned = text.strip()

        # Defang injection directives by wrapping with data qualifiers
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(cleaned):
                cleaned = pattern.sub("[FILTERED_COMMAND]", cleaned)

        return cleaned

    @classmethod
    def mask_credentials(cls, text: str) -> str:
        """Mask potential tokens or API keys from logs and storage."""
        if not text:
            return ""

        masked = text
        for pattern in cls.CREDENTIAL_PATTERNS:
            masked = pattern.sub("[REDACTED_SECRET]", masked)
        return masked

    @classmethod
    def filter_content_for_storage(cls, content: str) -> str:
        """
        Respects DISABLE_EMAIL_CONTENT_PERSISTENCE setting.
        If persistence is disabled, returns an unrecoverable marker.
        """
        if settings.DISABLE_EMAIL_CONTENT_PERSISTENCE:
            return "[PERSISTENCE_DISABLED_FOR_PRIVACY]"
        return content
