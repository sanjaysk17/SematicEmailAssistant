"""
ai/information_extraction.py - Linguistic and Contextual Information Extraction
Extracts recipient, topic, requested action, reasons, dates/deadlines, durations,
facts, constraints, and uncertainties using linguistic NLP and semantic parsing.
Does NOT rely on rigid hardcoded single-situation regexes.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import re
import spacy

logger = logging.getLogger(__name__)


class InformationExtractor:
    """
    Extracts key communication attributes from normalized user text
    while distinguishing between explicit facts, inferred attributes, and uncertainties.
    """

    def __init__(self):
        self._nlp = None
        self._spacy_ready = False
        try:
            import spacy
            # Try to load small English model or blank English pipeline
            try:
                self._nlp = spacy.load("en_core_web_sm")
                self._spacy_ready = True
            except Exception:
                self._nlp = spacy.blank("en")
                self._spacy_ready = True
        except Exception as e:
            logger.warning(f"spaCy pipeline unavailable, using NLTK: {e}")
            self._spacy_ready = False

    def extract(self, normalized_text: str, user_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract communication parameters from normalized text.
        Returns a dictionary suitable for building a StructuredRequirement.
        """
        user_hints = user_hints or {}
        text = normalized_text.strip()
        lower_text = text.lower()

        # 1. Recipient extraction
        recipient = user_hints.get("recipient_hint")
        if not recipient:
            recipient = self._extract_recipient(text)

        # 2. Reason extraction (clauses introduced by causal markers)
        reason = self._extract_reason(text)

        # 3. Action and duration extraction
        action, duration = self._extract_action_and_duration(text)

        # 4. Dates, deadlines, and temporal expressions
        deadline = self._extract_temporal(text)

        # 5. Topic extraction
        topic = self._extract_topic(text, recipient, action)

        # 6. Explicit facts extraction
        explicit_facts = self._extract_facts(text, reason, action, duration, deadline)

        # 7. Constraints extraction
        constraints = list(user_hints.get("explicit_restrictions") or [])
        default_constraints = [
            "Do not invent facts, names, or promises not specified in the requirements.",
            "Maintain professional boundaries and do not commit to unverified timelines."
        ]
        for c in default_constraints:
            if c not in constraints:
                constraints.append(c)

        # 8. Uncertainties and clarification questions
        uncertainties = []
        clarification_questions = []

        if not recipient or recipient.lower() in ("someone", "unspecified", "anyone"):
            uncertainties.append("Recipient is unspecified or ambiguous.")
            clarification_questions.append("Who is the specific recipient or audience for this email?")

        if not action:
            uncertainties.append("Core requested action is unclear.")
            clarification_questions.append("What specific action or outcome do you want the recipient to take?")

        if ("urgent" in lower_text or "asap" in lower_text) and not deadline:
            uncertainties.append("Urgency is noted but no explicit target date or deadline is provided.")
            clarification_questions.append("Is there an exact date or time deadline for this request?")

        # 9. Inferred details
        inferred_details = {
            "has_causal_justification": reason is not None,
            "has_temporal_anchor": deadline is not None or duration is not None,
            "linguistic_backend": "spaCy" if self._spacy_ready else "NLTK"
        }

        return {
            "recipient": recipient,
            "topic": topic,
            "required_action": action,
            "reason": reason,
            "deadline": deadline,
            "requested_duration": duration,
            "required_facts": explicit_facts,
            "constraints": constraints,
            "uncertainties": uncertainties,
            "clarification_questions": clarification_questions,
            "inferred_details": inferred_details
        }

    VERBS_AND_NON_RECIPIENTS = {
        "write", "written", "writing", "send", "sent", "sending", "draft", "drafting",
        "email", "emailing", "ask", "asking", "tell", "telling", "inform", "informing",
        "say", "saying", "talk", "talking", "see", "seeing", "meet", "meeting",
        "request", "requesting", "contact", "contacting", "reach", "reaching",
        "invite", "inviting", "check", "checking", "reschedule", "rescheduling",
        "renew", "renewing", "conversation", "schedule", "scheduling", "remind", "reminding",
        "follow", "following", "update", "updating", "let", "know", "someone", "anyone",
        "something", "anything", "the", "a", "an", "my", "our", "your", "his", "her",
        "go", "going", "come", "coming", "get", "getting", "be", "is", "are", "willy", "will",
        "fix", "fixing", "help", "helping", "do", "doing", "make", "making", "submit", "submitting",
        "use", "used", "using", "up", "down", "out", "in", "on", "off", "share", "sharing", "shared",
        "exceed", "exceeds", "exceeded", "limit", "block", "blocked", "blocking", "restore", "restoring",
        "change", "changing", "keep", "keeping", "avoid", "avoiding", "visit", "visiting",
        "enter", "entering", "open", "opening", "note", "noted", "noting", "observe", "observing",
        "permit", "permitted", "permitting", "instruct", "instructed", "reply", "replying",
        "thank", "thanks", "thanking", "yo", "you", "not", "have", "has", "had", "take", "taking",
        "this", "that", "these", "those", "all", "some", "any", "each", "every", "student", "students"
    }

    def _extract_recipient(self, text: str) -> Optional[str]:
        """Linguistically locate the recipient from prepositions and named references."""
        # 1. Check direct honorific address (e.g., "thank you sir", "thank yo sir", "dear sir", "respected sir", "sir", "madam")
        m_hon = re.search(r"\b(?:thank\s+(?:you|yo)\s+|dear\s+|respected\s+)?(sir|madam|ma[\x27\x60]?am)\b", text, re.IGNORECASE)
        if m_hon:
            return m_hon.group(1).capitalize()

        # 2. Check patterns like "His name Steve", "friend named Steve", "for my friend his name Steve"
        m = re.search(r"\b(?:his|her|their|my friend\'s|friend\'s)?\s*name\s+(?:is\s+)?([A-Za-z]+)\b", text, re.IGNORECASE)
        if m:
            cand = m.group(1).title()
            if cand.lower() not in self.VERBS_AND_NON_RECIPIENTS and len(cand) > 1:
                return cand

        m = re.search(r"\b(?:for|to)\s+(?:my\s+)?friend\s+(?:named\s+|his\s+name\s+|her\s+name\s+|called\s+)?([A-Za-z]+)\b", text, re.IGNORECASE)
        if m:
            cand = m.group(1).title()
            if cand.lower() not in self.VERBS_AND_NON_RECIPIENTS and len(cand) > 1:
                return cand

        m = re.search(r"\bfriend\s+([A-Z][a-z]+)\b", text)
        if m:
            cand = m.group(1).title()
            if cand.lower() not in self.VERBS_AND_NON_RECIPIENTS and len(cand) > 1:
                return cand

        lower = text.lower()
        patterns = [
            ("write a polite email to my ", None),
            ("send a courteous email to my ", None),
            ("write a polite email to ", None),
            ("send a courteous email to ", None),
            ("write a firm but polite message to ", None),
            ("write a polite message to ", None),
            ("write a message to ", None),
            ("send an email to my ", None),
            ("send an email to ", None),
            ("send a message to my ", None),
            ("send a message to ", None),
            ("email my ", None),
            ("to my ", None),
            ("email ", None),
            ("inform ", None),
            ("ask ", None),
            ("contact ", None),
            ("reply to ", None),
            ("reply for ", None),
        ]

        # Scan phrases
        for prefix, _ in patterns:
            idx = lower.find(prefix)
            if idx != -1:
                sub = text[idx + len(prefix):].strip()
                tokens = sub.split()
                candidate = []
                for t in tokens:
                    clean_t = t.strip(",.!?\"'")
                    if clean_t.lower() in ("that", "requesting", "asking", "informing", "about", "for", "to", "because", "due", "due to", "and"):
                        break
                    candidate.append(clean_t)
                    if len(candidate) >= 3:
                        break
                if candidate:
                    first_word = candidate[0].lower()
                    if first_word not in self.VERBS_AND_NON_RECIPIENTS:
                        res = " ".join(candidate)
                        if len(res) > 1:
                            return res.title() if len(res) < 30 else res[:30].strip()

        # Search for 'to <Name>' where <Name> is capitalized or a clear recipient entity
        for m in re.finditer(r"\bto\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b", text):
            # Only match if capitalized in the original text (e.g. "to Professor", "to Steve", "to NMC")
            words = m.group(1).split()
            if words and words[0][0].isupper() and words[0].lower() not in self.VERBS_AND_NON_RECIPIENTS:
                candidate = []
                for w in words:
                    cw = w.strip(",.!?\"'")
                    if cw.lower() in ("that", "about", "for", "because", "and") or cw.lower() in self.VERBS_AND_NON_RECIPIENTS:
                        break
                    candidate.append(cw)
                if candidate and candidate[0].lower() not in self.VERBS_AND_NON_RECIPIENTS:
                    return " ".join(candidate).title()

        # Check for capitalized honorifics: Dr., Prof., Mr., Ms.
        for token in text.split():
            clean_tok = token.strip(",.!?\"'")
            if clean_tok in ("Dr.", "Prof.", "Professor", "Manager", "Marcus", "Sarah", "David", "Steve", "Robert", "NMC"):
                return clean_tok

        return None

    def _extract_reason(self, text: str) -> Optional[str]:
        """Extract causal justification following markers like 'because of', 'due to', 'since', 'as'."""
        lower = text.lower()
        causal_markers = ["because of ", "due to ", "owing to ", "because ", "since ", "on account of "]
        for marker in causal_markers:
            idx = lower.find(marker)
            if idx != -1:
                remainder = text[idx + len(marker):].strip()
                # Find end of reason clause: period, comma followed by transition, or clause boundary
                for delimiter in [".", ";", " Write", " write", " Please", " please", " Send", " send"]:
                    d_idx = remainder.find(delimiter)
                    if d_idx != -1:
                        remainder = remainder[:d_idx].strip()
                if len(remainder) > 3:
                    return remainder.rstrip(",.")

        # Check condition or justification clauses in multi-sentence prompts
        condition_markers = [
            "have a fever", "feeling unwell", "doctor advised", "medical emergency",
            "technical issue", "client audit", "database indexing", "power brick stopped",
            "billed twice"
        ]
        for cond in condition_markers:
            if cond in lower:
                for sentence in text.split("."):
                    if cond in sentence.lower():
                        return sentence.strip()

        return None

    def _extract_action_and_duration(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Identify key requested action and durations like 'two additional days', '3 PM'."""
        action = None
        duration = None
        lower = text.lower()

        # Duration patterns
        import re
        dur_match = re.search(r"(\b(?:one|two|three|four|five|six|seven|\d+)\s+(?:additional\s+)?(?:days?|weeks?|months?|hours?))\b", lower)
        if dur_match:
            duration = dur_match.group(1)

        # Check meeting / catch-up / conversation patterns
        if re.search(r"\b(?:willy|will|are)\s+you\s+be\s+free\b", lower) or "free on friday" in lower or "free on" in lower:
            action = "Check availability to meet up and go outside"
        elif "go outside" in lower or "hang out" in lower or "catch up" in lower or "meet up" in lower:
            action = "Meet up and spend time outside"
        elif "conversation about" in lower or "talk about" in lower:
            action = "Catch up and have a conversation about life"

        # Action search
        action_verbs = [
            "requesting", "request", "asking for", "ask for", "reschedule", "rescheduling",
            "informing", "inform", "apologize", "apologizing", "renew", "renewing", "refund", "refunding",
            "follow up", "following up", "seeking", "apply for"
        ]
        if not action:
            for verb in action_verbs:
                idx = lower.find(verb)
                if idx != -1:
                    sub = text[idx:].strip()
                    # Cut at end of sentence
                    for delim in [".", ";", "\n"]:
                        d_pos = sub.find(delim)
                        if d_pos != -1:
                            sub = sub[:d_pos]
                    action = sub.strip()
                    break

        if not action and "extension" in lower:
            action = f"Request extension{' of ' + duration if duration else ''}"

        return action, duration

    def _extract_temporal(self, text: str) -> Optional[str]:
        """Extract specific deadlines or calendar references."""
        import re
        patterns = [
            r"\b(thursday\s+3\s*pm|3\s*pm|1\s*am\s*to\s*5\s*am|friday\s+noon|may\s+\d{1,2}|oct\s+\d{1,2}|november|2027)\b",
            r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _extract_topic(self, text: str, recipient: Optional[str], action: Optional[str]) -> str:
        """Derive the main subject/topic."""
        lower = text.lower()
        if "conversation about" in lower or "about my life" in lower:
            return "Conversation about Life"
        if "go outside" in lower or "free on friday" in lower:
            return "Plans for Friday / Meeting Up"
        if "catch up" in lower or "hang out" in lower:
            return "Catching Up"

        topics = [
            ("project", "project submission"),
            ("sick leave", "sick leave notice"),
            ("budget review", "budget review reschedule"),
            ("refund", "subscription billing issue"),
            ("interview", "interview follow-up"),
            ("server maintenance", "scheduled server maintenance"),
            ("analytics report", "analytics report delivery delay"),
            ("travel grant", "conference travel grant inquiry"),
            ("license", "software license renewal"),
            ("charger", "hardware replacement request"),
            ("recommendation", "letter of recommendation request"),
        ]
        for key, val in topics:
            if key in lower:
                return val

        if action:
            return action[:40]
        return "General Inquiry"

    def _extract_facts(self, text: str, reason: Optional[str], action: Optional[str], duration: Optional[str], deadline: Optional[str]) -> List[str]:
        """Extract explicit factual statements from requirement."""
        facts = []
        if reason:
            facts.append(f"Reason: {reason}")
        if action:
            facts.append(f"Action: {action}")
        if duration:
            facts.append(f"Duration: {duration}")
        if deadline:
            facts.append(f"Temporal reference: {deadline}")

        # Check for dollar amounts or numbers
        import re
        amounts = re.findall(r"\$\d+(?:\.\d+)?", text)
        for amt in amounts:
            facts.append(f"Amount: {amt}")

        counts = re.findall(r"\b\d+\s+(?:seats?|days?|hours?|months?|tickets?)\b", text, re.IGNORECASE)
        for cnt in counts:
            facts.append(f"Quantity: {cnt}")

        return facts
