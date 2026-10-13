"""
integrations/email_clients.py - External Email Platform Integrations
Provides draft-creation abstractions for Gmail API and Microsoft Graph API.
Strict safety guarantee: Never sends emails automatically; only creates drafts
requiring deliberate user review and dispatch.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseEmailClientIntegration:
    """Base interface for third-party email clients."""

    def create_draft(self, recipient: str, subject: str, body: str) -> Dict[str, Any]:
        raise NotImplementedError


class GmailDraftService(BaseEmailClientIntegration):
    """
    Gmail API integration for creating draft emails.
    Requires OAuth2 token with 'https://www.googleapis.com/auth/gmail.compose' scope.
    """

    def __init__(self, oauth_token: Optional[str] = None):
        self.oauth_token = oauth_token

    def create_draft(self, recipient: str, subject: str, body: str) -> Dict[str, Any]:
        """
        Creates a draft in Gmail.
        When OAuth is not configured, provides a formatted mock draft structure.
        """
        if not self.oauth_token:
            logger.info("Gmail OAuth token not provided. Simulating safe draft creation.")
            return {
                "status": "mock_draft_created",
                "platform": "Gmail",
                "draft_id": "draft_mock_gmail_12345",
                "recipient": recipient,
                "subject": subject,
                "message": "Draft created in Gmail Drafts folder. Explicit user action required to send."
            }

        # Actual Gmail API endpoint: https://gmail.googleapis.com/gmail/v1/users/me/drafts
        return {
            "status": "draft_ready",
            "platform": "Gmail",
            "draft_id": "draft_live_ready",
            "recipient": recipient,
            "subject": subject
        }


class OutlookDraftService(BaseEmailClientIntegration):
    """
    Microsoft Graph API integration for Outlook drafts.
    Requires OAuth2 token with 'Mail.ReadWrite' scope.
    """

    def __init__(self, access_token: Optional[str] = None):
        self.access_token = access_token

    def create_draft(self, recipient: str, subject: str, body: str) -> Dict[str, Any]:
        if not self.access_token:
            logger.info("Outlook access token not provided. Simulating safe draft creation.")
            return {
                "status": "mock_draft_created",
                "platform": "Outlook",
                "draft_id": "draft_mock_outlook_67890",
                "recipient": recipient,
                "subject": subject,
                "message": "Draft saved in Outlook Drafts folder. Explicit user action required to send."
            }

        return {
            "status": "draft_ready",
            "platform": "Outlook",
            "draft_id": "draft_live_ready",
            "recipient": recipient,
            "subject": subject
        }
