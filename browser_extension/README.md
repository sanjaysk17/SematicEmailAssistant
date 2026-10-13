# Semantic Email Assistant - Browser Extension (Manifest V3)

A browser extension that brings the **AI-Powered Email Writing Assistant with Semantic Analysis** directly into Google Chrome, Microsoft Edge, and Brave.

---

## Key Features

- ✍️ **Natural Language Email Assistant**: Generate email drafts from brief natural language instructions or selected web text.
- 📁 **Persistent User Drafts & Prompt Library**: Saves user email drafts and the original prompts that generated them in the SQLite database. Search, load back into editor, copy, or directly inject into webmail anytime.
- 🔒 **Zero-Config for End Users (Backend API Key Architecture)**: All AI API calls and processing run exclusively through the backend. Only **one single API key** (configured in the backend `.env`) processes requests for all extension users. End users never need their own API keys!
- 🔐 **Google Authenticator (TOTP 2FA) & Login**: Optional secure account protection with QR code scanning and 6-digit TOTP validation.
- 👤 **Persistent Persona Memory**: Remembers your identity (name, role, organization, preferred sign-off, personal background notes) and automatically tailors every draft.
- 📥 **Active Tab / Webmail Context Grabber**: Extract email thread content or selected text directly from the active webpage with one click.
- 🧠 **Interactive Semantic Representation**: View and edit the extracted structured representation (intent, tone, recipient, required action, reason, deadline, facts, constraints) before generation.
- 🔄 **Redo & Refine Draft Option**: If you are not satisfied with a generated draft, redo it instantly using 1-click presets (Shorter, More Formal, Friendlier, More Urgent, Alternative) or custom feedback instructions.
- ✉️ **One-Click Webmail Draft Injection**: Automatically injects generated subject and email body directly into Gmail (`mail.google.com`) or Outlook Web (`outlook.live.com` / `outlook.office.com`) compose boxes.
- 🛡️ **Multi-Criteria Validation Diagnostics**: View instant feedback on intent preservation, semantic similarity score, tone compliance, and slot completeness.

---

## Backend Setup (Centralized API Key)

The browser extension connects to the backend API. All LLM calls (Gemini, OpenAI, etc.) are processed by the backend using a single central API key.

### 1. Configure the API Key in the Backend
Edit `.env` in the project root:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-1.5-flash
```

*(Note: If no API key is provided, the backend seamlessly runs with the local offline mock provider for testing).*

### 2. Start the Backend Server
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Check health at: `http://127.0.0.1:8000/api/health`

---

## How to Install & Deploy the Extension

### In Google Chrome / Brave / Edge:
1. Open your browser and navigate to:
   - Chrome: `chrome://extensions`
   - Brave: `brave://extensions`
   - Edge: `edge://extensions`
2. Enable **Developer mode** (toggle switch in the top-right corner).
3. Click the **Load unpacked** button in the top-left corner.
4. Select the `browser_extension` folder:
   ```text
   /home/sanjay/Desktop/miniproject/browser_extension
   ```
5. The **Semantic Email Writing Assistant** icon will appear in your browser toolbar! Pin it for quick access.

### Production / Remote Backend Deployment
If you deploy the backend to a cloud host (Render, Railway, AWS, Fly.io, etc.):
1. Click the user persona icon 👤 in the extension header.
2. In the **Backend Server URL** field, enter your deployed backend URL (e.g. `https://my-email-assistant.onrender.com`).
3. Click **Save Settings**. The extension will now route all requests to your deployed server!

---

## How to Use

### 1. Generating an Email Draft
1. Click the extension icon.
2. In the **Assistant** tab, type your instruction (e.g., *"Ask professor for 2-day extension on assignment due to server failure"*).
3. Click **"⚡ 2. Generate & Validate Draft"**.
4. The draft is generated, validated, and **automatically saved to your Drafts Library with your prompt**!

### 2. Viewing & Managing Saved Drafts
1. Click the **📁 Drafts** tab in the top navigation bar.
2. View your complete list of saved drafts and their original prompts!
3. Use the search bar to filter drafts by prompt, subject, or recipient.
4. Actions available on each saved draft card:
   - **✏️ Load**: Loads the draft and original prompt into the editor.
   - **📨 Webmail**: Automatically injects the saved draft into Gmail or Outlook.
   - **📋 Copy**: Copies the full email to your clipboard.
   - **🗑️ Delete**: Permanently removes the draft from the database.

### 3. Auto-Injecting into Gmail or Outlook
1. Open Gmail or Outlook in a browser tab and click **Compose**.
2. Open the **Semantic Email Assistant** popup.
3. Click **"📨 Insert into Gmail / Outlook / Active Tab"**.
4. The draft subject and body are automatically inserted into your compose window!
