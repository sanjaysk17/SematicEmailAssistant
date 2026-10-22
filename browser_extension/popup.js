/**
 * browser_extension/popup.js
 * Controller for the Semantic Email Writing Assistant Browser Extension.
 * Features:
 * - Authentication & Google Authenticator (TOTP 2FA)
 * - Persistent Persona Memory (remembers user name, role, background notes, sign-off)
 * - Semantic NLP requirement analysis & output validation
 * - Direct Gmail / Outlook compose box injection
 * - Scientific research evaluation benchmark runner
 */

let BACKEND_SERVER_URL = localStorage.getItem("backend_server_url") || "http://127.0.0.1:8000";
let API_BASE = BACKEND_SERVER_URL.replace(/\/+$/, "") + "/api";

document.addEventListener("DOMContentLoaded", () => {
  // --- STATE ---
  const state = {
    user: {
      id: null,
      name: "User",
      email: "user@example.com",
      job_title: "Professional",
      organization: "",
      preferred_sign_off: "Best regards,\nUser",
      about_me: ""
    },
    token: null,
    representation: null,
    requestId: null,
    draft: null,
    validation: null,
    savedDrafts: [],
    currentDraftId: null
  };

  // --- DOM ELEMENTS ---
  const tabs = document.querySelectorAll(".nav-tab");
  const tabContents = document.querySelectorAll(".tab-content");
  const backendStatus = document.getElementById("backend-status");
  const backendStatusText = document.getElementById("backend-status-text");
  const globalSpinner = document.getElementById("global-spinner");
  const globalSpinnerText = document.getElementById("global-spinner-text");
  const appToast = document.getElementById("app-toast");
  const btnPopout = document.getElementById("btn-popout");

  // Profile Header Elements
  const btnUserProfile = document.getElementById("btn-user-profile");
  const userDisplayName = document.getElementById("user-display-name");
  const userSubtitle = document.getElementById("user-subtitle");
  const personaSummaryName = document.getElementById("persona-summary-name");
  const personaSummaryRole = document.getElementById("persona-summary-role");

  // Profile Modal Elements
  const modalProfile = document.getElementById("modal-profile");
  const btnCloseProfile = document.getElementById("btn-close-profile");
  const profileNameInput = document.getElementById("profile-name");
  const profileJobInput = document.getElementById("profile-job");
  const profileOrgInput = document.getElementById("profile-org");
  const profileSignoffInput = document.getElementById("profile-signoff");
  const profileAboutmeInput = document.getElementById("profile-aboutme");
  const backendUrlInput = document.getElementById("backend-url-input");
  const btnSaveProfile = document.getElementById("btn-save-profile");
  const btnLogout = document.getElementById("btn-logout");

  // Auth Modal Elements
  const modalAuth = document.getElementById("modal-auth");
  const btnCloseAuth = document.getElementById("btn-close-auth");
  const authTabLogin = document.getElementById("auth-tab-login");
  const authTabRegister = document.getElementById("auth-tab-register");
  const viewLogin = document.getElementById("view-login");
  const viewRegister = document.getElementById("view-register");
  const loginEmail = document.getElementById("login-email");
  const loginPassword = document.getElementById("login-password");
  const viewLoginTotp = document.getElementById("view-login-totp");
  const loginTotpCode = document.getElementById("login-totp-code");
  const btnSubmitLogin = document.getElementById("btn-submit-login");
  const btnQuickDemoLogin = document.getElementById("btn-quick-demo-login");

  // Register View Elements
  const regStep1 = document.getElementById("register-step-1");
  const regStep2 = document.getElementById("register-step-2");
  const regName = document.getElementById("reg-name");
  const regEmail = document.getElementById("reg-email");
  const regPassword = document.getElementById("reg-password");
  const regAboutme = document.getElementById("reg-aboutme");
  const btnSubmitRegister = document.getElementById("btn-submit-register");
  const regQrImg = document.getElementById("reg-qr-img");
  const regSecretKey = document.getElementById("reg-secret-key");
  const regTotpVerify = document.getElementById("reg-totp-verify");
  const btnVerifyRegTotp = document.getElementById("btn-verify-reg-totp");

  // Tab 1 (Assistant) Elements
  const reqInput = document.getElementById("requirement-input");
  const promptSpecificityWarning = document.getElementById("prompt-specificity-warning");
  const extensionTemplateSelect = document.getElementById("extension-template-select");
  const btnGrabContext = document.getElementById("btn-grab-context");
  const recipientHint = document.getElementById("recipient-hint");
  const toneHint = document.getElementById("tone-hint");
  const lengthHint = document.getElementById("length-hint");
  const languageHint = document.getElementById("language-hint");
  const contextInput = document.getElementById("context-input");
  const btnAnalyze = document.getElementById("btn-analyze");
  const btnGenerateDirect = document.getElementById("btn-generate-direct");
  const btnReplyDirect = document.getElementById("btn-reply-direct");

  // Professional Template Catalog for 1-Click Scenario Auto-Fill
  const EXTENSION_TEMPLATES = {
    exec_status: {
      req: "Please draft a concise executive progress report to my department head summarizing our team's completion of Phase 2 deliverables, system stability, and upcoming roadmap milestones for next quarter.",
      recipient: "Department Head",
      tone: "polite and professional",
      length: "medium"
    },
    exec_reschedule: {
      req: "Our team needs to reschedule tomorrow's Q3 budget review call to Thursday 3 PM due to an urgent client audit. Send a courteous email to David.",
      recipient: "David",
      tone: "courteous and urgent",
      length: "medium"
    },
    exec_budget: {
      req: "Submit a formal proposal to director requesting $15,000 additional budget for cloud infrastructure upgrades to support user growth.",
      recipient: "Director",
      tone: "assertive and direct",
      length: "medium"
    },
    client_billing: {
      req: "I was billed twice ($49.99 each) for my monthly cloud subscription on Oct 10. Write a firm but polite message to billing support asking for a refund of the duplicate charge.",
      recipient: "Billing Support",
      tone: "firm and polite",
      length: "medium"
    },
    client_renewal: {
      req: "Negotiate annual software contract renewal with account manager Rachel, requesting a 15% multi-year discount and revised SLA terms.",
      recipient: "Rachel",
      tone: "diplomatic and respectful",
      length: "medium"
    },
    client_followup: {
      req: "Follow up with client Mark on the enterprise software proposal submitted last Monday. Inquire if they have reviewed it and offer a 15-minute walkthrough.",
      recipient: "Mark",
      tone: "persuasive and enthusiastic",
      length: "medium"
    },
    leave_sick: {
      req: "I have a fever and doctor advised rest. Email my manager Sarah informing her I will be taking sick leave today Monday and tomorrow Tuesday.",
      recipient: "Sarah",
      tone: "polite and professional",
      length: "short"
    },
    leave_vacation: {
      req: "Submit a formal vacation leave notice to manager Michael for next week Monday to Friday, confirming handover of active tasks to Sarah.",
      recipient: "Michael",
      tone: "polite and professional",
      length: "short"
    },
    acad_extension: {
      req: "I could not complete my project because of a technical issue. Write a polite email to my professor requesting two additional days.",
      recipient: "Professor",
      tone: "polite and professional",
      length: "medium"
    },
    acad_rec: {
      req: "Write a respectful email to Professor Davis requesting a letter of recommendation for a graduate fellowship application before November 15th.",
      recipient: "Professor Davis",
      tone: "appreciative and formal",
      length: "medium"
    },
    net_coffee: {
      req: "I want to write to my friend Steve asking if he will be free on Friday to go outside and have a conversation about my life.",
      recipient: "Steve",
      tone: "friendly and casual",
      length: "short"
    },
    vague_test: {
      req: "Things happened and stuff is weird. Email someone to fix it ASAP.",
      recipient: "",
      tone: "",
      length: "medium"
    }
  };

  if (extensionTemplateSelect) {
    extensionTemplateSelect.addEventListener("change", (e) => {
      const key = e.target.value;
      if (!key || !EXTENSION_TEMPLATES[key]) return;
      const tmpl = EXTENSION_TEMPLATES[key];
      if (reqInput) reqInput.value = tmpl.req;
      if (recipientHint) recipientHint.value = tmpl.recipient || "";
      if (toneHint) toneHint.value = tmpl.tone || "";
      if (lengthHint) lengthHint.value = tmpl.length || "medium";
      checkPromptSpecificity();
      showToast("Template scenario applied!", "info");
    });
  }

  // Real-time Prompt Specificity Checker
  function checkPromptSpecificity() {
    if (!promptSpecificityWarning || !reqInput) return;
    const text = reqInput.value.trim();
    if (!text) {
      promptSpecificityWarning.classList.add("hidden");
      return;
    }
    const words = text.split(/\s+/);
    const triggers = ["weird", "stuff", "asap", "things", "whatever", "someone", "something", "fix it"];
    const hasTrigger = triggers.some(t => text.toLowerCase().includes(t));
    if (words.length < 8 || hasTrigger) {
      promptSpecificityWarning.classList.remove("hidden");
    } else {
      promptSpecificityWarning.classList.add("hidden");
    }
  }
  if (reqInput) {
    reqInput.addEventListener("input", checkPromptSpecificity);
  }

  // Tab 2 (Semantics) Elements
  const semanticsEmpty = document.getElementById("semantics-empty");
  const semanticsContent = document.getElementById("semantics-content");
  const repIntentBadge = document.getElementById("rep-intent-badge");
  const repToneBadge = document.getElementById("rep-tone-badge");
  const repConfidenceBadge = document.getElementById("rep-confidence-badge");
  const repIntentInput = document.getElementById("rep-intent-input");
  const repRecipientInput = document.getElementById("rep-recipient-input");
  const repToneInput = document.getElementById("rep-tone-input");
  const repTopicInput = document.getElementById("rep-topic-input");
  const repActionInput = document.getElementById("rep-action-input");
  const repDeadlineInput = document.getElementById("rep-deadline-input");
  const repReasonInput = document.getElementById("rep-reason-input");
  const repFactsContainer = document.getElementById("rep-facts-container");
  const repConstraintsContainer = document.getElementById("rep-constraints-container");
  const repUncertaintiesCard = document.getElementById("rep-uncertainties-card");
  const repUncertaintiesList = document.getElementById("rep-uncertainties-list");
  const btnGenerateFromRep = document.getElementById("btn-generate-from-rep");

  // Tab 3 (Draft) Elements
  const draftEmpty = document.getElementById("draft-empty");
  const draftContent = document.getElementById("draft-content");
  const draftGeneralNotice = document.getElementById("draft-general-notice");
  const draftSubject = document.getElementById("draft-subject");
  const draftBody = document.getElementById("draft-body");
  const draftSignName = document.getElementById("draft-sign-name");
  const draftWordCount = document.getElementById("draft-word-count");
  const draftModelTag = document.getElementById("draft-model-tag");
  const draftLatencyTag = document.getElementById("draft-latency-tag");
  const draftValTag = document.getElementById("draft-val-tag");
  const btnCopySubject = document.getElementById("btn-copy-subject");
  const btnCopyAll = document.getElementById("btn-copy-all");
  const btnInjectWebmail = document.getElementById("btn-inject-webmail");
  const btnViewValidation = document.getElementById("btn-view-validation");
  const btnSaveCurrentDraft = document.getElementById("btn-save-current-draft");
  const btnViewAllDrafts = document.getElementById("btn-view-all-drafts");
  const draftAutosaveTag = document.getElementById("draft-autosave-tag");
  const redoFeedbackInput = document.getElementById("redo-feedback-input");
  const btnRedoDraft = document.getElementById("btn-redo-draft");

  // Tab: Saved Drafts Elements
  const draftsCountBadge = document.getElementById("drafts-count-badge");
  const draftsCountSummary = document.getElementById("drafts-count-summary");
  const btnRefreshDrafts = document.getElementById("btn-refresh-drafts");
  const draftsSearchInput = document.getElementById("drafts-search-input");
  const draftsEmpty = document.getElementById("drafts-empty");
  const draftsListContainer = document.getElementById("drafts-list-container");

  // Tab: Validation Elements
  const valEmpty = document.getElementById("val-empty");
  const valContent = document.getElementById("val-content");
  const valStatusPill = document.getElementById("val-status-pill");
  const metricIntent = document.getElementById("metric-intent");
  const metricIntentDesc = document.getElementById("metric-intent-desc");
  const metricTone = document.getElementById("metric-tone");
  const metricToneDesc = document.getElementById("metric-tone-desc");
  const metricCompleteness = document.getElementById("metric-completeness");
  const metricCompletenessBar = document.getElementById("metric-completeness-bar");
  const metricSimilarity = document.getElementById("metric-similarity");
  const metricSimilarityBar = document.getElementById("metric-similarity-bar");
  const valMissingCard = document.getElementById("val-missing-card");
  const valMissingList = document.getElementById("val-missing-list");
  const valUnsupportedCard = document.getElementById("val-unsupported-card");
  const valUnsupportedList = document.getElementById("val-unsupported-list");
  const valSuggestionsList = document.getElementById("val-suggestions-list");

  let pendingRegistrationEmail = null;

  // --- INITIALIZATION ---
  loadPersistedUser();
  checkBackendHealth();
  checkPendingContextMenuRequirement();
  loadSavedDrafts();

  // --- AUTH HEADER HELPER ---
  function getAuthHeaders(extraHeaders = {}) {
    const headers = { "Content-Type": "application/json", ...extraHeaders };
    if (state.token) {
      headers["Authorization"] = `Bearer ${state.token}`;
    }
    return headers;
  }

  // --- PERSISTENCE: LOAD USER & PERSONA MEMORY ---
  function loadPersistedUser() {
    const saved = localStorage.getItem("assistant_user");
    if (saved) {
      try {
        state.user = JSON.parse(saved);
        state.token = localStorage.getItem("assistant_token");
      } catch (e) {
        console.warn("Could not parse saved user:", e);
      }
    }
    updateUserUI();
    if (state.token) {
      syncProfileFromServer();
    }
  }

  async function syncProfileFromServer() {
    if (!state.token) return;
    try {
      const resp = await fetch(`${API_BASE}/auth/profile`, {
        method: "GET",
        headers: getAuthHeaders()
      });
      if (resp.ok) {
        const prof = await resp.json();
        state.user = { ...state.user, ...prof };
        localStorage.setItem("assistant_user", JSON.stringify(state.user));
        updateUserUI();
      } else if (resp.status === 401) {
        // Token expired or invalid
        console.warn("Session token expired.");
        state.token = null;
        localStorage.removeItem("assistant_token");
        updateUserUI();
      }
    } catch (e) {
      console.warn("Could not sync profile from server:", e);
    }
  }

  function saveUserPersona(user, token) {
    state.user = { ...state.user, ...user };
    if (token) state.token = token;
    localStorage.setItem("assistant_user", JSON.stringify(state.user));
    if (token) localStorage.setItem("assistant_token", state.token);
    updateUserUI();
    // Immediately reload drafts isolated to this user
    loadSavedDrafts();
  }

  function updateUserUI() {
    const isLoggedIn = !!state.token;
    const name = state.user.name || (isLoggedIn ? "User" : "Log In");
    userDisplayName.innerText = isLoggedIn ? name : "Log In / Demo";
    userSubtitle.innerText = isLoggedIn ? `Personalized for ${name}` : "Click to Sign In / Demo Login";
    personaSummaryName.innerText = isLoggedIn ? name : "Unauthenticated (Click to Log In)";
    personaSummaryRole.innerText = isLoggedIn ? (state.user.job_title || "Identity Active") : "Sign in to isolate & save your drafts";
    draftSignName.innerText = state.user.name || "User";

    // Sync input fields in Profile Modal
    profileNameInput.value = state.user.name || "";
    profileJobInput.value = state.user.job_title || "";
    profileOrgInput.value = state.user.organization || "";
    profileSignoffInput.value = state.user.preferred_sign_off || (state.user.name ? `Best regards,\n${state.user.name}` : "Best regards,\nUser");
    profileAboutmeInput.value = state.user.about_me || "";
  }

  // --- TAB NAVIGATION ---
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      switchTab(tab.getAttribute("data-tab"));
    });
  });

  function switchTab(tabId) {
    tabs.forEach(t => t.classList.toggle("active", t.getAttribute("data-tab") === tabId));
    tabContents.forEach(c => c.classList.toggle("active", c.id === tabId));
  }

  // --- POP-OUT TO FULL TAB ---
  btnPopout.addEventListener("click", () => {
    if (typeof chrome !== "undefined" && chrome.tabs && chrome.tabs.create) {
      chrome.tabs.create({ url: chrome.runtime.getURL("popup.html") });
    } else {
      window.open(window.location.href, "_blank");
    }
  });

  // --- BACKEND HEALTH CHECK ---
  async function checkBackendHealth() {
    try {
      const resp = await fetch(`${API_BASE}/health`, { method: "GET" });
      if (resp.ok) {
        const data = await resp.json();
        backendStatus.className = "status-pill online";
        backendStatusText.innerText = "API Connected";
        backendStatus.title = `Provider: ${data.llm_provider || 'local'} | Engine: ${data.embedding_engine || 'Active'}`;
      } else {
        throw new Error("HTTP " + resp.status);
      }
    } catch (e) {
      backendStatus.className = "status-pill offline";
      backendStatusText.innerText = "API Offline";
      backendStatus.title = "Ensure backend is running at http://127.0.0.1:8000";
    }
  }

  // --- CHECK CONTEXT MENU SELECTION ---
  function checkPendingContextMenuRequirement() {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      chrome.storage.local.get(["pending_requirement", "pending_source_url"], (data) => {
        if (data.pending_requirement) {
          reqInput.value = data.pending_requirement;
          if (data.pending_source_url) {
            contextInput.value = `Extracted from: ${data.pending_source_url}`;
          }
          chrome.storage.local.remove(["pending_requirement", "pending_source_url"]);
          showToast("Imported selected text from web page!");
        }
      });
    }
  }

  // --- PROFILE MODAL CONTROLS ---
  btnUserProfile.addEventListener("click", () => {
    if (!state.token) {
      modalAuth.classList.remove("hidden");
    } else {
      updateUserUI();
      if (backendUrlInput) {
        backendUrlInput.value = localStorage.getItem("backend_server_url") || "http://127.0.0.1:8000";
      }
      modalProfile.classList.remove("hidden");
    }
  });

  // Persona banner click also navigates to profile / login
  const personaBanner = document.querySelector(".persona-active-banner");
  if (personaBanner) {
    personaBanner.style.cursor = "pointer";
    personaBanner.addEventListener("click", () => {
      if (!state.token) {
        modalAuth.classList.remove("hidden");
      } else {
        updateUserUI();
        modalProfile.classList.remove("hidden");
      }
    });
  }

  btnCloseProfile.addEventListener("click", () => {
    modalProfile.classList.add("hidden");
  });

  btnSaveProfile.addEventListener("click", async () => {
    const updated = {
      name: profileNameInput.value.trim() || "User",
      job_title: profileJobInput.value.trim() || null,
      organization: profileOrgInput.value.trim() || null,
      preferred_sign_off: profileSignoffInput.value.trim() || (profileNameInput.value.trim() ? `Best regards,\n${profileNameInput.value.trim()}` : null),
      about_me: profileAboutmeInput.value.trim() || null
    };

    saveUserPersona(updated);

    // Save Backend Server URL if modified
    if (backendUrlInput && backendUrlInput.value.trim()) {
      const cleanUrl = backendUrlInput.value.trim().replace(/\/+$/, "");
      localStorage.setItem("backend_server_url", cleanUrl);
      BACKEND_SERVER_URL = cleanUrl;
      API_BASE = cleanUrl + "/api";
      checkBackendHealth();
      loadSavedDrafts();
    }

    // Sync to backend SQLite
    try {
      const resp = await fetch(`${API_BASE}/auth/profile`, {
        method: "PUT",
        headers: getAuthHeaders(),
        body: JSON.stringify(updated)
      });
      if (resp.ok) {
        const saved = await resp.json();
        state.user = { ...state.user, ...saved };
        updateUserUI();
      }
    } catch (err) {
      console.warn("Could not sync profile to backend:", err);
    }

    modalProfile.classList.add("hidden");
    showToast(`Persona memory updated for ${updated.name}!`);
  });

  btnLogout.addEventListener("click", () => {
    state.token = null;
    state.user = {
      id: null,
      name: "Guest",
      email: "",
      job_title: "",
      organization: "",
      preferred_sign_off: "",
      about_me: ""
    };
    state.savedDrafts = [];
    localStorage.removeItem("assistant_token");
    localStorage.removeItem("assistant_user");
    localStorage.removeItem("assistant_local_drafts");
    updateUserUI();
    renderDrafts([]);
    updateDraftsCounter();
    modalProfile.classList.add("hidden");
    modalAuth.classList.remove("hidden");
    showToast("Logged out securely. Session ended.");
  });

  // --- AUTH MODAL & GOOGLE AUTHENTICATOR (2FA) CONTROLS ---
  btnCloseAuth.addEventListener("click", () => {
    modalAuth.classList.add("hidden");
  });

  authTabLogin.addEventListener("click", () => {
    authTabLogin.classList.add("active");
    authTabRegister.classList.remove("active");
    viewLogin.classList.remove("hidden");
    viewRegister.classList.add("hidden");
  });

  authTabRegister.addEventListener("click", () => {
    authTabRegister.classList.add("active");
    authTabLogin.classList.remove("active");
    viewRegister.classList.remove("hidden");
    viewLogin.classList.add("hidden");
    regStep1.classList.remove("hidden");
    regStep2.classList.add("hidden");
  });

  // 1-Click Instant Demo Login
  btnQuickDemoLogin.addEventListener("click", async () => {
    setLoading(true, "Authenticating demo user...");
    try {
      const resp = await fetch(`${API_BASE}/auth/demo-login`, { method: "POST" });
      if (!resp.ok) throw new Error("Demo login failed");
      const data = await resp.json();
      saveUserPersona(data.user, data.access_token);
      modalAuth.classList.add("hidden");
      showToast(`Welcome, ${data.user.name || "User"}! Persona memory active.`);
    } catch (e) {
      showToast("Login error: " + e.message);
    } finally {
      setLoading(false);
    }
  });

  // Regular Email/Password Login
  btnSubmitLogin.addEventListener("click", async () => {
    const email = loginEmail.value.trim();
    const password = loginPassword.value;
    if (!email || !password) return showToast("Enter email and password.");

    // If 2FA code is already showing, verify code
    if (!viewLoginTotp.classList.contains("hidden")) {
      const code = loginTotpCode.value.trim();
      if (!code || code.length < 6) return showToast("Enter the 6-digit Google Authenticator code.");

      setLoading(true, "Verifying 2FA code...");
      try {
        const resp = await fetch(`${API_BASE}/auth/verify-2fa`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, totp_code: code })
        });
        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.detail || "Invalid 2FA code");
        }
        const data = await resp.json();
        saveUserPersona(data.user, data.access_token);
        modalAuth.classList.add("hidden");
        showToast(`Welcome back, ${data.user.name}!`);
      } catch (e) {
        showToast(e.message);
      } finally {
        setLoading(false);
      }
      return;
    }

    setLoading(true, "Validating credentials...");
    try {
      const resp = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Invalid login");
      }
      const data = await resp.json();

      if (data.requires_2fa) {
        viewLoginTotp.classList.remove("hidden");
        loginTotpCode.focus();
        showToast("Enter 6-digit code from Google Authenticator.");
      } else {
        saveUserPersona(data.user, data.access_token);
        modalAuth.classList.add("hidden");
        showToast(`Welcome, ${data.user.name}!`);
      }
    } catch (e) {
      showToast(e.message);
    } finally {
      setLoading(false);
    }
  });

  // Registration Step 1: Submit details -> Get Google Authenticator QR Code
  btnSubmitRegister.addEventListener("click", async () => {
    const name = regName.value.trim();
    const email = regEmail.value.trim();
    const password = regPassword.value;
    const aboutMe = regAboutme.value.trim();

    if (!name || !email || !password) return showToast("Please fill in Name, Email, and Password.");

    setLoading(true, "Creating account & generating Google Authenticator QR code...");
    try {
      const resp = await fetch(`${API_BASE}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          email,
          password,
          about_me: aboutMe || null
        })
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Registration failed");
      }

      const data = await resp.json();
      pendingRegistrationEmail = data.email;

      // Show QR Code and Secret Key in Step 2
      regQrImg.src = data.qr_code_data_url;
      regSecretKey.innerText = data.totp_secret;

      regStep1.classList.add("hidden");
      regStep2.classList.remove("hidden");
      showToast("Scan the QR code with Google Authenticator on your phone!");
    } catch (e) {
      showToast(e.message);
    } finally {
      setLoading(false);
    }
  });

  // Registration Step 2: Verify 6-digit TOTP code
  btnVerifyRegTotp.addEventListener("click", async () => {
    const code = regTotpVerify.value.trim();
    if (!code || code.length < 6) return showToast("Enter the 6-digit Google Authenticator code.");

    setLoading(true, "Verifying 2FA setup...");
    try {
      const resp = await fetch(`${API_BASE}/auth/verify-2fa`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: pendingRegistrationEmail,
          totp_code: code
        })
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Invalid 2FA code");
      }

      const data = await resp.json();
      saveUserPersona(data.user, data.access_token);
      modalAuth.classList.add("hidden");
      showToast(`2FA Setup complete! Logged in as ${data.user.name}.`);
    } catch (e) {
      showToast(e.message);
    } finally {
      setLoading(false);
    }
  });

  // --- ACTIVE TAB / WEB PAGE CONTEXT GRABBER (WEB LOAD BUTTON) ---
  btnGrabContext.addEventListener("click", async () => {
    if (typeof chrome === "undefined" || !chrome.tabs || !chrome.tabs.query) {
      showToast("Web text load is active in browser extension mode.");
      return;
    }

    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab) return showToast("No active browser tab found.");

      let extractedText = "";
      let pageTitle = tab.title || "";
      let pageUrl = tab.url || "";

      // Step 1: Extract selection across all frames using executeScript
      if (chrome.scripting && chrome.scripting.executeScript) {
        try {
          const results = await chrome.scripting.executeScript({
            target: { tabId: tab.id, allFrames: true },
            func: () => {
              // 1. Text selection on window
              let text = window.getSelection ? window.getSelection().toString().trim() : "";
              // 2. Active input/textarea selection
              if (!text && document.activeElement) {
                const el = document.activeElement;
                if ((el.tagName === "TEXTAREA" || el.tagName === "INPUT") && typeof el.selectionStart === "number") {
                  text = el.value.substring(el.selectionStart, el.selectionEnd).trim();
                }
              }
              // 3. Fallback: If inside Gmail or Outlook and no text highlighted, grab email thread
              if (!text && window.location.href.includes("mail.google.com")) {
                const bodies = document.querySelectorAll(".a3s.aiL, div[dir='ltr'], .ii.gt, .adn.ads");
                if (bodies.length > 0) text = bodies[bodies.length - 1].innerText.trim().substring(0, 2000);
              } else if (!text && (window.location.href.includes("outlook.live.com") || window.location.href.includes("outlook.office.com") || window.location.href.includes("outlook.office365.com"))) {
                const bodies = document.querySelectorAll("div[aria-label='Message body'], .ItemPartBody, div[role='document']");
                if (bodies.length > 0) text = bodies[bodies.length - 1].innerText.trim().substring(0, 2000);
              }
              return {
                text: text || "",
                title: document.title || "",
                url: window.location.href || ""
              };
            }
          });

          if (results && results.length > 0) {
            const hit = results.find(r => r.result && r.result.text);
            if (hit && hit.result.text) {
              extractedText = hit.result.text;
              if (hit.result.title) pageTitle = hit.result.title;
              if (hit.result.url) pageUrl = hit.result.url;
            }
          }
        } catch (e) {
          console.warn("executeScript extraction fallback to message passing:", e);
        }
      }

      // Step 2: Fallback to sendMessage if executeScript produced no text
      if (!extractedText) {
        try {
          const msgRes = await new Promise((resolve) => {
            chrome.tabs.sendMessage(tab.id, { action: "EXTRACT_PAGE_CONTEXT" }, (res) => {
              if (chrome.runtime.lastError || !res) resolve(null);
              else resolve(res);
            });
          });
          if (msgRes && msgRes.selectedText) {
            extractedText = msgRes.selectedText;
            if (msgRes.pageTitle) pageTitle = msgRes.pageTitle;
            if (msgRes.url) pageUrl = msgRes.url;
          }
        } catch (e) {}
      }

      // Step 3: Inject extracted text directly into requirement/prompt box
      if (extractedText) {
        const currentPrompt = reqInput.value.trim();
        if (!currentPrompt) {
          reqInput.value = extractedText;
        } else {
          reqInput.value = `${currentPrompt}\n\n[Web Selection]:\n${extractedText}`;
        }

        // Also record full context in preferences
        if (contextInput) {
          contextInput.value = `Web Source: "${pageTitle}" (${pageUrl})\n${extractedText.substring(0, 1000)}`;
          const accordion = document.querySelector(".accordion");
          if (accordion) accordion.open = true;
        }

        reqInput.focus();
        reqInput.dispatchEvent(new Event("input", { bubbles: true }));
        showToast("Web selection injected into email prompt!");
      } else {
        showToast("No text selected. Highlight text on any page and click Web Load!");
      }
    } catch (e) {
      showToast("Could not grab tab text: " + e.message);
    }
  });

  // --- 1. ANALYZE SEMANTICS ---
  btnAnalyze.addEventListener("click", async () => {
    const text = reqInput.value.trim();
    if (!text) {
      showToast("Please describe what email you would like to write.");
      reqInput.focus();
      return;
    }

    setLoading(true, "Analyzing requirement with semantic NLP engine...");
    try {
      const payload = {
        requirement: text,
        context: contextInput.value.trim() || null,
        recipient_hint: recipientHint.value.trim() || null,
        tone_hint: toneHint.value ? toneHint.value.toLowerCase() : null,
        desired_length: lengthHint.value || "medium",
        language: languageHint.value || "English"
      };

      const resp = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Analysis failed");
      }

      const data = await resp.json();
      state.representation = data.representation;
      state.requestId = data.request_id;

      populateSemanticsTab(data);
      switchTab("tab-semantics");
      showToast("Semantic analysis complete!");
    } catch (err) {
      showToast("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  });

  function populateSemanticsTab(data) {
    const rep = data.representation;
    repIntentBadge.innerText = `Intent: ${rep.intent || 'Unspecified'}`;
    repToneBadge.innerText = `Tone: ${rep.tone || 'Professional'}`;
    repConfidenceBadge.innerText = `Confidence: ${(data.confidence * 100).toFixed(0)}%`;

    repIntentInput.value = rep.intent || "";
    repRecipientInput.value = rep.recipient || "";
    repToneInput.value = rep.tone || "";
    repTopicInput.value = rep.topic || "";
    repActionInput.value = rep.required_action || "";
    repDeadlineInput.value = rep.deadline || "";
    repReasonInput.value = rep.reason || "";

    // Facts
    repFactsContainer.innerHTML = "";
    if (rep.required_facts && rep.required_facts.length > 0) {
      rep.required_facts.forEach(f => {
        const item = document.createElement("div");
        item.className = "tag-item";
        item.innerText = `• ${f}`;
        repFactsContainer.appendChild(item);
      });
    } else {
      repFactsContainer.innerHTML = '<span class="text-light">No explicit facts required.</span>';
    }

    // Constraints
    repConstraintsContainer.innerHTML = "";
    if (rep.constraints && rep.constraints.length > 0) {
      rep.constraints.forEach(c => {
        const item = document.createElement("div");
        item.className = "tag-item";
        item.innerText = `🛡️ ${c}`;
        repConstraintsContainer.appendChild(item);
      });
    } else {
      repConstraintsContainer.innerHTML = '<span class="text-light">Standard professional boundaries applied.</span>';
    }

    // Ambiguities / Specificity Warning
    if (repUncertaintiesCard && repUncertaintiesList) {
      const uncertainties = rep.uncertainties || [];
      if (uncertainties.length > 0 || data.is_underspecified) {
        repUncertaintiesList.innerHTML = "";
        const items = uncertainties.length > 0 ? uncertainties : ["Prompt lacks specific details. Generated email will be generic unless more context (recipient, reason, dates) is provided."];
        items.forEach(u => {
          const div = document.createElement("div");
          div.style.marginBottom = "4px";
          div.innerText = `• ${u}`;
          repUncertaintiesList.appendChild(div);
        });
        repUncertaintiesCard.classList.remove("hidden");
      } else {
        repUncertaintiesCard.classList.add("hidden");
      }
    }

    semanticsEmpty.classList.add("hidden");
    semanticsContent.classList.remove("hidden");
  }

  // --- 2. GENERATE & VALIDATE (WITH SENDER PERSONA MEMORY) ---
  btnGenerateDirect.addEventListener("click", () => executeGenerateAndValidate(false));
  btnGenerateFromRep.addEventListener("click", () => executeGenerateAndValidate(true));
  if (btnReplyDirect) {
    btnReplyDirect.addEventListener("click", () => executeReplyDirect());
  }

  async function executeReplyDirect() {
    const text = reqInput.value.trim();
    if (!text) {
      showToast("Please enter or paste the email you want to reply to.");
      reqInput.focus();
      return;
    }

    setLoading(true, "Analyzing incoming email and drafting reply...");
    try {
      // Step A: Analyze incoming email in reply mode
      const aPayload = {
        requirement: text,
        context: contextInput.value.trim() || null,
        recipient_hint: recipientHint.value.trim() || null,
        tone_hint: toneHint.value ? toneHint.value.toLowerCase() : null,
        desired_length: lengthHint.value || "medium",
        language: languageHint.value || "English",
        generation_mode: "reply"
      };
      const aResp = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify(aPayload)
      });
      if (!aResp.ok) throw new Error("Semantic analysis failed");
      const aData = await aResp.json();
      state.representation = aData.representation;
      state.requestId = aData.request_id;
      populateSemanticsTab(aData);

      // Step B: Call /reply endpoint
      setLoading(true, `Drafting contextual reply for ${state.user.name}...`);
      const gPayload = {
        requirement: text,
        representation: state.representation,
        context: contextInput.value.trim() || null,
        desired_length: lengthHint.value || "medium",
        language: languageHint.value || "English",
        system_version: "proposed",
        generation_mode: "reply",
        request_id: state.requestId,
        sender_name: state.user.name,
        sender_role: state.user.job_title,
        sender_organization: state.user.organization,
        sender_sign_off: state.user.preferred_sign_off,
        sender_about_me: state.user.about_me
      };

      const gResp = await fetch(`${API_BASE}/reply`, {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify(gPayload)
      });
      if (!gResp.ok) throw new Error("Reply generation failed");
      const gData = await gResp.json();
      state.draft = gData;
      state.currentDraftId = gData.draft_id || null;
      populateDraftTab(gData);

      loadSavedDrafts();
      switchTab("tab-draft");
      showToast("✅ Reply generated successfully! You can review and edit below.");
    } catch (err) {
      console.error(err);
      showToast(`Error: ${err.message || "Failed to generate reply"}`);
    } finally {
      setLoading(false);
    }
  }

  async function executeGenerateAndValidate(fromEditedRep = false) {
    const text = reqInput.value.trim();
    if (!text) {
      showToast("Please describe the email you want to write.");
      reqInput.focus();
      return;
    }

    setLoading(true, "Personalizing and drafting email...");
    try {
      // Step A: Analyze if not analyzed
      if (!state.representation && !fromEditedRep) {
        setLoading(true, "Interpreting semantic requirements...");
        const aPayload = {
          requirement: text,
          context: contextInput.value.trim() || null,
          recipient_hint: recipientHint.value.trim() || null,
          tone_hint: toneHint.value ? toneHint.value.toLowerCase() : null,
          desired_length: lengthHint.value || "medium",
          language: languageHint.value || "English"
        };
        const aResp = await fetch(`${API_BASE}/analyze`, {
          method: "POST",
          headers: getAuthHeaders(),
          body: JSON.stringify(aPayload)
        });
        if (!aResp.ok) throw new Error("Semantic analysis failed");
        const aData = await aResp.json();
        state.representation = aData.representation;
        state.requestId = aData.request_id;
        populateSemanticsTab(aData);
      } else if (fromEditedRep && state.representation) {
        // Read edited values from Tab 2
        state.representation.intent = repIntentInput.value.trim();
        state.representation.recipient = repRecipientInput.value.trim() || null;
        state.representation.tone = repToneInput.value.trim();
        state.representation.topic = repTopicInput.value.trim() || null;
        state.representation.required_action = repActionInput.value.trim() || null;
        state.representation.deadline = repDeadlineInput.value.trim() || null;
        state.representation.reason = repReasonInput.value.trim() || null;
        state.representation.user_confirmed = true;
      }

      // Step B: Call /generate with Persona Memory
      setLoading(true, `Drafting personalized email for ${state.user.name}...`);
      const gPayload = {
        requirement: text,
        representation: state.representation,
        context: contextInput.value.trim() || null,
        desired_length: lengthHint.value || "medium",
        language: languageHint.value || "English",
        system_version: "proposed",
        request_id: state.requestId,
        sender_name: state.user.name,
        sender_role: state.user.job_title,
        sender_organization: state.user.organization,
        sender_sign_off: state.user.preferred_sign_off,
        sender_about_me: state.user.about_me
      };

      const gResp = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify(gPayload)
      });
      if (!gResp.ok) throw new Error("Email generation failed");
      const gData = await gResp.json();
      state.draft = gData;

      state.currentDraftId = gData.draft_id || null;
      populateDraftTab(gData);

      // Auto-refresh Drafts tab with newly generated email and prompt
      loadSavedDrafts();

      switchTab("tab-draft");
      if (gData.is_general_draft) {
        showToast("⚠️ Generated with general assumptions — prompt lacked specifics");
      } else {
        showToast(`Email drafted and saved to Drafts!`);
      }
    } catch (err) {
      showToast("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  function populateDraftTab(draft, validation = null) {
    draftSubject.value = draft.subject;
    draftBody.value = draft.body;
    updateWordCount();

    draftSignName.innerText = state.user.name || "User";
    draftModelTag.innerText = `Model: ${draft.model_identifier || 'gemini-3.5-flash-lite'}`;
    draftLatencyTag.innerText = `Latency: ${(draft.generation_time_ms || 0).toFixed(0)} ms`;

    // Generic Draft Notice Warning
    if (draftGeneralNotice) {
      const isGeneral = draft.is_general_draft || (draft.warnings && draft.warnings.some(w => w.toLowerCase().includes("specific") || w.toLowerCase().includes("general")));
      if (isGeneral) {
        draftGeneralNotice.classList.remove("hidden");
      } else {
        draftGeneralNotice.classList.add("hidden");
      }
    }

    if (draftValTag) {
      if (draft.validation_score !== undefined && draft.validation_score !== null) {
        const pct = Math.round(draft.validation_score * 100);
        draftValTag.style.display = "inline-flex";
        draftValTag.innerText = draft.passed_validation ? `🛡️ Quality: ${pct}% (≥80% Pass)` : `⚠️ Quality: ${pct}% (<80%)`;
        draftValTag.className = draft.passed_validation ? "meta-tag tag-pass" : "meta-tag tag-review";
      } else {
        draftValTag.style.display = "none";
      }
    }

    draftEmpty.classList.add("hidden");
    draftContent.classList.remove("hidden");
  }

  function updateWordCount() {
    const text = draftBody.value.trim();
    const count = text ? text.split(/\s+/).length : 0;
    draftWordCount.innerText = `${count} words`;
  }
  draftBody.addEventListener("input", updateWordCount);

  // --- COPY BUTTONS ---
  btnCopySubject.addEventListener("click", () => {
    copyText(draftSubject.value, "Subject copied to clipboard!");
  });

  btnCopyAll.addEventListener("click", () => {
    const full = `Subject: ${draftSubject.value}\n\n${draftBody.value}`;
    copyText(full, "Full email copied to clipboard!");
  });

  function copyText(text, msg) {
    navigator.clipboard.writeText(text).then(() => {
      showToast(msg);
    }).catch(() => {
      showToast("Failed to copy text.");
    });
  }

  // --- REDO / REFINE DRAFT WHEN USER IS NOT SATISFIED ---
  async function handleRedoDraft(forcedFeedback = null) {
    const prevSubject = draftSubject.value.trim();
    const prevBody = draftBody.value.trim();
    if (!prevSubject && !prevBody) {
      showToast("No draft available to redo. Generate an email first.");
      return;
    }

    const revisionFeedback = forcedFeedback || (redoFeedbackInput ? redoFeedbackInput.value.trim() : "") || "Generate an alternative phrasing variation.";
    const originalReq = reqInput.value.trim() || prevSubject;

    setLoading(true, `Redoing draft: "${revisionFeedback}"...`);
    try {
      const gPayload = {
        requirement: originalReq,
        representation: state.representation || {
          intent: "general_communication",
          tone: "polite and professional",
          recipient: null
        },
        context: contextInput ? contextInput.value.trim() || null : null,
        desired_length: lengthHint ? lengthHint.value || "medium" : "medium",
        language: languageHint ? languageHint.value || "English" : "English",
        system_version: "proposed",
        request_id: state.requestId || null,
        sender_name: state.user.name,
        sender_role: state.user.job_title,
        sender_organization: state.user.organization,
        sender_sign_off: state.user.preferred_sign_off,
        sender_about_me: state.user.about_me,
        revision_instruction: revisionFeedback,
        previous_draft: `Subject: ${prevSubject}\n\n${prevBody}`
      };

      const gResp = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify(gPayload)
      });
      if (!gResp.ok) {
        const errJson = await gResp.json().catch(() => ({}));
        throw new Error(errJson.detail || "Redo generation failed");
      }
      const gData = await gResp.json();
      state.draft = gData;
      state.currentDraftId = gData.draft_id || null;
      populateDraftTab(gData);
      loadSavedDrafts();

      if (redoFeedbackInput) {
        redoFeedbackInput.value = "";
      }
      showToast("✨ Draft redone successfully!");
    } catch (err) {
      showToast("Error redoing draft: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  if (btnRedoDraft) {
    btnRedoDraft.addEventListener("click", () => handleRedoDraft());
  }

  // Connect preset chips
  document.querySelectorAll(".redo-presets .btn-preset").forEach(btn => {
    btn.addEventListener("click", () => {
      const preset = btn.getAttribute("data-preset");
      if (redoFeedbackInput) {
        redoFeedbackInput.value = preset;
      }
      handleRedoDraft(preset);
    });
  });

  // --- INJECT DRAFT INTO GMAIL / OUTLOOK / ACTIVE WEBMAIL ---
  btnInjectWebmail.addEventListener("click", async () => {
    const subject = draftSubject.value.trim();
    const body = draftBody.value.trim();

    if (!body && !subject) {
      showToast("No draft to inject. Generate an email first.");
      return;
    }

    if (typeof chrome === "undefined" || !chrome.tabs || !chrome.tabs.query) {
      copyText(`Subject: ${subject}\n\n${body}`, "Copied! Paste directly into your email client.");
      return;
    }

    setLoading(true, "Injecting draft into active email composer...");
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab) {
        setLoading(false);
        return showToast("No active browser tab found.");
      }

      // Step 1: Ensure content script is running in tab
      if (chrome.scripting && chrome.scripting.executeScript) {
        try {
          await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            files: ["content.js"]
          });
        } catch (e) {
          // May be restricted page or already loaded
        }
      }

      // Step 2: Message content script
      chrome.tabs.sendMessage(tab.id, {
        action: "INJECT_EMAIL",
        data: { subject, body }
      }, async (res) => {
        setLoading(false);

        if (chrome.runtime.lastError || !res) {
          // Step 3: Direct fallback attempt via executeScript
          if (chrome.scripting && chrome.scripting.executeScript) {
            try {
              const execRes = await chrome.scripting.executeScript({
                target: { tabId: tab.id, allFrames: true },
                func: (sub, bdy) => {
                  if (typeof window.__semanticAssistantInject === "function") {
                    return window.__semanticAssistantInject(sub, bdy);
                  }
                  return { success: false };
                },
                args: [subject, body]
              });
              const anySuccess = execRes && execRes.some(r => r.result && r.result.success);
              if (anySuccess) {
                showToast("Inserted draft into Gmail/Outlook composer!");
                return;
              }
            } catch (fallbackErr) {}
          }

          copyText(`Subject: ${subject}\n\n${body}`, "No active compose box found. Email copied to clipboard! (Open 'Compose' in Gmail to auto-inject)");
          return;
        }

        if (res.success) {
          showToast(res.message || "Inserted draft into email composer!");
        } else {
          copyText(`Subject: ${subject}\n\n${body}`, res.message || "Draft copied to clipboard! Please open or click inside email composer.");
        }
      });
    } catch (e) {
      setLoading(false);
      copyText(`Subject: ${subject}\n\n${body}`, "Draft copied to clipboard!");
    }
  });

  if (btnViewValidation) {
    btnViewValidation.addEventListener("click", () => {
      switchTab("tab-saved-drafts");
    });
  }

  // --- POPULATE VALIDATION TAB ---
  function populateValidationTab(report) {
    const status = report.overall_status;
    valStatusPill.innerText = status;
    if (status === "Valid" || status === "PASSED") {
      valStatusPill.className = "status-badge-lg badge-success";
    } else if (status === "NEEDS_REVIEW" || status === "WARNING") {
      valStatusPill.className = "status-badge-lg badge-warning";
    } else {
      valStatusPill.className = "status-badge-lg badge-danger";
    }

    // Intent
    metricIntent.innerText = report.intent_preserved ? "Preserved" : "Altered";
    metricIntent.className = report.intent_preserved ? "metric-value text-success" : "metric-value text-danger";
    metricIntentDesc.innerText = report.intent_explanation || "Primary intent evaluated";

    // Tone
    metricTone.innerText = report.tone_suitable ? "Compliant" : "Mismatched";
    metricTone.className = report.tone_suitable ? "metric-value text-success" : "metric-value text-warning";
    metricToneDesc.innerText = report.tone_explanation || `Detected: ${report.tone_detected || 'Formal'}`;

    // Completeness
    const compPct = Math.round((report.completeness_score || 0) * 100);
    metricCompleteness.innerText = `${compPct}%`;
    metricCompletenessBar.style.width = `${compPct}%`;

    // Similarity
    const simVal = (report.semantic_similarity || 0).toFixed(2);
    metricSimilarity.innerText = simVal;
    metricSimilarityBar.style.width = `${Math.min(100, Math.round(report.semantic_similarity * 100))}%`;

    // Missing Items
    if (report.missing_details && report.missing_details.length > 0) {
      valMissingList.innerHTML = report.missing_details.map(m => `<li>${m}</li>`).join("");
      valMissingCard.classList.remove("hidden");
    } else {
      valMissingCard.classList.add("hidden");
    }

    // Unsupported Details
    if (report.unsupported_details && report.unsupported_details.length > 0) {
      valUnsupportedList.innerHTML = report.unsupported_details.map(u => `<li>${u}</li>`).join("");
      valUnsupportedCard.classList.remove("hidden");
    } else {
      valUnsupportedCard.classList.add("hidden");
    }

    // Actionable Suggestions
    if (report.actionable_warnings && report.actionable_warnings.length > 0) {
      valSuggestionsList.innerHTML = report.actionable_warnings.map(w => `<li>${w}</li>`).join("");
    } else {
      valSuggestionsList.innerHTML = "<li>Draft satisfies all factual, semantic, and register criteria.</li>";
    }

    valEmpty.classList.add("hidden");
    valContent.classList.remove("hidden");
  }

  // --- TAB: SAVED DRAFTS & PROMPTS CONTROLS ---

  btnRefreshDrafts.addEventListener("click", () => {
    loadSavedDrafts(true);
  });

  if (btnSaveCurrentDraft) {
    btnSaveCurrentDraft.addEventListener("click", () => {
      saveCurrentDraft();
    });
  }

  if (btnViewAllDrafts) {
    btnViewAllDrafts.addEventListener("click", () => {
      switchTab("tab-saved-drafts");
      loadSavedDrafts();
    });
  }

  draftsSearchInput.addEventListener("input", (e) => {
    filterDrafts(e.target.value);
  });

  async function loadSavedDrafts(showToastNotice = false) {
    try {
      const resp = await fetch(`${API_BASE}/drafts`, {
        method: "GET",
        headers: getAuthHeaders()
      });
      if (resp.ok) {
        const data = await resp.json();
        state.savedDrafts = data.drafts || [];
        localStorage.setItem("assistant_local_drafts", JSON.stringify(state.savedDrafts));
      } else if (resp.status === 401) {
        state.savedDrafts = [];
        localStorage.removeItem("assistant_local_drafts");
      } else {
        throw new Error("HTTP " + resp.status);
      }
    } catch (err) {
      console.warn("Could not fetch drafts from server, using local cache:", err);
      const cached = localStorage.getItem("assistant_local_drafts");
      if (cached) {
        try { state.savedDrafts = JSON.parse(cached); } catch (e) {}
      }
    }

    updateDraftsCounter();
    renderDrafts(state.savedDrafts);

    if (showToastNotice) {
      showToast(`Loaded ${state.savedDrafts.length} drafts.`);
    }
  }

  function updateDraftsCounter() {
    const count = state.savedDrafts.length;
    draftsCountSummary.innerText = `${count} draft${count === 1 ? '' : 's'} saved`;
    if (draftsCountBadge) {
      if (count > 0) {
        draftsCountBadge.innerText = count;
        draftsCountBadge.classList.remove("hidden");
      } else {
        draftsCountBadge.classList.add("hidden");
      }
    }
  }

  function filterDrafts(query) {
    const q = (query || "").trim().toLowerCase();
    if (!q) {
      renderDrafts(state.savedDrafts);
      return;
    }

    const filtered = state.savedDrafts.filter(d => {
      const p = (d.prompt || "").toLowerCase();
      const s = (d.subject || "").toLowerCase();
      const b = (d.body || "").toLowerCase();
      const r = (d.recipient || "").toLowerCase();
      return p.includes(q) || s.includes(q) || b.includes(q) || r.includes(q);
    });

    renderDrafts(filtered);
  }

  function renderDrafts(drafts) {
    draftsListContainer.innerHTML = "";

    if (!drafts || drafts.length === 0) {
      draftsEmpty.classList.remove("hidden");
      draftsListContainer.classList.add("hidden");
      return;
    }

    draftsEmpty.classList.add("hidden");
    draftsListContainer.classList.remove("hidden");

    drafts.forEach(draft => {
      const card = document.createElement("div");
      card.className = "draft-card";

      // Date formatting
      let dateStr = "Recent";
      if (draft.created_at) {
        try {
          const d = new Date(draft.created_at);
          dateStr = d.toLocaleDateString(undefined, {
            month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
          });
        } catch (e) {}
      }

      // Metadata tags
      let metaHtml = "";
      if (draft.intent) {
        metaHtml += `<span class="badge badge-indigo">🎯 ${escapeHtml(draft.intent)}</span>`;
      }
      if (draft.tone) {
        metaHtml += `<span class="badge badge-purple">🎭 ${escapeHtml(draft.tone)}</span>`;
      }
      if (draft.recipient) {
        metaHtml += `<span class="badge badge-neutral">👤 To: ${escapeHtml(draft.recipient)}</span>`;
      }

      card.innerHTML = `
        <div class="draft-card-header">
          <div class="draft-subject-title">✉️ ${escapeHtml(draft.subject || 'Untitled Draft')}</div>
          <span class="draft-time-pill">${dateStr}</span>
        </div>

        <div class="draft-prompt-box">
          <div class="draft-prompt-header">💬 User Prompt:</div>
          <div class="draft-prompt-text">${escapeHtml(draft.prompt || '(No prompt recorded)')}</div>
        </div>

        <div class="draft-body-preview">${escapeHtml(draft.body || '')}</div>

        ${metaHtml ? `<div class="draft-meta-tags">${metaHtml}</div>` : ''}

        <div class="draft-card-actions">
          <button class="btn-card-action btn-card-primary btn-load-draft" title="Load draft and prompt into composer">
            ✏️ Load
          </button>
          <button class="btn-card-action btn-card-inject btn-inject-draft" title="Insert into active Gmail / Outlook tab">
            📨 Webmail
          </button>
          <button class="btn-card-action btn-card-secondary btn-copy-draft" title="Copy subject and body">
            📋 Copy
          </button>
          <button class="btn-card-action btn-card-danger btn-delete-draft" title="Delete draft from database">
            🗑️ Delete
          </button>
        </div>
      `;

      // Attach button handlers
      card.querySelector(".btn-load-draft").addEventListener("click", () => {
        loadDraftIntoEditor(draft);
      });

      card.querySelector(".btn-inject-draft").addEventListener("click", () => {
        injectDraftDirectly(draft);
      });

      card.querySelector(".btn-copy-draft").addEventListener("click", () => {
        const full = `Subject: ${draft.subject}\n\n${draft.body}`;
        copyText(full, "Draft copied to clipboard!");
      });

      card.querySelector(".btn-delete-draft").addEventListener("click", () => {
        deleteDraft(draft.id);
      });

      draftsListContainer.appendChild(card);
    });
  }

  function loadDraftIntoEditor(draft) {
    draftSubject.value = draft.subject || "";
    draftBody.value = draft.body || "";
    updateWordCount();

    if (draft.prompt && reqInput) {
      reqInput.value = draft.prompt;
    }

    state.currentDraftId = draft.id;
    draftEmpty.classList.add("hidden");
    draftContent.classList.remove("hidden");

    switchTab("tab-draft");
    showToast("Loaded draft and prompt into editor!");
  }

  async function deleteDraft(draftId) {
    if (!draftId) return;

    try {
      const resp = await fetch(`${API_BASE}/drafts/${draftId}`, {
        method: "DELETE",
        headers: getAuthHeaders()
      });
      if (!resp.ok && resp.status !== 404) {
        throw new Error("HTTP " + resp.status);
      }

      state.savedDrafts = state.savedDrafts.filter(d => d.id !== draftId);
      localStorage.setItem("assistant_local_drafts", JSON.stringify(state.savedDrafts));
      updateDraftsCounter();
      renderDrafts(state.savedDrafts);
      showToast("Draft deleted from database.");
    } catch (e) {
      showToast("Could not delete draft: " + e.message);
    }
  }

  async function saveCurrentDraft() {
    const subject = draftSubject.value.trim();
    const body = draftBody.value.trim();
    const prompt = reqInput.value.trim() || subject || "Custom draft";

    if (!body && !subject) {
      return showToast("No draft content to save. Generate or write an email first.");
    }

    setLoading(true, "Saving draft & prompt to database...");
    try {
      const payload = {
        prompt: prompt,
        subject: subject || "Untitled Draft",
        body: body,
        intent: state.representation ? state.representation.intent : null,
        tone: state.representation ? state.representation.tone : null,
        recipient: state.representation ? state.representation.recipient : (recipientHint ? recipientHint.value.trim() : null),
        user_id: state.user.id || null,
        user_email: state.user.email || null,
        draft_id: state.currentDraftId || null
      };

      const resp = await fetch(`${API_BASE}/drafts`, {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify(payload)
      });

      if (!resp.ok) throw new Error("Server error " + resp.status);
      const saved = await resp.json();
      state.currentDraftId = saved.id;

      await loadSavedDrafts();
      showToast("Draft and prompt saved to database!");
    } catch (err) {
      // Local fallback
      const localDraft = {
        id: "local-" + Date.now(),
        prompt: prompt,
        subject: subject || "Untitled Draft",
        body: body,
        intent: state.representation ? state.representation.intent : null,
        tone: state.representation ? state.representation.tone : null,
        created_at: new Date().toISOString()
      };
      state.savedDrafts.unshift(localDraft);
      localStorage.setItem("assistant_local_drafts", JSON.stringify(state.savedDrafts));
      updateDraftsCounter();
      renderDrafts(state.savedDrafts);
      showToast("Draft saved locally (offline mode).");
    } finally {
      setLoading(false);
    }
  }

  function injectDraftDirectly(draft) {
    const subject = (draft.subject || "").trim();
    const body = (draft.body || "").trim();

    if (typeof chrome === "undefined" || !chrome.tabs || !chrome.tabs.query) {
      copyText(`Subject: ${subject}\n\n${body}`, "Copied! Paste into your webmail composer.");
      return;
    }

    setLoading(true, "Injecting draft into composer...");
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      const tab = tabs && tabs[0];
      if (!tab) {
        setLoading(false);
        copyText(`Subject: ${subject}\n\n${body}`, "No active tab. Email copied!");
        return;
      }

      // Ensure content script loaded
      if (chrome.scripting && chrome.scripting.executeScript) {
        try {
          await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            files: ["content.js"]
          });
        } catch (e) {}
      }

      chrome.tabs.sendMessage(tab.id, {
        action: "INJECT_EMAIL",
        data: { subject, body }
      }, async (res) => {
        setLoading(false);

        if (chrome.runtime.lastError || !res) {
          // Fallback via executeScript directly
          if (chrome.scripting && chrome.scripting.executeScript) {
            try {
              const execRes = await chrome.scripting.executeScript({
                target: { tabId: tab.id, allFrames: true },
                func: (sub, bdy) => {
                  if (typeof window.__semanticAssistantInject === "function") {
                    return window.__semanticAssistantInject(sub, bdy);
                  }
                  return { success: false };
                },
                args: [subject, body]
              });
              if (execRes && execRes.some(r => r.result && r.result.success)) {
                showToast("Inserted draft into email composer!");
                return;
              }
            } catch (e) {}
          }

          copyText(`Subject: ${subject}\n\n${body}`, "No compose box detected. Email copied! (Open 'Compose' in Gmail to auto-inject)");
          return;
        }

        if (res.success) {
          showToast(res.message || "Inserted draft into active webmail!");
        } else {
          copyText(`Subject: ${subject}\n\n${body}`, res.message || "Copied! Please click inside your compose box.");
        }
      });
    });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // --- HELPERS ---
  function setLoading(isLoading, text = "Processing...") {
    if (isLoading) {
      globalSpinnerText.innerText = text;
      globalSpinner.classList.remove("hidden");
    } else {
      globalSpinner.classList.add("hidden");
    }
  }

  function showToast(message) {
    appToast.innerText = message;
    appToast.classList.remove("hidden");
    setTimeout(() => {
      appToast.classList.add("hidden");
    }, 2800);
  }
});
