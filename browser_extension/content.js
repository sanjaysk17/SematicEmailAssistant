/**
 * browser_extension/content.js
 * Content script running in webmail & browser tabs.
 * Enables context extraction and direct one-click email injection into Gmail, Outlook, etc.
 */

(() => {
  // Prevent duplicate listener registration
  if (window.__semanticAssistantContentScriptLoaded) return;
  window.__semanticAssistantContentScriptLoaded = true;

  // Listen for messages from popup or background
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "EXTRACT_PAGE_CONTEXT") {
      extractPageContext().then(res => sendResponse(res));
      return true; // Keep message channel open for async response
    }

    if (request.action === "INJECT_EMAIL") {
      const { subject, body } = request.data || {};
      injectIntoActiveComposerAsync(subject, body).then(res => sendResponse(res));
      return true; // Keep message channel open for async response
    }
  });

  /**
   * Comprehensive text extraction:
   * 1. Highlighted selection on page
   * 2. Active input/textarea selection
   * 3. Iframe selections
   * 4. Gmail / Outlook active email thread body fallback
   */
  async function extractPageContext() {
    const pageTitle = document.title || "";
    const url = window.location.href || "";

    // 1. Primary window selection
    let selectedText = window.getSelection ? window.getSelection().toString().trim() : "";

    // 2. Active input or textarea selection
    if (!selectedText && document.activeElement) {
      const el = document.activeElement;
      if ((el.tagName === "TEXTAREA" || el.tagName === "INPUT") && typeof el.selectionStart === "number") {
        selectedText = el.value.substring(el.selectionStart, el.selectionEnd).trim();
      }
    }

    // 3. Search child iframes if selection is inside an iframe
    if (!selectedText) {
      const iframes = document.querySelectorAll("iframe");
      for (const frame of iframes) {
        try {
          const fDoc = frame.contentDocument || frame.contentWindow?.document;
          if (fDoc && fDoc.getSelection) {
            const s = fDoc.getSelection().toString().trim();
            if (s) {
              selectedText = s;
              break;
            }
          }
        } catch (e) {
          // Cross-origin iframe security restriction, skip
        }
      }
    }

    // 4. Fallback: If inside Gmail or Outlook and no text highlighted, grab thread context
    if (!selectedText) {
      if (url.includes("mail.google.com")) {
        const emailBodies = document.querySelectorAll(".a3s.aiL, div[dir='ltr'], .ii.gt, .adn.ads");
        if (emailBodies.length > 0) {
          const lastEmail = emailBodies[emailBodies.length - 1];
          selectedText = lastEmail.innerText.trim().substring(0, 2000);
        }
      } else if (url.includes("outlook.live.com") || url.includes("outlook.office.com") || url.includes("outlook.office365.com")) {
        const readPanes = document.querySelectorAll("div[aria-label='Message body'], .ItemPartBody, div[role='document']");
        if (readPanes.length > 0) {
          selectedText = readPanes[readPanes.length - 1].innerText.trim().substring(0, 2000);
        }
      }
    }

    return {
      success: true,
      selectedText: selectedText,
      pageTitle: pageTitle,
      url: url
    };
  }

  /**
   * Helper: checks if an element is rendered and visible on screen
   */
  function isElementVisible(el) {
    if (!el) return false;
    return !!(el.offsetParent !== null || el.offsetWidth > 0 || el.offsetHeight > 0 || (el.getClientRects && el.getClientRects().length > 0));
  }

  /**
   * Finds Gmail compose components:
   * Returns { subjectEl, bodyEl }
   */
  function findGmailComposeElements() {
    const subjectSelectors = [
      'input[name="subjectbox"]',
      'input[aria-label="Subject"]',
      'input[name="subject"]',
      'input[placeholder="Subject"]',
      'input.aoT',
      'input[id*="subject" i]'
    ];

    const bodySelectors = [
      'div[role="textbox"][aria-label*="Message Body"]',
      'div[role="textbox"][aria-label*="Message body"]',
      'div[role="textbox"][aria-label*="Body" i]',
      'div[aria-label="Message Body"]',
      'div[aria-label="Message body"]',
      'div.Am.Al.editable[role="textbox"]',
      'div.Am.Al.editable',
      'div.editable[role="textbox"]',
      'div[g_editable="true"][role="textbox"]',
      'div[g_editable="true"]',
      'div[contenteditable="true"][role="textbox"]',
      'div[contenteditable="true"][aria-label*="Body" i]',
      'div[contenteditable="true"][aria-label*="Message" i]'
    ];

    // Search main document first
    let subjectEl = null;
    let bodyEl = null;

    for (const sel of subjectSelectors) {
      const candidates = Array.from(document.querySelectorAll(sel));
      const visible = candidates.reverse().find(isElementVisible);
      if (visible) { subjectEl = visible; break; }
    }

    for (const sel of bodySelectors) {
      const candidates = Array.from(document.querySelectorAll(sel));
      const visible = candidates.reverse().find(isElementVisible);
      if (visible) { bodyEl = visible; break; }
    }

    // If not found in main document, search iframes
    if (!bodyEl) {
      const iframes = document.querySelectorAll("iframe");
      for (const frame of iframes) {
        try {
          const fDoc = frame.contentDocument || frame.contentWindow?.document;
          if (!fDoc) continue;
          for (const sel of bodySelectors) {
            const b = fDoc.querySelector(sel);
            if (isElementVisible(b)) {
              bodyEl = b;
              if (!subjectEl) {
                for (const sSel of subjectSelectors) {
                  const s = fDoc.querySelector(sSel);
                  if (isElementVisible(s)) { subjectEl = s; break; }
                }
              }
              break;
            }
          }
        } catch (e) {}
      }
    }

    return { subjectEl, bodyEl };
  }

  /**
   * Finds Outlook web compose components
   */
  function findOutlookComposeElements() {
    const subjectSelectors = [
      'input[aria-label="Add a subject"]',
      'input[placeholder*="subject" i]',
      'input[aria-label*="subject" i]',
      'input[id*="subject" i]'
    ];

    const bodySelectors = [
      'div[role="textbox"][aria-label*="Message body" i]',
      'div[aria-label*="Message body" i]',
      'div[role="textbox"][aria-label*="Body" i]',
      'div[aria-label*="Message" i][role="textbox"]',
      'div[contenteditable="true"][role="textbox"]'
    ];

    let subjectEl = null;
    let bodyEl = null;

    for (const sel of subjectSelectors) {
      const candidates = Array.from(document.querySelectorAll(sel));
      const visible = candidates.reverse().find(isElementVisible);
      if (visible) { subjectEl = visible; break; }
    }

    for (const sel of bodySelectors) {
      const candidates = Array.from(document.querySelectorAll(sel));
      const visible = candidates.reverse().find(isElementVisible);
      if (visible) { bodyEl = visible; break; }
    }

    return { subjectEl, bodyEl };
  }

  /**
   * Tries clicking Gmail Compose button if no compose box is open
   */
  function triggerGmailComposeOpen() {
    const composeButtons = [
      'div[role="button"][gh="cm"]',
      'div.T-I.T-I-KE.L3',
      'div[aria-label*="Compose" i][role="button"]',
      'div[data-tooltip*="Compose" i]',
      '.z0 > .L3'
    ];
    for (const sel of composeButtons) {
      const btn = document.querySelector(sel);
      if (isElementVisible(btn)) {
        btn.click();
        return true;
      }
    }
    return false;
  }

  /**
   * Injects text into a rich contenteditable div with full event notifications
   */
  function injectIntoContentEditable(targetEl, text) {
    if (!targetEl || !text) return false;

    targetEl.focus();

    // Convert newlines into standard HTML paragraph/div structure
    const lines = text.split("\n");
    const htmlContent = lines.map(line => {
      const escaped = line.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      return escaped.trim() === "" ? "<div><br></div>" : `<div>${escaped}</div>`;
    }).join("");

    // 1. Try document.execCommand first
    let execSuccess = false;
    try {
      const sel = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(targetEl);
      sel.removeAllRanges();
      sel.addRange(range);
      execSuccess = document.execCommand("insertHTML", false, htmlContent);
    } catch (e) {
      execSuccess = false;
    }

    // 2. Direct innerHTML fallback if execCommand was refused
    if (!execSuccess || !targetEl.innerText.trim()) {
      targetEl.innerHTML = htmlContent;
    }

    // 3. Move cursor to end of text
    try {
      const sel = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(targetEl);
      range.collapse(false);
      sel.removeAllRanges();
      sel.addRange(range);
    } catch (e) {}

    // 4. Dispatch rich input events for frameworks (Closure, React, Angular)
    targetEl.dispatchEvent(new Event("focus", { bubbles: true }));
    try {
      targetEl.dispatchEvent(new InputEvent("input", {
        bubbles: true,
        cancelable: true,
        inputType: "insertText",
        data: text
      }));
    } catch (e) {
      targetEl.dispatchEvent(new Event("input", { bubbles: true }));
    }
    targetEl.dispatchEvent(new Event("change", { bubbles: true }));
    targetEl.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, key: " " }));
    targetEl.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true, key: " " }));

    return true;
  }

  /**
   * Injects text into standard input field with event notifications
   */
  function injectIntoInput(inputEl, text) {
    if (!inputEl || !text) return false;
    inputEl.focus();
    inputEl.value = text;
    inputEl.dispatchEvent(new Event("focus", { bubbles: true }));
    try {
      inputEl.dispatchEvent(new InputEvent("input", {
        bubbles: true,
        cancelable: true,
        inputType: "insertText",
        data: text
      }));
    } catch (e) {
      inputEl.dispatchEvent(new Event("input", { bubbles: true }));
    }
    inputEl.dispatchEvent(new Event("change", { bubbles: true }));
    inputEl.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true }));
    return true;
  }

  /**
   * Asynchronous injection coordinator with auto-open and retry
   */
  async function injectIntoActiveComposerAsync(subject, body) {
    let subjectInjected = false;
    let bodyInjected = false;
    const url = window.location.href || "";

    // Step 1: Detect Gmail
    if (url.includes("mail.google.com")) {
      let { subjectEl, bodyEl } = findGmailComposeElements();

      // If no compose box is currently open, trigger the Compose button automatically
      if (!bodyEl) {
        const clicked = triggerGmailComposeOpen();
        if (clicked) {
          // Wait 650ms for Gmail animation and DOM mount
          await new Promise(resolve => setTimeout(resolve, 650));
          const retry = findGmailComposeElements();
          subjectEl = retry.subjectEl;
          bodyEl = retry.bodyEl;
        }
      }

      if (subjectEl && subject) {
        subjectInjected = injectIntoInput(subjectEl, subject);
      }
      if (bodyEl && body) {
        bodyInjected = injectIntoContentEditable(bodyEl, body);
      }
    }

    // Step 2: Detect Outlook Web
    if (!bodyInjected && (url.includes("outlook.live.com") || url.includes("outlook.office.com") || url.includes("outlook.office365.com"))) {
      const { subjectEl, bodyEl } = findOutlookComposeElements();
      if (subjectEl && subject) {
        subjectInjected = injectIntoInput(subjectEl, subject);
      }
      if (bodyEl && body) {
        bodyInjected = injectIntoContentEditable(bodyEl, body);
      }
    }

    // Step 3: Generic fallback to currently active element
    if (!bodyInjected) {
      const active = document.activeElement;
      if (active && (active.tagName === "TEXTAREA" || active.tagName === "INPUT" || active.isContentEditable)) {
        if (active.isContentEditable) {
          bodyInjected = injectIntoContentEditable(active, body);
        } else {
          active.value = body;
          active.dispatchEvent(new Event("input", { bubbles: true }));
          bodyInjected = true;
        }
      }
    }

    const success = subjectInjected || bodyInjected;
    showInjectionToast(success);

    return {
      success: success,
      subjectInjected: subjectInjected,
      bodyInjected: bodyInjected,
      message: success 
        ? "Email draft inserted into active composer!" 
        : "No active email compose box found. Email copied to clipboard."
    };
  }

  function showInjectionToast(success) {
    const existing = document.getElementById("semantic-assistant-injection-toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.id = "semantic-assistant-injection-toast";
    toast.style.cssText = `
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 2147483647;
      background: ${success ? "#10b981" : "#f59e0b"};
      color: #ffffff;
      padding: 12px 20px;
      border-radius: 8px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px;
      font-weight: 600;
      box-shadow: 0 10px 25px rgba(0,0,0,0.25);
      transition: all 0.3s ease;
      opacity: 0;
      transform: translateY(12px);
      pointer-events: none;
    `;
    toast.innerText = success 
      ? "✨ Semantic Assistant: Draft inserted into email composer!" 
      : "⚠️ Note: Open or click an email compose window to inject.";
    document.body.appendChild(toast);

    requestAnimationFrame(() => {
      toast.style.opacity = "1";
      toast.style.transform = "translateY(0)";
    });

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(12px)";
      setTimeout(() => toast.remove(), 300);
    }, 3800);
  }

  // Expose global methods for direct programmatic invocation via chrome.scripting.executeScript
  window.__semanticAssistantExtract = extractPageContext;
  window.__semanticAssistantInject = injectIntoActiveComposerAsync;
})();
