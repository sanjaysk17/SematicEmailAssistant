/**
 * browser_extension/background.js
 * Service worker for context menu, badge notifications, and state handling.
 */

chrome.runtime.onInstalled.addListener(() => {
  // Create Context Menu for selected text
  chrome.contextMenus.create({
    id: "semantic_assist_selection",
    title: "Draft email using Semantic Assistant...",
    contexts: ["selection"]
  });
  console.log("Semantic Email Assistant Extension installed.");
});

// Context Menu click handler
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "semantic_assist_selection" && info.selectionText) {
    chrome.storage.local.set({
      pending_requirement: info.selectionText,
      pending_source_url: tab?.url || ""
    }, () => {
      chrome.action.setBadgeText({ text: "1" });
      chrome.action.setBadgeBackgroundColor({ color: "#2563eb" });
    });
  }
});

// Clear badge when action popup is opened
chrome.runtime.onConnect.addListener((port) => {
  chrome.action.setBadgeText({ text: "" });
});
