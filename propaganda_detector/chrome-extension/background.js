// background.js – Service worker for the Propaganda Detector Chrome extension.

const DEFAULT_SETTINGS = {
  apiUrl: "http://localhost:5050",   // must match server.py port
  modelType: "transformer",
  threshold: 0.5,
};

// ---------------------------------------------------------------------------
// Context menu
// ---------------------------------------------------------------------------

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "check-bias",
    title: 'Check for Bias: "%s"',
    contexts: ["selection"],
  });
});

// ---------------------------------------------------------------------------
// Keyboard shortcut
// ---------------------------------------------------------------------------

chrome.commands.onCommand.addListener(async (command) => {
  if (command !== "check-bias") return;

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;

  // Inject content script first in case it hasn't loaded yet
  await ensureContentScript(tab.id);

  chrome.tabs.sendMessage(tab.id, { action: "getSelection" }, (response) => {
    if (chrome.runtime.lastError || !response?.text) return;
    analyzeAndRespond(tab.id, response.text);
  });
});

// ---------------------------------------------------------------------------
// Context menu click
// ---------------------------------------------------------------------------

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "check-bias" || !info.selectionText || !tab?.id) return;

  await ensureContentScript(tab.id);
  analyzeAndRespond(tab.id, info.selectionText);
});

// ---------------------------------------------------------------------------
// Core: call API → send result to content script
// ---------------------------------------------------------------------------

async function analyzeAndRespond(tabId, text) {
  // Show loading panel immediately
  chrome.tabs.sendMessage(tabId, { action: "showLoading" });

  const settings = await getSettings();

  try {
    const response = await fetch(`${settings.apiUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text:       text,
        model_type: settings.modelType,
        threshold:  settings.threshold,
      }),
    });

    if (!response.ok) {
      const errBody = await response.text();
      throw new Error(`API error ${response.status}: ${errBody}`);
    }

    const result = await response.json();

    if (result.error) {
      throw new Error(result.error);
    }

    // Attach the original text so the panel can show a preview
    result.text = text;

    chrome.tabs.sendMessage(tabId, { action: "showResult", result });

  } catch (err) {
    let message = err.message || "Failed to reach the API.";

    // Give a clearer message for connection errors
    if (message.includes("Failed to fetch") || message.includes("NetworkError")) {
      message = "Cannot connect to the API. Make sure server.py is running on port 5050.";
    }

    chrome.tabs.sendMessage(tabId, { action: "showError", error: message });
  }
}

// ---------------------------------------------------------------------------
// Ensure content script is injected (handles navigated/refreshed tabs)
// ---------------------------------------------------------------------------

async function ensureContentScript(tabId) {
  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ["content.js"],
    });
  } catch (_) {
    // Already injected or page doesn't allow injection — safe to ignore
  }
}

// ---------------------------------------------------------------------------
// Settings helpers
// ---------------------------------------------------------------------------

async function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(DEFAULT_SETTINGS, (items) => resolve(items));
  });
}
