// background.js – Service worker for the Propaganda Detector Chrome extension.
//
// Responsibilities:
//   1. Create the right-click context menu item
//   2. Send selected text to the API
//   3. Forward results to the content script for rendering

const DEFAULT_SETTINGS = {
  apiUrl: "http://localhost:8000",
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
  if (command === "check-bias") {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) return;

    // Ask content script for the current selection
    chrome.tabs.sendMessage(tab.id, { action: "getSelection" }, async (response) => {
      if (chrome.runtime.lastError || !response?.text) return;
      await analyzeAndRespond(tab.id, response.text);
    });
  }
});

// ---------------------------------------------------------------------------
// Context menu click handler
// ---------------------------------------------------------------------------

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "check-bias" || !info.selectionText) return;
  await analyzeAndRespond(tab.id, info.selectionText);
});

// ---------------------------------------------------------------------------
// Core: call API and send result to content script
// ---------------------------------------------------------------------------

async function analyzeAndRespond(tabId, text) {
  // Tell content script to show loading state
  chrome.tabs.sendMessage(tabId, { action: "showLoading" });

  const settings = await getSettings();

  try {
    const response = await fetch(`${settings.apiUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        model_type: settings.modelType,
        threshold: settings.threshold,
      }),
    });

    if (!response.ok) {
      const errBody = await response.text();
      throw new Error(`API returned ${response.status}: ${errBody}`);
    }

    const result = await response.json();
    chrome.tabs.sendMessage(tabId, { action: "showResult", result });
  } catch (err) {
    chrome.tabs.sendMessage(tabId, {
      action: "showError",
      error: err.message || "Failed to reach the API.",
    });
  }
}

// ---------------------------------------------------------------------------
// Settings helpers
// ---------------------------------------------------------------------------

async function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(DEFAULT_SETTINGS, (items) => {
      resolve(items);
    });
  });
}
