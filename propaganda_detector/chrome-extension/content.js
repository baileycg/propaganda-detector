// content.js – Injected into every page.
//
// Responsibilities:
//   1. Reply with the current text selection when asked
//   2. Render a floating result panel near the selected text

(() => {
  // Guard against double-injection
  if (window.__biasDetectorLoaded) return;
  window.__biasDetectorLoaded = true;

  const PANEL_ID = "bias-detector-panel";

  // -----------------------------------------------------------------------
  // Message listener
  // -----------------------------------------------------------------------

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    switch (msg.action) {
      case "getSelection":
        sendResponse({ text: window.getSelection()?.toString()?.trim() || "" });
        break;
      case "showLoading":
        showPanel(loadingHTML());
        break;
      case "showResult":
        showPanel(resultHTML(msg.result));
        break;
      case "showError":
        showPanel(errorHTML(msg.error));
        break;
    }
  });

  // -----------------------------------------------------------------------
  // Panel positioning
  // -----------------------------------------------------------------------

  function getSelectionCoords() {
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return null;
    const range = sel.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    return {
      top: rect.bottom + window.scrollY + 8,
      left: Math.max(8, rect.left + window.scrollX),
    };
  }

  // -----------------------------------------------------------------------
  // Show / hide panel
  // -----------------------------------------------------------------------

  function showPanel(innerHTML) {
    removePanel();

    const panel = document.createElement("div");
    panel.id = PANEL_ID;
    panel.innerHTML = innerHTML;

    // Position near selection
    const coords = getSelectionCoords();
    if (coords) {
      panel.style.top = `${coords.top}px`;
      panel.style.left = `${coords.left}px`;
    }

    // Close button handler
    panel.addEventListener("click", (e) => {
      if (e.target.classList.contains("bdp-close")) removePanel();
    });

    document.body.appendChild(panel);

    // Dismiss on outside click
    setTimeout(() => {
      document.addEventListener("mousedown", outsideClick);
    }, 100);

    // Keep panel in viewport
    requestAnimationFrame(() => {
      const panelRect = panel.getBoundingClientRect();
      if (panelRect.right > window.innerWidth - 8) {
        panel.style.left = `${window.innerWidth - panelRect.width - 16 + window.scrollX}px`;
      }
    });
  }

  function removePanel() {
    document.getElementById(PANEL_ID)?.remove();
    document.removeEventListener("mousedown", outsideClick);
  }

  function outsideClick(e) {
    const panel = document.getElementById(PANEL_ID);
    if (panel && !panel.contains(e.target)) removePanel();
  }

  // -----------------------------------------------------------------------
  // HTML builders
  // -----------------------------------------------------------------------

  function loadingHTML() {
    return `
      <div class="bdp-header">
        <span class="bdp-title">Analyzing…</span>
        <button class="bdp-close" title="Close">✕</button>
      </div>
      <div class="bdp-body">
        <div class="bdp-spinner"></div>
        <p class="bdp-loading-text">Checking for bias…</p>
      </div>
    `;
  }

  function errorHTML(error) {
    return `
      <div class="bdp-header bdp-header-error">
        <span class="bdp-title">Error</span>
        <button class="bdp-close" title="Close">✕</button>
      </div>
      <div class="bdp-body">
        <p class="bdp-error-text">${escapeHTML(error)}</p>
        <p class="bdp-hint">Make sure the API server is running at the configured URL.</p>
      </div>
    `;
  }

  function resultHTML(r) {
    const isBiased = r.label === "Propaganda / Biased";
    const labelClass = isBiased ? "bdp-label-biased" : "bdp-label-clean";
    const headerClass = isBiased ? "bdp-header-biased" : "bdp-header-clean";
    const pct = (r.confidence * 100).toFixed(1);
    const barWidth = Math.round(r.confidence * 100);

    // Truncate displayed text
    const displayText =
      r.text.length > 120 ? r.text.slice(0, 120) + "…" : r.text;

    // Triggered words
    const wordsHTML = r.triggered_words?.length
      ? `<div class="bdp-section">
           <div class="bdp-section-title">Triggered Words</div>
           <div class="bdp-words">${r.triggered_words.map((w) => `<span class="bdp-word">${escapeHTML(w)}</span>`).join("")}</div>
         </div>`
      : "";

    // Signal summary
    let signalsHTML = "";
    if (r.signal_summary && Object.keys(r.signal_summary).length) {
      const rows = Object.entries(r.signal_summary)
        .map(
          ([k, v]) =>
            `<tr><td class="bdp-sig-name">${escapeHTML(formatSignalName(k))}</td><td class="bdp-sig-val">${Number(v).toFixed(4)}</td></tr>`
        )
        .join("");
      signalsHTML = `
        <div class="bdp-section">
          <div class="bdp-section-title">Linguistic Signals</div>
          <table class="bdp-signals">${rows}</table>
        </div>`;
    }

    // Emotions
    let emotionsHTML = "";
    if (r.emotions && Object.keys(r.emotions).length) {
      const sorted = Object.entries(r.emotions)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);
      const bars = sorted
        .map(([name, val]) => {
          const w = Math.round(val * 100);
          const label = name.replace(/^(nrc_|hf_)/, "");
          const source = name.startsWith("hf_") ? "hf" : "nrc";
          return `
            <div class="bdp-emotion-row">
              <span class="bdp-emotion-label">${escapeHTML(label)}<sup>${source}</sup></span>
              <div class="bdp-emotion-bar-bg"><div class="bdp-emotion-bar" style="width:${w}%"></div></div>
              <span class="bdp-emotion-val">${(val * 100).toFixed(1)}%</span>
            </div>`;
        })
        .join("");
      emotionsHTML = `
        <div class="bdp-section">
          <div class="bdp-section-title">Emotions Detected</div>
          ${bars}
        </div>`;
    }

    return `
      <div class="bdp-header ${headerClass}">
        <span class="bdp-title">Bias Detector</span>
        <button class="bdp-close" title="Close">✕</button>
      </div>
      <div class="bdp-body">
        <div class="bdp-result-label ${labelClass}">${escapeHTML(r.label)}</div>

        <div class="bdp-confidence">
          <div class="bdp-confidence-bar-bg">
            <div class="bdp-confidence-bar ${labelClass}-bar" style="width:${barWidth}%"></div>
          </div>
          <span class="bdp-confidence-text">${pct}% confidence</span>
        </div>

        <div class="bdp-probabilities">
          <span>P(biased): <strong>${r.probability_biased.toFixed(4)}</strong></span>
          <span>P(non-biased): <strong>${r.probability_nonbiased.toFixed(4)}</strong></span>
        </div>

        ${wordsHTML}
        ${signalsHTML}
        ${emotionsHTML}

        <div class="bdp-text-preview">"${escapeHTML(displayText)}"</div>
      </div>
    `;
  }

  // -----------------------------------------------------------------------
  // Utilities
  // -----------------------------------------------------------------------

  function escapeHTML(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function formatSignalName(name) {
    return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }
})();
