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

    const coords = getSelectionCoords();
    if (coords) {
      panel.style.top = `${coords.top}px`;
      panel.style.left = `${coords.left}px`;
    }

    panel.addEventListener("click", (e) => {
      if (e.target.classList.contains("bdp-close")) removePanel();
    });

    document.body.appendChild(panel);

    setTimeout(() => {
      document.addEventListener("mousedown", outsideClick);
    }, 100);

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
  // Friendly signal explanations
  // Maps raw signal keys → { label, explain(value) }
  // -----------------------------------------------------------------------

  const SIGNAL_MAP = {
    // Sentiment
    vader_compound: {
      label: "Overall Tone",
      explain: (v) => {
        if (v >  0.5) return "Very positive language";
        if (v >  0.1) return "Slightly positive language";
        if (v < -0.5) return "Very negative language";
        if (v < -0.1) return "Slightly negative language";
        return "Neutral language";
      },
    },
    vader_neg: {
      label: "Negative Sentiment",
      explain: (v) => v > 0.2 ? "High negativity in word choice" : "Low negativity",
    },
    vader_pos: {
      label: "Positive Sentiment",
      explain: (v) => v > 0.2 ? "High positivity in word choice" : "Low positivity",
    },
    textblob_polarity: {
      label: "Emotional Slant",
      explain: (v) => {
        if (v >  0.3) return "Words lean positive / favorable";
        if (v < -0.3) return "Words lean negative / critical";
        return "Fairly balanced word choice";
      },
    },
    textblob_subjectivity: {
      label: "Opinion vs. Fact",
      explain: (v) =>
        v > 0.5
          ? "Mostly opinion-based writing"
          : v > 0.3
          ? "Mix of opinion and fact"
          : "Mostly factual writing",
    },

    // Loaded language
    loaded_word_ratio: {
      label: "Charged Language",
      explain: (v) =>
        v > 0.15
          ? "Many emotionally loaded words"
          : v > 0.05
          ? "Some charged words present"
          : "Few emotionally charged words",
    },
    loaded_word_count: {
      label: "Charged Word Count",
      explain: (v) => `${Math.round(v)} emotionally loaded word${v !== 1 ? "s" : ""} found`,
    },

    // Punctuation / style
    exclamation_ratio: {
      label: "Exclamation Marks",
      explain: (v) =>
        v > 0.05
          ? "Heavy use of exclamation marks — adds urgency or alarm"
          : "Normal punctuation",
    },
    exclamation_count: {
      label: "Exclamation Count",
      explain: (v) => `${Math.round(v)} exclamation mark${Math.round(v) !== 1 ? "s" : ""} found`,
    },
    caps_ratio: {
      label: "ALL CAPS Usage",
      explain: (v) =>
        v > 0.05
          ? "Significant all-caps text — often used for emphasis or alarm"
          : "Normal capitalization",
    },
    question_ratio: {
      label: "Rhetorical Questions",
      explain: (v) =>
        v > 0.05
          ? "Frequent questions — may be used to imply rather than state"
          : "Few or no rhetorical questions",
    },

    // Readability
    flesch_reading_ease: {
      label: "Reading Level",
      explain: (v) => {
        if (v > 70) return "Easy to read (everyday language)";
        if (v > 50) return "Moderately complex writing";
        return "Complex or academic writing";
      },
    },
    avg_word_length: {
      label: "Word Complexity",
      explain: (v) =>
        v > 6
          ? "Long, complex words — may obscure meaning"
          : "Simple, everyday vocabulary",
    },
    avg_sentence_length: {
      label: "Sentence Length",
      explain: (v) =>
        v > 25
          ? "Long sentences — harder to follow"
          : v < 10
          ? "Short, punchy sentences — can feel aggressive"
          : "Normal sentence length",
    },

    // Lexical diversity
    type_token_ratio: {
      label: "Vocabulary Variety",
      explain: (v) =>
        v > 0.7
          ? "Rich, varied vocabulary"
          : v < 0.4
          ? "Repetitive vocabulary — may signal deliberate repetition"
          : "Average vocabulary variety",
    },

    // Named entities / specificity
    named_entity_ratio: {
      label: "Names & Places",
      explain: (v) =>
        v > 0.1
          ? "Lots of specific names and places referenced"
          : "Few specific names or places",
    },

    assertive_ratio: {
      label: "Assertive Language",
      explain: (v) =>
        v > 0.1
          ? "Strong assertive tone — states things as definitive facts"
          : "Low assertiveness, more measured language",
    },
    _default: {
      label: null,
      explain: (v) => `${Number(v).toFixed(3)}`,
    },
  };

  function getSignalInfo(key, value) {
    const entry = SIGNAL_MAP[key] || SIGNAL_MAP._default;
    const label = entry.label || formatSignalName(key);
    const explanation = entry.explain(value);
    return { label, explanation };
  }

  // -----------------------------------------------------------------------
  // Emotion label prettifier
  // -----------------------------------------------------------------------

  const EMOTION_LABELS = {
    anger: "Anger",
    fear: "Fear",
    anticipation: "Anticipation",
    trust: "Trust",
    surprise: "Surprise",
    sadness: "Sadness",
    disgust: "Disgust",
    joy: "Joy",
    neutral: "Neutral",
  };

  function prettyEmotion(raw) {
    const clean = raw.replace(/^(nrc_|hf_)/, "").toLowerCase();
    return EMOTION_LABELS[clean] || clean.charAt(0).toUpperCase() + clean.slice(1);
  }

  function emotionSource(raw) {
    if (raw.startsWith("hf_"))  return "AI model";
    if (raw.startsWith("nrc_")) return "word analysis";
    return "";
  }

  // -----------------------------------------------------------------------
  // Bias meter label
  // -----------------------------------------------------------------------

  function biasLevel(prob) {
    if (prob >= 0.85) return { text: "Very likely biased",  color: "#ff2d4e" };
    if (prob >= 0.65) return { text: "Probably biased",     color: "#ff6b35" };
    if (prob >= 0.50) return { text: "Possibly biased",     color: "#ffb020" };
    if (prob >= 0.35) return { text: "Probably balanced",   color: "#8bc34a" };
    return                   { text: "Likely balanced",     color: "#00e8a0" };
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
    const isBiased   = r.label === "Propaganda / Biased";
    const headerClass = isBiased ? "bdp-header-biased" : "bdp-header-clean";
    const prob        = r.probability_biased ?? 0;
    const pct         = Math.round(prob * 100);
    const level       = biasLevel(prob);

    // ── Bias meter ──
    const meterHTML = `
      <div class="bdp-meter-wrap">
        <div class="bdp-meter-spectrum-labels">
          <span>Balanced</span>
          <span>Slightly Balanced</span>
          <span>Possibly Biased</span>
          <span>Biased</span>
        </div>
        <div class="bdp-meter-spectrum-track">
          <div class="bdp-meter-spectrum-marker" style="left:${pct}%"></div>
        </div>
        <div class="bdp-meter-bottom">
          <span style="color:${level.color}; font-weight:600">${level.text}</span>
          <span class="bdp-pct">${pct}% biased</span>
        </div>
      </div>`;

    // ── Auto-corrections ──
    let correctionsHTML = "";
    if (r.corrections && r.corrections.length) {
      const pills = r.corrections.slice(0, 6).map(c =>
        `<span class="bdp-fix-pill">
          <s class="bdp-fix-orig">${escapeHTML(c.original)}</s>
          → <span class="bdp-fix-new">${escapeHTML(c.corrected)}</span>
        </span>`
      ).join("");
      correctionsHTML = `
        <div class="bdp-section bdp-corrections">
          <div class="bdp-section-title">✦ ${r.corrections.length} typo${r.corrections.length > 1 ? "s" : ""} auto-corrected</div>
          <div class="bdp-fix-pills">${pills}</div>
        </div>`;
    }

    // ── Triggered words ──
    const wordsHTML = r.triggered_words?.length
      ? `<div class="bdp-section">
           <div class="bdp-section-title">⚡ Emotionally Charged Words</div>
           <p class="bdp-hint-text">These words tend to provoke a reaction rather than inform.</p>
           <div class="bdp-words">${r.triggered_words.map(w =>
               `<span class="bdp-word">${escapeHTML(w)}</span>`
             ).join("")}
           </div>
         </div>`
      : "";

    // ── Linguistic signals — plain English ──
    let signalsHTML = "";
    const signals = r.signal_summary || r.top_signals || {};
    if (Object.keys(signals).length) {
      const rows = Object.entries(signals)
        .map(([key, val]) => {
          const { label, explanation } = getSignalInfo(key, val);
          return `
            <div class="bdp-signal-row">
              <strong class="bdp-signal-label">${escapeHTML(label)}</strong>
              <span class="bdp-signal-sep"> — </span>
              <span class="bdp-signal-explain">${escapeHTML(explanation)}</span>
            </div>`;
        })
        .join("");
      signalsHTML = `
        <div class="bdp-section">
          <div class="bdp-section-title">📊 Writing Style Signals</div>
          <p class="bdp-hint-text">What the AI noticed about how this text is written.</p>
          ${rows}
        </div>`;
    }

    // ── Emotions — plain English ──
    let emotionsHTML = "";
    if (r.emotions && Object.keys(r.emotions).length) {
      const sorted = Object.entries(r.emotions)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);

      const bars = sorted.map(([key, val]) => {
        const w      = Math.round(val * 100);
        const name   = prettyEmotion(key);
        const source = emotionSource(key);
        return `
          <div class="bdp-emotion-row">
            <span class="bdp-emotion-name">${escapeHTML(name)}${source ? ` <span class="bdp-emotion-source">${source}</span>` : ""}</span>
            <div class="bdp-emotion-bar-bg">
              <div class="bdp-emotion-bar" style="width:${w}%"></div>
            </div>
            <span class="bdp-emotion-val">${w}%</span>
          </div>`;
      }).join("");

      emotionsHTML = `
        <div class="bdp-section">
          <div class="bdp-section-title">🎭 Emotional Tone</div>
          <p class="bdp-hint-text">Emotions detected in the writing style.</p>
          ${bars}
        </div>`;
    }

    // ── Snippet ──
    const displayText = r.text?.length > 100 ? r.text.slice(0, 100) + "…" : (r.text || "");

    return `
      <div class="bdp-header ${headerClass}">
        <span class="bdp-title">Bias Detector</span>
        <button class="bdp-close" title="Close">✕</button>
      </div>
      <div class="bdp-body">
        ${meterHTML}
        ${correctionsHTML}
        ${wordsHTML}
        ${signalsHTML}
        ${emotionsHTML}
        ${displayText ? `<div class="bdp-text-preview">"${escapeHTML(displayText)}"</div>` : ""}
      </div>
    `;
  }

  // -----------------------------------------------------------------------
  // Utilities
  // -----------------------------------------------------------------------

  function escapeHTML(str) {
    const div = document.createElement("div");
    div.textContent = String(str);
    return div.innerHTML;
  }

  function formatSignalName(name) {
    return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  // Inject alignment + spectrum bar styles
  function injectStyles() {
    if (document.getElementById("bdp-extra-styles")) return;
    const s = document.createElement("style");
    s.id = "bdp-extra-styles";
    s.textContent = `
      /* ── Spectrum bias bar ── */
      .bdp-meter-wrap { margin-bottom: 12px; }

      .bdp-meter-spectrum-labels {
        display: flex;
        justify-content: space-between;
        font-size: 9px;
        color: #6b7f8f;
        margin-bottom: 4px;
        letter-spacing: 0.03em;
      }

      .bdp-meter-spectrum-track {
        position: relative;
        height: 8px;
        border-radius: 99px;
        background: linear-gradient(to right, #00e8a0, #8bc34a, #ffb020, #ff6b35, #ff2d4e);
        margin-bottom: 6px;
      }

      .bdp-meter-spectrum-marker {
        position: absolute;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #fff;
        border: 2px solid #000;
        box-shadow: 0 0 4px rgba(0,0,0,0.4);
        transition: left 0.5s cubic-bezier(.23,1,.32,1);
      }

      .bdp-meter-bottom {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
      }

      /* ── Aligned emotion rows ── */
      .bdp-emotion-row {
        display: grid;
        grid-template-columns: 110px 1fr 36px;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
      }

      .bdp-emotion-name {
        font-size: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .bdp-emotion-source {
        font-size: 9px;
        opacity: 0.5;
        margin-left: 4px;
      }

      .bdp-emotion-bar-bg {
        height: 5px;
        border-radius: 99px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
      }

      .bdp-emotion-bar {
        height: 100%;
        border-radius: 99px;
        background: #c8f000;
      }

      .bdp-emotion-val {
        font-size: 11px;
        text-align: right;
        opacity: 0.7;
        white-space: nowrap;
      }

      /* ── Signal rows — forced single line ── */
      .bdp-signal-row {
        display: flex;
        align-items: baseline;
        gap: 0;
        margin-bottom: 5px;
        font-size: 11px;
        line-height: 1.4;
        white-space: nowrap;
        overflow: hidden;
      }

      .bdp-signal-label {
        flex-shrink: 0;
        white-space: nowrap;
      }

      .bdp-signal-sep {
        color: #4a5568;
        flex-shrink: 0;
        white-space: nowrap;
      }

      .bdp-signal-explain {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
    `;
    document.head.appendChild(s);
  }

  injectStyles();
})