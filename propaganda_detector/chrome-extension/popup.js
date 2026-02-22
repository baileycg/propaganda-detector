// popup.js – Settings popup logic.

const DEFAULTS = {
  apiUrl: "http://localhost:8000",
  modelType: "transformer",
  threshold: 0.5,
};

const $apiUrl = document.getElementById("apiUrl");
const $modelType = document.getElementById("modelType");
const $threshold = document.getElementById("threshold");
const $thresholdVal = document.getElementById("thresholdVal");
const $saveBtn = document.getElementById("saveBtn");
const $resetBtn = document.getElementById("resetBtn");
const $savedMsg = document.getElementById("savedMsg");
const $statusDot = document.getElementById("statusDot");
const $statusText = document.getElementById("statusText");

// -----------------------------------------------------------------------
// Load saved settings
// -----------------------------------------------------------------------

chrome.storage.sync.get(DEFAULTS, (items) => {
  $apiUrl.value = items.apiUrl;
  $modelType.value = items.modelType;
  $threshold.value = items.threshold;
  $thresholdVal.textContent = Number(items.threshold).toFixed(2);
  checkHealth(items.apiUrl);
});

// -----------------------------------------------------------------------
// Threshold slider live update
// -----------------------------------------------------------------------

$threshold.addEventListener("input", () => {
  $thresholdVal.textContent = Number($threshold.value).toFixed(2);
});

// -----------------------------------------------------------------------
// Save
// -----------------------------------------------------------------------

$saveBtn.addEventListener("click", () => {
  const settings = {
    apiUrl: $apiUrl.value.replace(/\/+$/, ""),
    modelType: $modelType.value,
    threshold: parseFloat($threshold.value),
  };

  chrome.storage.sync.set(settings, () => {
    $savedMsg.style.display = "block";
    setTimeout(() => ($savedMsg.style.display = "none"), 2000);
    checkHealth(settings.apiUrl);
  });
});

// -----------------------------------------------------------------------
// Reset
// -----------------------------------------------------------------------

$resetBtn.addEventListener("click", () => {
  $apiUrl.value = DEFAULTS.apiUrl;
  $modelType.value = DEFAULTS.modelType;
  $threshold.value = DEFAULTS.threshold;
  $thresholdVal.textContent = Number(DEFAULTS.threshold).toFixed(2);
});

// -----------------------------------------------------------------------
// Health check
// -----------------------------------------------------------------------

async function checkHealth(url) {
  try {
    const resp = await fetch(`${url}/health`, { signal: AbortSignal.timeout(3000) });
    if (!resp.ok) throw new Error();
    const data = await resp.json();
    $statusDot.className = "status-dot connected";
    $statusText.textContent = `Connected — ${data.models_loaded.join(", ")} loaded`;
  } catch {
    $statusDot.className = "status-dot disconnected";
    $statusText.textContent = "Cannot reach API";
  }
}

// Re-check when URL field changes
$apiUrl.addEventListener("change", () => checkHealth($apiUrl.value.replace(/\/+$/, "")));
