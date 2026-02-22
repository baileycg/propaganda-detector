import sys
import logging
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Path setup – allow `from src.predictor import ...` to resolve correctly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load the detector so the app still starts even if the model is absent.
# ---------------------------------------------------------------------------
_detector = None
_detector_error: str | None = None

def _get_detector():
    global _detector, _detector_error
    if _detector is not None:
        return _detector, None
    if _detector_error is not None:
        return None, _detector_error
    try:
        from src.predictor import TransformerDetector
        _detector = TransformerDetector(model_name="distilbert_model", threshold=0.5)
        logger.info("TransformerDetector loaded successfully.")
        return _detector, None
    except FileNotFoundError as exc:
        _detector_error = (
            f"Model not found: {exc}\n\n"
            "Run download_model.py first:\n"
            "  py download_model.py"
        )
        logger.error(_detector_error)
        return None, _detector_error
    except Exception as exc:
        _detector_error = f"Failed to load model: {exc}"
        logger.error(_detector_error)
        return None, _detector_error


# ==========================================
# 1. Helper Functions for Plotly Charts
# ==========================================

def create_gauge_chart(score: float) -> go.Figure:
    if score < 0.3:
        bar_color = "#2ecc71"
    elif score < 0.7:
        bar_color = "#f1c40f"
    else:
        bar_color = "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Overall Bias Probability", "font": {"size": 16}},
        number={"suffix": "%", "font": {"size": 36, "color": bar_color, "family": "Arial Black"}, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": bar_color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 30],  "color": "rgba(46, 204, 113, 0.15)"},
                {"range": [30, 70], "color": "rgba(241, 196, 15, 0.15)"},
                {"range": [70, 100], "color": "rgba(231, 76, 60, 0.15)"},
            ],
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_radar_chart(emotions_dict: dict) -> go.Figure:
    if not emotions_dict:
        fig = go.Figure()
        fig.update_layout(height=250, polar=dict(radialaxis=dict(visible=False)))
        return fig

    categories = list(emotions_dict.keys())
    scores = list(emotions_dict.values())
    categories = [*categories, categories[0]]
    scores = [*scores, scores[0]]

    # Auto-scale axis so small NRC scores are visible
    max_val = max(scores) if max(scores) > 0 else 0.1
    axis_max = round(max_val * 1.3, 2)

    fig = go.Figure(data=go.Scatterpolar(
        r=scores, theta=categories, fill="toself",
        line_color="rgba(220, 50, 50, 0.9)",
        fillcolor="rgba(220, 50, 50, 0.35)",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, axis_max],
                gridcolor="#aaaaaa",
                tickfont=dict(size=10, color="black"),
                tickformat=".2f",
            ),
            angularaxis=dict(tickfont=dict(size=12, color="black")),
            bgcolor="rgba(240,240,240,0.5)",
        ),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor="white",
    )
    return fig


# ==========================================
# 2. Bridge between PredictionResult and UI
# ==========================================

_SIGNAL_DISPLAY_NAMES = {
    "vader_compound":    "Sentiment (VADER compound)",
    "vader_neg":         "Negative Sentiment",
    "loaded_word_ratio": "Loaded Language Ratio",
    "assertive_ratio":   "Assertive Language Ratio",
    "caps_ratio":        "ALL-CAPS Ratio",
    "exclamation_count": "Exclamation Marks",
}

_NRC_DISPLAY = {
    "anger":        "Anger",
    "fear":         "Fear",
    "anticipation": "Anticipation",
    "trust":        "Trust",
    "surprise":     "Surprise",
    "sadness":      "Sadness",
    "disgust":      "Disgust",
    "joy":          "Joy",
}

_HF_DISPLAY = {
    "anger":   "Anger",
    "disgust": "Disgust",
    "fear":    "Fear",
    "joy":     "Joy",
    "neutral": "Neutral",
    "sadness": "Sadness",
    "surprise": "Surprise",
}


def _build_emotions_for_radar(emotions: dict | None) -> dict:
    if not emotions:
        return {}
    hf = {_HF_DISPLAY[k[3:]]: v for k, v in emotions.items()
          if k.startswith("hf_") and k[3:] in _HF_DISPLAY}
    if hf:
        return hf
    nrc = {_NRC_DISPLAY[k[4:]]: v for k, v in emotions.items()
           if k.startswith("nrc_") and k[4:] in _NRC_DISPLAY}
    return nrc


def _build_highlights(text: str, triggered_words: list) -> list:
    triggered_set = set(w.lower() for w in triggered_words)
    highlights = []
    for token in text.split(" "):
        clean = "".join(c for c in token if c.isalpha()).lower()
        if clean in triggered_set:
            highlights.append((token + " ", "Trigger Word"))
        else:
            highlights.append((token + " ", None))
    return highlights


def analyze_text(text: str) -> dict:
    detector, error = _get_detector()
    if detector is None:
        return {"error": error}

    result = detector.predict(text)

    signals = {
        _SIGNAL_DISPLAY_NAMES.get(k, k): abs(v)
        for k, v in result.signal_summary.items()
    }
    emotions = _build_emotions_for_radar(result.emotions)
    highlights = _build_highlights(text, result.triggered_words)

    triggered_str = (
        ", ".join(result.triggered_words[:6]) if result.triggered_words
        else "none detected"
    )
    explanation = (
        f"Classification: {result.label} (confidence {result.confidence * 100:.1f}%)\n\n"
        f"Bias probability: {result.probability_biased * 100:.1f}%. "
        f"Loaded / assertive trigger words found: {triggered_str}."
    )

    return {
        "overall_score": result.probability_biased,
        "signals": signals,
        "emotions": emotions,
        "highlights": highlights,
        "explanation": explanation,
    }


# ==========================================
# 3. Gradio UI Adapter Function
# ==========================================

def ui_analyze(text: str):
    text = (text or "").strip()

    if not text:
        empty_fig = go.Figure()
        empty_fig.update_layout(height=250)
        return empty_fig, empty_fig, pd.DataFrame(), [], "Please enter text to analyze."

    result = analyze_text(text)

    if "error" in result:
        empty_fig = go.Figure()
        empty_fig.update_layout(height=250)
        return empty_fig, empty_fig, pd.DataFrame(), [], result["error"]

    overall      = result.get("overall_score", 0.0)
    signals_data = result.get("signals", {})
    emotions_data= result.get("emotions", {})
    highlights   = result.get("highlights", [])
    explanation  = result.get("explanation", "")

    gauge_plot = create_gauge_chart(overall)
    radar_plot = create_radar_chart(emotions_data)

    def _fmt_signal(k, v):
        if k == "Exclamation Marks":
            return str(int(round(v)))
        return f"{v * 100:.1f}%"

    signals_df = pd.DataFrame(
        [{"Signal Type": k, "Score": _fmt_signal(k, v)} for k, v in signals_data.items()]
    ).sort_values("Signal Type").reset_index(drop=True)

    return gauge_plot, radar_plot, signals_df, highlights, explanation


# ==========================================
# 4. Gradio UI Layout
# ==========================================

theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(title="Propaganda & Bias Lens") as demo:
    gr.Markdown(
        """
        # Propaganda & Political Bias Lens
        **AI-Powered Dashboard for Detecting Political Bias and Propaganda in News & Text**
        """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            inp = gr.Textbox(
                label="Input Text for Analysis",
                placeholder="Paste a news article, social media post, or political speech here...",
                lines=12,
            )

            with gr.Row():
                run_btn   = gr.Button("Analyze Text", variant="primary", scale=2)
                paste_btn = gr.Button("Paste", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

            gr.Examples(
                examples=[
                    ["They are destroying our country! The corrupt regime must be stopped now before it's too late."],
                    ["The city council held a routine meeting on Tuesday to discuss the annual budget for public parks and recreation centers."],
                ],
                inputs=inp,
                label="Test Examples (Click to try)",
            )

            explanation = gr.Textbox(
                label="AI Analysis Summary",
                lines=4,
                interactive=False,
            )

        with gr.Column(scale=2):
            gr.Markdown("### Analysis Dashboard")

            with gr.Row():
                with gr.Column():
                    gauge_output = gr.Plot(label="Overall Bias Probability")
                with gr.Column():
                    radar_output = gr.Plot(label="Emotion Spectrum Radar")

            with gr.Tabs():
                with gr.TabItem("Evidence Highlights"):
                    highlight_output = gr.HighlightedText(
                        label="Detected Bias/Trigger Words in Source Text",
                        combine_adjacent=True,
                        show_legend=True,
                        color_map={"Trigger Word": "red"},
                    )

                with gr.TabItem("Detailed Signals Table"):
                    signals_tbl = gr.Dataframe(
                        headers=["Signal Type", "Score"],
                        datatype=["str", "str"],
                        interactive=False,
                    )

    run_btn.click(
        fn=ui_analyze,
        inputs=[inp],
        outputs=[gauge_output, radar_output, signals_tbl, highlight_output, explanation],
    )

    paste_btn.click(
        fn=None,
        inputs=[],
        outputs=[inp],
        js="""
        async () => {
            try {
                const text = await navigator.clipboard.readText();
                return text;
            } catch (err) {
                alert("Please allow clipboard access in your browser to use the Paste feature.");
                return "";
            }
        }
        """,
    )

    _empty = go.Figure()
    _empty.update_layout(height=250)
    clear_btn.click(
        fn=lambda: ("", _empty, _empty, pd.DataFrame(), [], ""),
        inputs=[],
        outputs=[inp, gauge_output, radar_output, signals_tbl, highlight_output, explanation],
    )

# ==========================================
# 5. Launch
# ==========================================
if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=theme)
