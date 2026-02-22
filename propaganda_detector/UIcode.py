import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import re

sys.path.insert(0, str(Path(__file__).parent))
from src.predictor import TransformerDetector

detector = TransformerDetector(model_name="distilbert_model")

# ==========================================
# 1. Helper Functions for Plotly Charts
# ==========================================
def create_gauge_chart(score):
    if score < 0.3: color = "green"
    elif score < 0.7: color = "gold"
    else: color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100, 
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Bias Probability", 'font': {'size': 16}},
        number = {'suffix': "%", 'font': {'size': 24}, 'valueformat': '.1f'},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.1)'},
                {'range': [30, 70], 'color': 'rgba(255, 255, 0, 0.1)'},
                {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.1)'}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_radar_chart(emotions_dict):
    if not emotions_dict:
        fig = go.Figure()
        fig.update_layout(height=250, polar=dict(radialaxis=dict(visible=False)))
        return fig

    categories = list(emotions_dict.keys())
    scores = list(emotions_dict.values())
    categories = [*categories, categories[0]]
    scores = [*scores, scores[0]]

    max_val = max(scores) if max(scores) > 0 else 0.1
    axis_max = round(max_val * 1.3, 2)

    fig = go.Figure(data=go.Scatterpolar(
        r=scores, theta=categories, fill='toself',
        line_color='rgba(220, 50, 50, 0.9)',
        fillcolor='rgba(220, 50, 50, 0.35)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, axis_max],
                gridcolor='#aaaaaa',
                tickfont=dict(size=10, color='black'),
                tickformat='.2f',
            ),
            angularaxis=dict(tickfont=dict(size=12, color='black')),
            bgcolor='rgba(240,240,240,0.5)'
        ),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor='white'
    )
    return fig

# ==========================================
# 2. Real Analysis Logic
# ==========================================
def analyze_text(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}

    result = detector.predict(text)

    signals = {k.replace("_", " ").title(): v for k, v in result.signal_summary.items()}

    emotions = {}
    if result.emotions:
        nrc = {k[4:].title(): v for k, v in result.emotions.items() if k.startswith("nrc_")}
        if nrc:
            emotions = nrc

    triggered = set(result.triggered_words)
    highlights = []
    for word in text.split(' '):
        clean = re.sub(r'[^\w]', '', word).lower()
        if clean in triggered:
            highlights.append((word + " ", "Trigger Word"))
        else:
            highlights.append((word + " ", None))

    explanation = (
        f"Label: {result.label} (confidence: {result.confidence*100:.1f}%)\n"
        f"P(biased): {result.probability_biased:.4f}\n"
        + (f"Triggered words: {', '.join(result.triggered_words)}" if result.triggered_words else "No trigger words detected.")
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
        empty_fig = go.Figure(); empty_fig.update_layout(height=250)
        return empty_fig, empty_fig, pd.DataFrame(), [], "Please enter text to analyze."

    result = analyze_text(text)

    overall = result.get("overall_score", 0.0)
    signals_data = result.get("signals", {})
    emotions_data = result.get("emotions", {})
    highlights_data = result.get("highlights", [])
    explanation_text = result.get("explanation", "")

    gauge_plot = create_gauge_chart(overall)
    radar_plot = create_radar_chart(emotions_data)

    signals_df = pd.DataFrame(
        [{"Signal Type": k, "Score": f"{v:.2f}"} for k, v in signals_data.items()]
    ).sort_values("Signal Type")

    return gauge_plot, radar_plot, signals_df, highlights_data, explanation_text

# ==========================================
# 4. Gradio UI Layout Definition (Blocks)
# ==========================================
theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(title="Propaganda & Bias Lens", theme=theme) as demo:
    gr.Markdown(
        """
        # 📰 Propaganda & Political Bias Lens
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
                run_btn = gr.Button("🔍 Analyze Text", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")
            
            gr.Examples(
                examples=[
                    ["They are destroying our country! The corrupt regime must be stopped now before it's too late."],
                    ["The city council held a routine meeting on Tuesday to discuss the annual budget for public parks and recreation centers."],
                ],
                inputs=inp,
                label="Test Examples (Click to try)"
            )
            
            explanation = gr.Textbox(
                label="💡 AI Analysis Summary",
                lines=4,
                interactive=False,
            )

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Dashboard")
            
            with gr.Row():
                with gr.Column():
                    gauge_output = gr.Plot(label="Overall Bias Probability")
                with gr.Column():
                    radar_output = gr.Plot(label="Emotion Spectrum Radar")
            
            with gr.Tabs():
                with gr.TabItem("🖍️ Evidence Highlights"):
                    highlight_output = gr.HighlightedText(
                        label="Detected Bias/Trigger Words in Source Text",
                        combine_adjacent=True,
                        show_legend=True,
                        color_map={"Trigger Word": "red", "Framing": "orange"}
                    )
                
                with gr.TabItem("📋 Detailed Signals Table"):
                    signals_tbl = gr.Dataframe(
                        headers=["Signal Type", "Score"],
                        datatype=["str", "str"],
                        interactive=False
                    )

    run_btn.click(
        fn=ui_analyze,
        inputs=[inp],
        outputs=[gauge_output, radar_output, signals_tbl, highlight_output, explanation],
    )
    
    empty_fig = go.Figure(); empty_fig.update_layout(height=250)
    clear_btn.click(
        fn=lambda: ("", empty_fig, empty_fig, pd.DataFrame(), [], ""),
        inputs=[],
        outputs=[inp, gauge_output, radar_output, signals_tbl, highlight_output, explanation],
    )

# ==========================================
# 5. Launch App
# ==========================================
if __name__ == "__main__":
    demo.launch(inbrowser=True)
