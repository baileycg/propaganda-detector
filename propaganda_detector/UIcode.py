import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import re

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
        number = {'suffix': "%", 'font': {'size': 24}},
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

    fig = go.Figure(data=go.Scatterpolar(
        r=scores, theta=categories, fill='toself',
        line_color='rgba(99, 110, 250, 0.8)',
        fillcolor='rgba(99, 110, 250, 0.4)'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.0], gridcolor='lightgrey')),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ==========================================
# 2. Dummy Analysis Logic (PLACEHOLDER)
# ==========================================
def analyze_text_dummy(text: str) -> dict:
    text = (text or "").strip()
    if not text: return {}

    is_propaganda = "destroy" in text.lower() or "corrupt" in text.lower()
    overall_score = 0.85 if is_propaganda else 0.15
    
    signals = {
        "Political Bias": overall_score * 0.9,
        "Loaded Language": overall_score * 0.8,
        "Fear Mongering": 0.7 if is_propaganda else 0.1
    }
    
    emotions = {
        "Anger": 0.8 if is_propaganda else 0.1,
        "Fear": 0.6 if is_propaganda else 0.1,
        "Joy": 0.05 if is_propaganda else 0.7,
        "Sadness": 0.3,
        "Trust": 0.1 if is_propaganda else 0.6
    }

    highlights = []
    words = text.split(' ')
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word).lower()
        if clean_word in ["destroying", "corrupt", "regime", "threat"]:
            highlights.append((word + " ", "Trigger Word"))
        elif clean_word in ["freedom", "democracy", "stop"]:
            highlights.append((word + " ", "Framing"))
        else:
            highlights.append((word + " ", None))
            
    explanation = (
        f"Analysis result: Bias probability is estimated at {overall_score*100:.0f}%. "
        f"{'Strong provocative tone and political framing detected.' if is_propaganda else 'The text appears to be relatively neutral.'}"
    )

   
    return {
        "overall_score": overall_score,
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

    result = analyze_text_dummy(text)

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

with gr.Blocks(title="Propaganda & Bias Lens") as demo:
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
    demo.launch(theme=theme, inbrowser=True, prevent_thread_lock=False)
