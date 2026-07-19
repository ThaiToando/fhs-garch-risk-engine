import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 1. Enforce wide layout and hide the default Streamlit UI cruft
st.set_page_config(
    page_title="Institutional Risk Engine", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# 2. Inject custom CSS for a cleaner, "app-like" terminal aesthetic
st.markdown("""
    <style>
    /* Hide Streamlit header, footer, and menu for a clean app look */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tighten up the top padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    
    /* Clean, professional typography */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.title("QUANTITATIVE RISK ANALYTICS")
st.markdown("### Filtered Historical Simulation (FHS) Engine")
st.markdown("---")

# Sidebar: Institutional Control Panel
with st.sidebar:
    st.markdown("## MODEL PARAMETERS")
    st.markdown("<span style='color:#888888; font-size: 0.85em;'>DATA: ROLLING OOS BACKTEST</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    ticker = st.text_input("ASSET TICKER", "^GSPC")
    model_choice = st.selectbox("VOLATILITY DYNAMICS", ["EGARCH", "GJR", "GARCH"])
    alpha_choice = st.select_slider("CONFIDENCE LEVEL (α)", [0.01, 0.025, 0.05], 0.01)

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('results/backtest_comparison.csv', index_col=0, parse_dates=True)

try:
    df = load_data()
    
    # Use tabs, but styled cleanly
    tab_chart, tab_score = st.tabs(["REPLAY TERMINAL", "COVERAGE SCORECARD"])
    
    with tab_chart:
        # Build a highly customized, Bloomberg-style Plotly chart
        fig = go.Figure()
        
        # Realized Returns - Plotted as subtle gray bars/lines in the background
        fig.add_trace(go.Bar(
            x=df.index, 
            y=df['realized'], 
            name='Realized Return', 
            marker_color='rgba(150, 150, 150, 0.3)',
            marker_line_width=0
        ))
        
        # FHS VaR - Bright Cyan for the "winning" model
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=-df['fhs_var'], 
            mode='lines', 
            name='FHS VaR', 
            line=dict(color='#00E5FF', width=2)
        ))
        
        # Plain HS VaR - Muted Amber for the baseline
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=-df['hs_var'], 
            mode='lines', 
            name='Plain HS VaR', 
            line=dict(color='#FF9F1C', width=1.5, dash='dot')
        ))
                                 
        # Professional Dark Theme Layout
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            height=550,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title="DAILY RETURN (%)",
            xaxis_title="",
            hovermode="x unified",
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=11)
            ),
            xaxis=dict(
                showgrid=False, 
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor="rgba(255,255,255,0.1)", 
                zeroline=True, 
                zerolinecolor="rgba(255,255,255,0.3)"
            )
        )
        
        st.plotly_chart(fig)
        
    with tab_score:
        st.markdown("#### BASELINE COMPARISON")
        st.markdown("Target exceptions for chosen confidence level vs. actual realized breaches.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Calculate sums
        fhs_exc = df['fhs_exc'].sum()
        hs_exc = df['hs_exc'].sum()
        normal_exc = df['normal_exc'].sum()
        
        # Display as metric cards
        col1, col2, col3 = st.columns(3)
        col1.metric("FHS-GARCH BREACHES", int(fhs_exc))
        col2.metric("PLAIN HS BREACHES", int(hs_exc))
        col3.metric("NORMAL PARAMETRIC BREACHES", int(normal_exc))
        
        st.markdown("---")
        st.markdown("**ANALYSIS:** FHS actively scales to conditional volatility regimes. Notice the absence of 'ghost plateaus' compared to the static Historical Simulation window.")

except FileNotFoundError:
    st.error("DATA MISSING: Ensure 'results/backtest_comparison.csv' is generated and available in the root directory.")