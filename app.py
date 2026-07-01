"""
APP.PY — Threat Copilot Dashboard
──────────────────────────────────────────────
Professional SOC-analyst-style Streamlit interface.
Dark theme, monospace data display, severity-coded
alerts — designed like a real threat intelligence
console, not a generic chatbot wrapper.

Usage: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from rag_engine import ThreatCopilot

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Threat Copilot — Cyber Incident Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# DESIGN SYSTEM — SOC console aesthetic
# Near-black base, signal colors for severity,
# monospace for data, Inter for UI text
# ──────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    :root {
        --bg-base: #0A0E14;
        --bg-panel: #101720;
        --bg-panel-raised: #161E29;
        --border: #1B2430;
        --signal-verified: #00D9A3;
        --signal-critical: #FF4757;
        --signal-medium: #FFA94D;
        --text-primary: #E4E8ED;
        --text-muted: #7B8794;
        --mono: 'JetBrains Mono', monospace;
        --sans: 'Inter', sans-serif;
    }

    .stApp { background-color: var(--bg-base); font-family: var(--sans); }
    
    /* Hide default streamlit chrome for cleaner console feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Header bar ── */
    .console-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 28px;
    }
    .console-title {
        font-family: var(--mono);
        font-size: 20px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .console-title .glyph { color: var(--signal-verified); }
    .console-subtitle {
        font-size: 12px;
        color: var(--text-muted);
        font-family: var(--mono);
        margin-top: 2px;
    }
    .status-pill {
        font-family: var(--mono);
        font-size: 11px;
        padding: 5px 12px;
        border-radius: 99px;
        background: rgba(0,217,163,0.1);
        border: 1px solid rgba(0,217,163,0.3);
        color: var(--signal-verified);
        font-weight: 500;
    }

    /* ── Stat strip ── */
    .stat-card {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px 18px;
        height: 100%;
    }
    .stat-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        font-family: var(--mono);
        margin-bottom: 6px;
    }
    .stat-value {
        font-family: var(--mono);
        font-size: 26px;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }
    .stat-value.critical { color: var(--signal-critical); }
    .stat-value.verified { color: var(--signal-verified); }

    /* ── Query console ── */
    .query-label {
        font-family: var(--mono);
        font-size: 11px;
        color: var(--signal-verified);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }

    /* Style streamlit text input to look like terminal */
    .stTextInput input {
        background-color: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        font-family: var(--mono) !important;
        font-size: 14px !important;
        padding: 14px 16px !important;
        border-radius: 6px !important;
    }
    .stTextInput input:focus {
        border-color: var(--signal-verified) !important;
        box-shadow: 0 0 0 1px var(--signal-verified) !important;
    }
    .stTextInput input::placeholder { color: var(--text-muted) !important; }

    .stButton button {
        background-color: var(--signal-verified) !important;
        color: #0A0E14 !important;
        font-family: var(--mono) !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 12px 24px !important;
        letter-spacing: 0.02em;
    }
    .stButton button:hover { background-color: #00BF8F !important; }

    /* ── Suggested query chips ── */
    .chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 4px; }

    /* ── Case file (answer) card ── */
    .case-file {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-left: 3px solid var(--signal-verified);
        border-radius: 6px;
        padding: 22px 24px;
        margin: 16px 0;
    }
    .case-file-header {
        font-family: var(--mono);
        font-size: 11px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
    }
    .case-file-body {
        font-size: 15px;
        line-height: 1.7;
        color: var(--text-primary);
    }

    /* ── Source incident cards ── */
    .source-card {
        background: var(--bg-panel-raised);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .source-card-name {
        font-weight: 600;
        font-size: 13px;
        color: var(--text-primary);
        margin-bottom: 6px;
    }
    .source-meta {
        font-family: var(--mono);
        font-size: 11px;
        color: var(--text-muted);
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
    }
    .source-meta span { white-space: nowrap; }

    /* ── Confidence gauge ── */
    .gauge-wrap { text-align: center; }
    .gauge-label {
        font-family: var(--mono);
        font-size: 10px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 8px;
    }

    /* ── Warning banner ── */
    .guard-warning {
        background: rgba(255,71,87,0.08);
        border: 1px solid rgba(255,71,87,0.3);
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 12px;
        font-family: var(--mono);
        font-size: 12px;
        color: var(--signal-critical);
    }
    .guard-verified {
        background: rgba(0,217,163,0.08);
        border: 1px solid rgba(0,217,163,0.3);
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 12px;
        font-family: var(--mono);
        font-size: 12px;
        color: var(--signal-verified);
    }

    .section-divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 28px 0;
    }

    h1, h2, h3 { color: var(--text-primary) !important; font-family: var(--sans) !important; }
    p, span, div { color: var(--text-primary); }
    .stMarkdown { color: var(--text-primary); }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-panel);
        border-right: 1px solid var(--border);
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD ENGINE — cached so it loads only once
# ──────────────────────────────────────────────
@st.cache_resource
def load_copilot():
    return ThreatCopilot()

try:
    copilot = load_copilot()
    engine_loaded = True
except Exception as e:
    engine_loaded = False
    load_error = str(e)


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown(f"""
<div class="console-header">
    <div>
        <div class="console-title"><span class="glyph">◈</span> THREAT COPILOT</div>
        <div class="console-subtitle">Cyber incident intelligence · {copilot.stats['total_incidents'] if engine_loaded else '—'} indexed records · 2021–2025</div>
    </div>
    <div class="status-pill">● KNOWLEDGE BASE ONLINE</div>
</div>
""", unsafe_allow_html=True)

if not engine_loaded:
    st.error(f"Engine failed to load: {load_error}")
    st.info("Make sure you've run `python ingest.py` first and your `.env` has a valid GROQ_API_KEY.")
    st.stop()


# ──────────────────────────────────────────────
# STAT STRIP
# ──────────────────────────────────────────────
stats = copilot.stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-label">Total Incidents</div>
        <div class="stat-value">{stats['total_incidents']:,}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    top_attacker = list(stats['top_attacker_countries'].keys())[0] if stats['top_attacker_countries'] else "—"
    st.markdown(f"""<div class="stat-card">
        <div class="stat-label">Top Origin Country</div>
        <div class="stat-value" style="font-size:18px">{top_attacker}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    top_sector = list(stats['top_target_sectors'].keys())[0] if stats['top_target_sectors'] else "—"
    st.markdown(f"""<div class="stat-card">
        <div class="stat-label">Most Targeted Sector</div>
        <div class="stat-value" style="font-size:18px">{top_sector}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-label">Zero-Day Incidents</div>
        <div class="stat-value critical">{stats.get('zero_day_count', 0)}</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TABS — Query console / Analytics
# ──────────────────────────────────────────────
tab1, tab2 = st.tabs(["◈  QUERY CONSOLE", "▤  ANALYTICS"])

with tab1:
    st.markdown('<div class="query-label">▸ ANALYST QUERY</div>', unsafe_allow_html=True)

    if "query_text" not in st.session_state:
        st.session_state.query_text = ""

    st.markdown('<div class="chip-row">', unsafe_allow_html=True)
    chip1, chip2, chip3, chip4 = st.columns(4)
    with chip1:
        if st.button("Chinese APT on energy", use_container_width=True):
            st.session_state.query_text = "Show me Chinese state-sponsored attacks on the energy sector"
    with chip2:
        if st.button("Zero-day exploits", use_container_width=True):
            st.session_state.query_text = "What attacks involved zero-day exploits?"
    with chip3:
        if st.button("Ransomware patterns", use_container_width=True):
            st.session_state.query_text = "What are the common patterns in ransomware and disruption attacks?"
    with chip4:
        if st.button("Critical infra attacks", use_container_width=True):
            st.session_state.query_text = "Summarize attacks on critical infrastructure targets"

    query = st.text_input(
        "query",
        value=st.session_state.query_text,
        placeholder="e.g. Which threat actors targeted financial institutions in 2024?",
        label_visibility="collapsed"
    )

    run = st.button("▸ RUN ANALYSIS", type="primary")

    if run and query.strip():
        with st.spinner("Searching knowledge base and generating analysis..."):
            answer, sources, guard = copilot.ask(query)

        # ── Case file result ──
        st.markdown(f"""
        <div class="case-file">
            <div class="case-file-header">
                <span>CASE FILE · QUERY: "{query[:60]}{'...' if len(query) > 60 else ''}"</span>
                <span>{len(sources)} RECORDS RETRIEVED</span>
            </div>
            <div class="case-file-body">{answer.replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Hallucination guard gauge + sources ──
        gcol1, gcol2 = st.columns([1, 3])

        with gcol1:
            conf = guard['confidence']
            color = "#00D9A3" if conf >= 90 else "#FFA94D" if conf >= 60 else "#FF4757"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf,
                number={'suffix': "%", 'font': {'color': color, 'family': 'JetBrains Mono', 'size': 28}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': "#1B2430"},
                    'bar': {'color': color, 'thickness': 0.25},
                    'bgcolor': "#161E29",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 60], 'color': 'rgba(255,71,87,0.1)'},
                        {'range': [60, 90], 'color': 'rgba(255,169,77,0.1)'},
                        {'range': [90, 100], 'color': 'rgba(0,217,163,0.1)'}
                    ],
                }
            ))
            fig.update_layout(
                height=180,
                margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#7B8794", 'family': 'JetBrains Mono'}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="gauge-label">GROUNDING CONFIDENCE</div>', unsafe_allow_html=True)

            if guard['verified']:
                st.markdown('<div class="guard-verified">✓ ALL NAMED ENTITIES VERIFIED IN SOURCE RECORDS</div>', unsafe_allow_html=True)
            else:
                terms = ", ".join(guard['unverified_terms'][:5])
                st.markdown(f'<div class="guard-warning">⚠ UNVERIFIED TERMS: {terms}</div>', unsafe_allow_html=True)

        with gcol2:
            st.markdown('<div class="query-label">▸ SOURCE INCIDENTS</div>', unsafe_allow_html=True)
            for s in sources:
                meta = s['metadata']
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-card-name">{meta.get('name', 'Unknown incident')}</div>
                    <div class="source-meta">
                        <span>◈ {meta.get('initiator_country', 'Unknown')}</span>
                        <span>→ {meta.get('receiver_country', 'Unknown')}</span>
                        <span>⊞ {meta.get('receiver_category', 'Unknown')}</span>
                        <span>◷ {meta.get('start_date', 'Unknown')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    elif run:
        st.warning("Enter a query first.")

with tab2:
    st.markdown('<div class="query-label">▸ THREAT LANDSCAPE OVERVIEW</div>', unsafe_allow_html=True)

    acol1, acol2 = st.columns(2)

    with acol1:
        attacker_df = pd.DataFrame(
            list(stats['top_attacker_countries'].items()),
            columns=['Country', 'Incidents']
        )
        fig1 = px.bar(
            attacker_df, x='Incidents', y='Country', orientation='h',
            title="Top Attack Origin Countries"
        )
        fig1.update_traces(marker_color='#00D9A3')
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#E4E8ED', 'family': 'Inter'},
            title_font={'size': 14, 'family': 'JetBrains Mono'},
            yaxis={'categoryorder': 'total ascending'},
            height=380, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig1, use_container_width=True)

    with acol2:
        sector_df = pd.DataFrame(
            list(stats['top_target_sectors'].items()),
            columns=['Sector', 'Incidents']
        )
        fig2 = px.bar(
            sector_df, x='Incidents', y='Sector', orientation='h',
            title="Most Targeted Sectors"
        )
        fig2.update_traces(marker_color='#FFA94D')
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#E4E8ED', 'family': 'Inter'},
            title_font={'size': 14, 'family': 'JetBrains Mono'},
            yaxis={'categoryorder': 'total ascending'},
            height=380, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    country_df = pd.DataFrame(
        list(stats['top_target_countries'].items()),
        columns=['Country', 'Incidents']
    )
    fig3 = px.bar(
        country_df, x='Country', y='Incidents',
        title="Most Targeted Countries"
    )
    fig3.update_traces(marker_color='#00D9A3')
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#E4E8ED', 'family': 'Inter'},
        title_font={'size': 14, 'family': 'JetBrains Mono'},
        height=340, margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; font-family:'JetBrains Mono'; font-size:11px; color:#7B8794;">
    THREAT COPILOT · RAG PIPELINE · FAISS · LLAMA 3 VIA GROQ · HALLUCINATION-GUARDED · STREAMLIT
</div>
""", unsafe_allow_html=True)
