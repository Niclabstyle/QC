import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
import py3Dmol
import numpy as np
import pandas as pd
import altair as alt
import base64
import os

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="PharmaChem AI | Quantum & Green Cheminformatics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. INYECCIÓN DE TIPOGRAFÍA PERSONALIZADA (NCL Wicky Bejari)
def get_font_base64(font_path):
    if os.path.exists(font_path):
        try:
            with open(font_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""

font_base64 = get_font_base64("NCLWickyBejari-Regular.ttf")
font_face_rule = f"""
@font-face {{
    font-family: 'WickyBejari';
    src: url(data:font/ttf;charset=utf-8;base64,{font_base64}) format('truetype');
    font-weight: normal;
    font-style: normal;
}}
""" if font_base64 else ""

# 3. SISTEMA DE DISEÑO CSS ULTRA-SMOOTH & GLASSMORPHISM
st.markdown(f"""
<style>
    {font_face_rule}
    @import url('https://fonts.googleapis.com/css2?family=Special+Elite&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {{
        --bg-deep: #01120e;
        --pine-dark: #02221b;
        --pine-main: #064E3B;
        --olive-mid: #556B2F;
        --lime-neon: #84CC16;
        --mint-soft: #D1FAE5;
        --mint-pale: #E6F4EA;
    }}

    .stApp {{
        background: radial-gradient(circle at 15% 10%, #064E3B 0%, #021a14 45%, #010c09 100%) !important;
        color: var(--mint-pale) !important;
        font-family: 'Inter', sans-serif !important;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #021712 0%, #010a08 100%) !important;
        border-right: 1px solid rgba(132, 204, 22, 0.2) !important;
    }}

    .brand-title {{
        font-family: 'WickyBejari', 'Special Elite', serif !important;
        font-size: 3rem !important;
        background: linear-gradient(90deg, #84CC16 0%, #D1FAE5 70%, #6B8E23 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.04em;
        line-height: 1.1;
        margin-bottom: 6px;
    }}

    .sec-header {{
        font-family: 'WickyBejari', 'Special Elite', serif !important;
        font-size: 1.4rem !important;
        color: #84CC16 !important;
        letter-spacing: 0.03em;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .hero-box {{
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.5) 0%, rgba(85, 107, 47, 0.2) 100%);
        border: 1px solid rgba(132, 204, 22, 0.3);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 22px;
        backdrop-filter: blur(14px);
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5);
    }}

    .hero-desc {{
        color: var(--mint-soft) !important;
        font-size: 1rem;
        line-height: 1.55;
    }}

    .badge-pill {{
        display: inline-block;
        background: rgba(132, 204, 22, 0.12);
        border: 1px solid #84CC16;
        color: #84CC16;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }}

    .kpi-card {{
        background: linear-gradient(150deg, rgba(6, 78, 59, 0.65) 0%, rgba(2, 22, 17, 0.85) 100%);
        border: 1px solid rgba(209, 250, 229, 0.15);
        border-top: 3px solid #84CC16;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
        transition: all 0.2s ease;
    }}

    .kpi-card:hover {{
        border-color: #84CC16;
        transform: translateY(-2px);
    }}

    .kpi-label {{
        color: var(--mint-soft);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        opacity: 0.8;
        margin-bottom: 4px;
    }}

    .kpi-val {{
        font-family: 'WickyBejari', 'Special Elite', serif;
        font-size: 1.85rem;
        color: #84CC16;
        line-height: 1.1;
    }}

    .kpi-sub {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--mint-pale);
        opacity: 0.65;
        margin-top: 3px;
    }}

    .specimen-bar {{
        background: rgba(2, 22, 17, 0.8);
        border: 1px solid rgba(132, 204, 22, 0.25);
        border-left: 5px solid #84CC16;
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }}

    .smiles-tag {{
        font-family: 'JetBrains Mono', monospace;
        background: rgba(209, 250, 229, 0.08);
        color: var(--mint-soft);
        padding: 5px 12px;
        border-radius: 6px;
        border: 1px solid rgba(209, 250, 229, 0.2);
        font-size: 0.85rem;
    }}

    .glass-panel {{
        background: rgba(6, 78, 59, 0.3);
        border: 1px solid rgba(209, 250, 229, 0.12);
        border-radius: 14px;
        padding: 18px;
        backdrop-filter: blur(8px);
        margin-bottom: 15px;
    }}

    .proto-card {{
        background: linear-gradient(145deg, rgba(6, 78, 59, 0.55), rgba(85, 107, 47, 0.2));
        border: 1px solid rgba(132, 204, 22, 0.2);
        border-radius: 12px;
        padding: 16px;
        height: 100%;
    }}

    .proto-head {{
        font-family: 'WickyBejari', 'Special Elite', serif;
        color: #84CC16;
        font-size: 1.1rem;
        margin-bottom: 6px;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: rgba(2, 18, 13, 0.5);
        padding: 5px;
        border-radius: 10px;
        border: 1px solid rgba(132, 204, 22, 0.18);
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 7px !important;
        color: var(--mint-soft) !important;
        font-weight: 500 !important;
        padding: 7px 14px !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: #064E3B !important;
        color: #84CC16 !important;
        border: 1px solid #84CC16 !important;
    }}

    div[data-baseweb="select"] > div, input {{
        background-color: #021a14 !important;
        color: #E6F4EA !important;
        border-color: rgba(132, 204, 22, 0.3) !important;
    }}

    p, span, label, div, .stMarkdown, .stText {{
        color: #E6F4EA !important;
    }}

    .stDataFrame, .dataframe, th, td {{
        color: #E6F4EA !important;
        background-color: transparent !important;
    }}
</style>
""", unsafe_allow_html=True)

# 4. HERO SECTION
st.markdown("""
<div class="hero-box">
    <div class="badge-pill">🌿 Green Lab & Quantum Core v5.2</div>
    <div class="brand-title">PharmaChem AI Studio</div>
    <div class="hero-desc
