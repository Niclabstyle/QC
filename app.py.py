import base64
import json
import os
import re
from urllib.parse import quote

import altair as alt
import numpy as np
import pandas as pd
import py3Dmol
import requests
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, FilterCatalog, QED, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

try:
    from streamlit_ketcher import st_ketcher

    KETCHER_AVAILABLE = True
except ImportError:
    KETCHER_AVAILABLE = False

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="PharmaChem AI Studio | Quantum & Green Cheminformatics",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2. TIPOGRAFÍA PERSONALIZADA (Opcional si existe el archivo .ttf)
def get_font_base64(font_path: str) -> str:
    if os.path.exists(font_path):
        try:
            with open(font_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


font_base64 = get_font_base64("NCLWickyBejari-Regular.ttf")
font_face_rule = (
    f"""
@font-face {{
    font-family: 'WickyBejari';
    src: url(data:font/ttf;charset=utf-8;base64,{font_base64}) format('truetype');
    font-weight: normal;
    font-style: normal;
}}
"""
    if font_base64
    else ""
)


# 3. ICONOS 3D GLASSMORPHISM, SPARKLINES Y HOLOGRAMAS ISOMÉTRICOS SVG
def svg_hero_emblem() -> str:
    return (
        "<svg width='52' height='52' viewBox='0 0 64 64' fill='none' xmlns='http://www.w3.org/2000/svg'>"
        "<defs>"
        "<linearGradient id='hg1' x1='0' y1='0' x2='64' y2='64' gradientUnits='userSpaceOnUse'>"
        "<stop offset='0%' stop-color='#134E3A' stop-opacity='0.95'/>"
        "<stop offset='100%' stop-color='#021712' stop-opacity='0.95'/>"
        "</linearGradient>"
        "<linearGradient id='hg2' x1='12' y1='12' x2='52' y2='52' gradientUnits='userSpaceOnUse'>"
        "<stop offset='0%' stop-color='#84CC16'/>"
        "<stop offset='100%' stop-color='#38BDF8'/>"
        "</linearGradient>"
        "</defs>"
        "<rect x='2' y='2' width='60' height='60' rx='16' fill='url(#hg1)' stroke='rgba(132,204,22,0.45)' stroke-width='1.5'/>"
        "<path d='M18 44C18 28 30 16 46 16C46 32 34 44 18 44Z' fill='rgba(132,204,22,0.18)' stroke='url(#hg2)' stroke-width='2'/>"
        "<path d='M20 42L42 20' stroke='#D1FAE5' stroke-width='1.6' stroke-linecap='round' opacity='0.8'/>"
        "<circle cx='42' cy='20' r='3.5' fill='#84CC16'/>"
        "<circle cx='28' cy='34' r='2.5' fill='#38BDF8'/>"
        "</svg>"
    )


def svg_kpi_badge(kind: str) -> str:
    icons = {
        "mw": "<path d='M9 7v10M15 7v10M6 12h12' stroke='#D1FAE5' stroke-width='1.8' stroke-linecap='round'/><circle cx='9' cy='9' r='1.5' fill='#84CC16'/><circle cx='15' cy='15' r='1.5' fill='#84CC16'/>",
        "logp": "<path d='M7 17L17 7M9 7h8v8' stroke='#D1FAE5' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/><circle cx='8' cy='16' r='2' fill='#84CC16'/>",
        "esol": "<path d='M12 5C12 5 7 10.5 7 14a5 5 0 0 0 10 0c0-3.5-5-9-5-9z' fill='rgba(56,189,248,0.25)' stroke='#38BDF8' stroke-width='1.7'/>",
        "qed": "<ellipse cx='12' cy='12' rx='7' ry='3' transform='rotate(35 12 12)' stroke='#84CC16' stroke-width='1.4'/><ellipse cx='12' cy='12' rx='7' ry='3' transform='rotate(-35 12 12)' stroke='#D1FAE5' stroke-width='1.4'/><circle cx='12' cy='12' r='2' fill='#84CC16'/>",
    }
    inner = icons.get(kind, icons["qed"])
    return (
        "<svg width='30' height='30' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'>"
        "<rect x='1' y='1' width='22' height='22' rx='7' fill='rgba(209,250,229,0.08)' stroke='rgba(209,250,229,0.22)' stroke-width='1'/>"
        f"{inner}</svg>"
    )


def svg_sparkline(color: str = "#84CC16", trend: str = "up") -> str:
    pts = "2,18 14,15 24,17 36,10 48,12 62,4" if trend == "up" else "2,6 14,9 26,7 38,14 50,12 62,17"
    end_x, end_y = ("62", "4") if trend == "up" else ("62", "17")
    return (
        "<svg width='68' height='22' viewBox='0 0 66 22' fill='none' xmlns='http://www.w3.org/2000/svg'>"
        f"<polyline points='{pts}' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/>"
        f"<circle cx='{end_x}' cy='{end_y}' r='2.8' fill='{color}'/>"
        "</svg>"
    )


def svg_hologram(kind: str) -> str:
    pedestal = (
        "<polygon points='40,68 70,53 40,38 10,53' fill='rgba(132,204,22,0.12)' stroke='rgba(132,204,22,0.45)' stroke-width='1.2'/>"
        "<polygon points='40,74 70,59 70,53 40,68' fill='rgba(6,78,59,0.7)'/>"
        "<polygon points='40,74 10,59 10,53 40,68' fill='rgba(2,22,17,0.85)'/>"
    )
    if kind == "cube":
        body = (
            "<polygon points='40,12 62,23 40,34 18,23' fill='rgba(209,250,229,0.22)' stroke='#D1FAE5' stroke-width='1.2'/>"
            "<polygon points='40,34 62,23 62,47 40,58' fill='rgba(132,204,22,0.18)' stroke='#84CC16' stroke-width='1.2'/>"
            "<polygon points='40,34 18,23 18,47 40,58' fill='rgba(56,189,248,0.15)' stroke='#38BDF8' stroke-width='1.2'/>"
            "<circle cx='40' cy='34' r='3' fill='#84CC16'/><circle cx='29' cy='28' r='2' fill='#D1FAE5'/><circle cx='51' cy='28' r='2' fill='#38BDF8'/>"
        )
    elif kind == "wave":
        body = (
            "<path d='M18 42 Q29 14 40 34 T62 24' fill='none' stroke='#84CC16' stroke-width='2.2' stroke-linecap='round'/>"
            "<path d='M18 46 Q29 22 40 38 T62 30' fill='none' stroke='#38BDF8' stroke-width='1.8' stroke-linecap='round' opacity='0.8'/>"
            "<ellipse cx='40' cy='32' rx='18' ry='8' transform='rotate(-20 40 32)' fill='rgba(209,250,229,0.08)' stroke='rgba(209,250,229,0.5)' stroke-width='1.1'/>"
            "<circle cx='40' cy='32' r='3' fill='#D1FAE5'/>"
        )
    else:
        body = (
            "<path d='M40 12 L58 20 V35 C58 46 49 53 40 57 C31 53 22 46 22 35 V20 Z' fill='rgba(132,204,22,0.16)' stroke='#84CC16' stroke-width='1.5'/>"
            "<path d='M33 34 L38 39 L48 28' fill='none' stroke='#D1FAE5' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'/>"
        )
    return (
        "<svg width='80' height='78' viewBox='0 0 80 78' fill='none' xmlns='http://www.w3.org/2000/svg' "
        "style='flex-shrink:0; filter: drop-shadow(0 0 10px rgba(132,204,22,0.28));'>"
        f"{pedestal}{body}</svg>"
    )


def svg_section_icon(kind: str) -> str:
    paths = {
        "atom": "<circle cx='12' cy='12' r='2.2' fill='#84CC16'/><ellipse cx='12' cy='12' rx='9' ry='3.5' transform='rotate(30 12 12)' stroke='#D1FAE5' stroke-width='1.4'/><ellipse cx='12' cy='12' rx='9' ry='3.5' transform='rotate(-30 12 12)' stroke='#84CC16' stroke-width='1.4'/>",
        "search": "<circle cx='11' cy='11' r='6.5' stroke='#84CC16' stroke-width='1.8'/><path d='M16 16l4.5 4.5' stroke='#D1FAE5' stroke-width='1.8' stroke-linecap='round'/>",
        "cube": "<path d='M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z' stroke='#84CC16' stroke-width='1.6'/><path d='M12 12l8-4.5M12 12v9M12 12L4 7.5' stroke='#D1FAE5' stroke-width='1.3'/>",
        "shield": "<path d='M12 3l7 3.5v5c0 4.5-3 7.8-7 9.5-4-1.7-7-5-7-9.5v-5L12 3z' stroke='#84CC16' stroke-width='1.6' fill='rgba(132,204,22,0.1)'/>",
    }
    return f"<svg width='20' height='20' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'>{paths.get(kind, paths['atom'])}</svg>"


def svg_data_uri(path_markup: str) -> str:
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none'>{path_markup}</svg>"
    return f"data:image/svg+xml;utf8,{quote(svg)}"


ICON_DASH = svg_data_uri("<path d='M3 10.5L12 3l9 7.5V20a1 1 0 0 1-1 1h-5.5v-6h-5v6H4a1 1 0 0 1-1-1v-9.5z' stroke='%2384CC16' stroke-width='1.8' stroke-linejoin='round'/>")
ICON_MOL = svg_data_uri("<circle cx='12' cy='12' r='2.5' fill='%2384CC16'/><circle cx='6' cy='7' r='2' stroke='%23D1FAE5' stroke-width='1.5'/><circle cx='18' cy='7' r='2' stroke='%23D1FAE5' stroke-width='1.5'/><circle cx='12' cy='19' r='2' stroke='%2338BDF8' stroke-width='1.5'/><path d='M7.5 8.5L10.2 10.5M16.5 8.5L13.8 10.5M12 14.5V17' stroke='%23D1FAE5' stroke-width='1.4'/>")
ICON_ORB = svg_data_uri("<circle cx='12' cy='12' r='2.2' fill='%2384CC16'/><ellipse cx='12' cy='12' rx='9' ry='3.5' transform='rotate(35 12 12)' stroke='%23D1FAE5' stroke-width='1.5'/><ellipse cx='12' cy='12' rx='9' ry='3.5' transform='rotate(-35 12 12)' stroke='%2338BDF8' stroke-width='1.5'/>")
ICON_WAVE = svg_data_uri("<path d='M3 12c2.5-5 4.5-5 7 0s4.5 5 7 0 2.5-3 4-3' stroke='%2384CC16' stroke-width='1.8' stroke-linecap='round'/><path d='M3 16c2.5-3 4.5-3 7 0s4.5 3 7 0' stroke='%2338BDF8' stroke-width='1.4' stroke-linecap='round'/>")
ICON_SHIELD = svg_data_uri("<path d='M12 3l7 3.5v5c0 4.5-3 7.8-7 9.5-4-1.7-7-5-7-9.5v-5L12 3z' stroke='%2384CC16' stroke-width='1.7' fill='rgba(132,204,22,0.12)'/><path d='M9 12l2 2 4-4' stroke='%23D1FAE5' stroke-width='1.6' stroke-linecap='round'/>")


# 4. SISTEMA DE DISEÑO CSS GLASSMORPHISM 3D + BOTONES LATERALES ESTILO MOCKUP
st.markdown(
    f"""
<style>
    {font_face_rule}
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {{
        --bg-deep: #010c09;
        --pine-dark: #021a14;
        --pine-main: #064E3B;
        --lime-neon: #84CC16;
        --cyan-soft: #38BDF8;
        --mint-soft: #D1FAE5;
        --mint-pale: #E6F4EA;
    }}

    .stApp {{
        background: radial-gradient(circle at 18% 8%, #074736 0%, #021a14 42%, #010b08 100%) !important;
        color: var(--mint-pale);
        font-family: 'Inter', sans-serif !important;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #031510 0%, #010907 100%) !important;
        border-right: 1px solid rgba(132, 204, 22, 0.18) !important;
    }}

    /* Botones de navegación en Sidebar idénticos al mockup */
    [data-testid="stSidebar"] .stButton > button {{
        width: 100% !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }}

    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(209, 250, 229, 0.06) !important;
        color: rgba(209, 250, 229, 0.78) !important;
    }}

    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
        background: rgba(132, 204, 22, 0.08) !important;
        border-color: rgba(132, 204, 22, 0.3) !important;
        color: #FFFFFF !important;
    }}

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, rgba(132, 204, 22, 0.2) 0%, rgba(6, 78, 59, 0.45) 100%) !important;
        border: 1px solid rgba(132, 204, 22, 0.6) !important;
        color: #84CC16 !important;
        box-shadow: 0 0 16px rgba(132, 204, 22, 0.16) !important;
    }}

    [data-testid="stSidebar"] .stButton > button::before {{
        content: "";
        width: 18px;
        height: 18px;
        display: inline-block;
        background-size: contain;
        background-repeat: no-repeat;
        flex-shrink: 0;
    }}

    .st-key-nav_dash button::before {{ background-image: url("{ICON_DASH}"); }}
    .st-key-nav_mol button::before {{ background-image: url("{ICON_MOL}"); }}
    .st-key-nav_qm button::before {{ background-image: url("{ICON_ORB}"); }}
    .st-key-nav_spec button::before {{ background-image: url("{ICON_WAVE}"); }}
    .st-key-nav_adme button::before {{ background-image: url("{ICON_SHIELD}"); }}

    .hero-bar {{
        background: linear-gradient(135deg, rgba(8, 66, 51, 0.58) 0%, rgba(2, 24, 18, 0.85) 100%);
        border: 1px solid rgba(209, 250, 229, 0.16);
        border-radius: 18px;
        padding: 18px 26px;
        margin-bottom: 18px;
        backdrop-filter: blur(18px);
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(209, 250, 229, 0.14);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        flex-wrap: wrap;
    }}

    .hero-left {{
        display: flex;
        align-items: center;
        gap: 16px;
    }}

    .brand-title {{
        font-family: 'WickyBejari', 'Inter', sans-serif !important;
        font-size: 2.05rem !important;
        font-weight: 700;
        background: linear-gradient(90deg, #FFFFFF 0%, #D1FAE5 55%, #84CC16 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.01em;
        line-height: 1.1;
        margin: 0;
    }}

    .hero-sub {{
        color: rgba(209, 250, 229, 0.78);
        font-size: 0.88rem;
        margin-top: 4px;
    }}

    .badge-pill {{
        background: rgba(132, 204, 22, 0.12);
        border: 1px solid rgba(132, 204, 22, 0.45);
        color: #84CC16 !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 6px 14px;
        border-radius: 999px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        box-shadow: 0 0 18px rgba(132, 204, 22, 0.15);
    }}

    .sec-header {{
        font-size: 1.05rem !important;
        font-weight: 600;
        color: #E6F4EA !important;
        letter-spacing: 0.01em;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 14px;
        margin-bottom: 18px;
    }}

    .kpi-card {{
        background: linear-gradient(155deg, rgba(9, 64, 49, 0.55) 0%, rgba(2, 20, 15, 0.9) 100%);
        border: 1px solid rgba(209, 250, 229, 0.14);
        border-radius: 14px;
        padding: 15px 18px;
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(209, 250, 229, 0.12);
        transition: all 0.22s ease;
    }}

    .kpi-card:hover {{
        border-color: rgba(132, 204, 22, 0.55);
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.55), 0 0 18px rgba(132, 204, 22, 0.12);
    }}

    .kpi-top {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }}

    .kpi-label {{
        color: rgba(209, 250, 229, 0.8);
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }}

    .kpi-bottom {{
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 8px;
    }}

    .kpi-val {{
        font-size: 1.65rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1;
    }}

    .kpi-unit {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #84CC16;
        margin-top: 4px;
    }}

    .specimen-bar {{
        background: linear-gradient(90deg, rgba(6, 78, 59, 0.4) 0%, rgba(2, 22, 17, 0.85) 100%);
        border: 1px solid rgba(132, 204, 22, 0.25);
        border-left: 4px solid #84CC16;
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }}

    .smiles-tag {{
        font-family: 'JetBrains Mono', monospace;
        background: rgba(209, 250, 229, 0.07);
        color: var(--mint-soft);
        padding: 5px 12px;
        border-radius: 8px;
        border: 1px solid rgba(209, 250, 229, 0.18);
        font-size: 0.82rem;
    }}

    .glass-panel {{
        background: rgba(6, 78, 59, 0.24);
        border: 1px solid rgba(209, 250, 229, 0.12);
        border-radius: 14px;
        padding: 14px 16px;
        backdrop-filter: blur(10px);
        margin-bottom: 14px;
    }}

    .proto-card {{
        background: linear-gradient(145deg, rgba(7, 58, 44, 0.52) 0%, rgba(2, 20, 15, 0.88) 100%);
        border: 1px solid rgba(209, 250, 229, 0.14);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(209, 250, 229, 0.12);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        min-height: 112px;
        transition: all 0.22s ease;
    }}

    .proto-card:hover {{
        border-color: rgba(132, 204, 22, 0.45);
        transform: translateY(-2px);
    }}

    .proto-head {{
        color: #FFFFFF;
        font-weight: 600;
        font-size: 0.98rem;
        margin-bottom: 5px;
    }}

    .proto-text {{
        font-size: 0.8rem;
        color: rgba(209, 250, 229, 0.78);
        line-height: 1.42;
        margin: 0;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: rgba(2, 18, 13, 0.65);
        padding: 5px;
        border-radius: 12px;
        border: 1px solid rgba(209, 250, 229, 0.12);
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        color: rgba(209, 250, 229, 0.78) !important;
        font-weight: 500 !important;
        font-size: 0.86rem !important;
        padding: 7px 14px !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: rgba(132, 204, 22, 0.16) !important;
        color: #FFFFFF !important;
        border: 1px solid #84CC16 !important;
        box-shadow: 0 0 14px rgba(132, 204, 22, 0.22) !important;
    }}

    div[data-baseweb="select"] > div, input {{
        background-color: #021a14 !important;
        color: #E6F4EA !important;
        border-color: rgba(132, 204, 22, 0.28) !important;
    }}

    .stMarkdown p, .stMarkdown li, label[data-testid="stWidgetLabel"] p {{
        color: #E6F4EA;
    }}
</style>
""",
    unsafe_allow_html=True,
)


# 5. FUNCIONES DE CÁLCULO Y API CACHEADAS
@st.cache_resource(show_spinner=False)
def obtener_catalogo_alertas():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


@st.cache_data(ttl=3600, show_spinner=False)
def buscar_pubchem(nombre: str):
    if not nombre.strip():
        return None
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(nombre.strip())}"
        "/property/CanonicalSMILES,IUPACName,MolecularFormula/JSON"
    )
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            return None
        props = resp.json().get("PropertyTable", {}).get("Properties", [])
        if not props:
            return None
        item = props[0]
        smiles = item.get("CanonicalSMILES") or item.get("ConnectivitySMILES")
        if not smiles:
            return None
        return {
            "smiles": smiles,
            "iupac": item.get("IUPACName", nombre),
            "formula": item.get("MolecularFormula", ""),
            "cid": item.get("CID", ""),
        }
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def calcular_huckel_tom(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    pi_atoms, pi_electrons, h_params = [], 0, []
    for atom in mol.GetAtoms():
        hyb = atom.GetHybridization()
        if hyb not in (Chem.HybridizationType.SP2, Chem.HybridizationType.SP):
            continue
        sym = atom.GetSymbol()
        if sym == "C":
            n_e, h_x = 1, 0.0
        elif sym == "N":
            if atom.GetIsAromatic() and (atom.GetTotalNumHs() > 0 or atom.GetDegree() == 3):
                n_e, h_x = 2, 1.5
            elif not atom.GetIsAromatic() and not any(
                b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE) for b in atom.GetBonds()
            ):
                n_e, h_x = 2, 1.5
            else:
                n_e, h_x = 1, 0.5
        elif sym == "O":
            n_e, h_x = (1, 1.0) if atom.GetDegree() == 1 else (2, 2.0)
        elif sym == "S":
            n_e, h_x = (1, 0.5) if atom.GetDegree() == 1 else (2, 1.0)
        else:
            continue

        pi_atoms.append(atom.GetIdx())
        pi_electrons += n_e
        h_params.append(h_x)

    n = len(pi_atoms)
    if n < 2:
        return None

    idx_map = {atom_idx: i for i, atom_idx in enumerate(pi_atoms)}
    mat = np.zeros((n, n), dtype=float)
    for i, h_val in enumerate(h_params):
        mat[i, i] = h_val

    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in idx_map and a2 in idx_map:
            i, j = idx_map[a1], idx_map[a2]
            s1, s2 = mol.GetAtomWithIdx(a1).GetSymbol(), mol.GetAtomWithIdx(a2).GetSymbol()
            k_xy = 1.0 if (s1 == "C" and s2 == "C") else 0.85
            mat[i, j] = k_xy
            mat[j, i] = k_xy

    eigvals, eigvecs = np.linalg.eigh(-mat)
    alpha_ev, beta_abs_ev = -6.0, 2.4

    n_occ = min(n, (pi_electrons + 1) // 2)
    homo_idx = max(0, n_occ - 1)
    lumo_idx = min(n - 1, homo_idx + 1)

    orbitales = []
    for k in range(n):
        e_beta = float(eigvals[k])
        e_ev = round(alpha_ev + e_beta * beta_abs_ev, 3)
        if k < homo_idx:
            occ, tipo, spin_txt = 2, "Ocupado (π)", "↑↓"
        elif k == homo_idx:
            occ = 1 if pi_electrons % 2 == 1 else 2
            tipo = "HOMO"
            spin_txt = "↑" if occ == 1 else "↑↓"
        elif k == lumo_idx and lumo_idx > homo_idx:
            occ, tipo, spin_txt = 0, "LUMO", ""
        else:
            occ, tipo, spin_txt = 0, "Virtual (π*)", ""

        orbitales.append(
            {
                "Orbital": f"Ψ{k + 1}",
                "Índice": k + 1,
                "Energía (|β|)": round(e_beta, 3),
                "Energía (eV)": e_ev,
                "Ocupación": occ,
                "Spin": spin_txt,
                "Tipo": tipo,
            }
        )

    homo_ev = orbitales[homo_idx]["Energía (eV)"]
    lumo_ev = orbitales[lumo_idx]["Energía (eV)"] if lumo_idx > homo_idx else homo_ev
    gap_ev = round(abs(lumo_ev - homo_ev), 3)
    gap_beta = round(abs(orbitales[lumo_idx]["Energía (|β|)"] - orbitales[homo_idx]["Energía (|β|)"]), 3)

    i_pot, e_aff = -homo_ev, -lumo_ev
    mu_chem = round(-(i_pot + e_aff) / 2.0, 3)
    eta_hard = round(max(0.01, (i_pot - e_aff) / 2.0), 3)
    omega_elec = round((mu_chem**2) / (2.0 * eta_hard), 3)

    homo_vec, lumo_vec = eigvecs[:, homo_idx], eigvecs[:, lumo_idx]
    densidad_atoms = [
        {
            "ID Átomo": a_idx,
            "Átomo": f"{mol.GetAtomWithIdx(a_idx).GetSymbol()}#{a_idx}",
            "c² (HOMO)": round(float(homo_vec[i] ** 2), 4),
            "c² (LUMO)": round(float(lumo_vec[i] ** 2), 4),
        }
        for i, a_idx in enumerate(pi_atoms)
    ]

    return {
        "n_pi_atoms": n,
        "pi_electrons": pi_electrons,
        "orbitales": orbitales,
        "homo_idx": homo_idx,
        "homo_ev": homo_ev,
        "lumo_ev": lumo_ev,
        "gap_ev": gap_ev,
        "gap_beta": gap_beta,
        "mu_chem": mu_chem,
        "eta_hard": eta_hard,
        "omega_elec": omega_elec,
        "densidad_atoms": densidad_atoms,
    }


def construir_diagrama_tom_correlacion(tom_res: dict):
    """Genera el diagrama de correlación LCAO-MO (AO izquierda -> MO centro -> AO derecha) igual al mockup."""
    orbs = tom_res["orbitales"]
    homo_i = tom_res["homo_idx"]
    # Ventana de hasta 8 orbitales frontera para mantener legibilidad visual con flechas y líneas
    start = max(0, homo_i - 3)
    end = min(len(orbs), start + 8)
    start = max(0, end - 8)
    ventana = orbs[start:end]

    e_media = float(np.mean([o["Energía (eV)"] for o in ventana]))
    e_ao_a = round(e_media + 0.35, 3)
    e_ao_b = round(e_media - 0.35, 3)

    barras = []
    correlaciones = []
    etiquetas = []

    # Niveles AO izquierda y derecha
    barras.append({"id": "AO_A", "x": 0.15, "y": e_ao_a, "Tipo": "Orbital Atómico"})
    barras.append({"id": "AO_A", "x": 0.85, "y": e_ao_a, "Tipo": "Orbital Atómico"})
    barras.append({"id": "AO_B", "x": 3.15, "y": e_ao_b, "Tipo": "Orbital Atómico"})
    barras.append({"id": "AO_B", "x": 3.85, "y": e_ao_b, "Tipo": "Orbital Atómico"})

    etiquetas.append({"x": 0.5, "y": e_ao_a, "texto": "φA (AO)", "Tipo": "Orbital Atómico"})
    etiquetas.append({"x": 3.5, "y": e_ao_b, "texto": "φB (AO)", "Tipo": "Orbital Atómico"})

    for idx, o in enumerate(ventana):
        e_val = o["Energía (eV)"]
        tipo = o["Tipo"]
        bar_id = f"MO_{idx}"
        # Barra horizontal del orbital molecular en el centro
        barras.append({"id": bar_id, "x": 1.65, "y": e_val, "Tipo": tipo})
        barras.append({"id": bar_id, "x": 2.35, "y": e_val, "Tipo": tipo})

        # Líneas punteadas de correlación desde AO_A (0.85) -> MO (1.65) y MO (2.35) -> AO_B (3.15)
        correlaciones.append({"id": f"CL_{idx}", "x": 0.85, "y": e_ao_a, "Tipo": tipo})
        correlaciones.append({"id": f"CL_{idx}", "x": 1.65, "y": e_val, "Tipo": tipo})
        correlaciones.append({"id": f"CR_{idx}", "x": 2.35, "y": e_val, "Tipo": tipo})
        correlaciones.append({"id": f"CR_{idx}", "x": 3.15, "y": e_ao_b, "Tipo": tipo})

        if o["Spin"]:
            etiquetas.append({"x": 2.0, "y": e_val, "texto": o["Spin"], "Tipo": tipo})
        if tipo in ("HOMO", "LUMO"):
            etiquetas.append({"x": 2.55, "y": e_val, "texto": f"{tipo} ({o['Orbital']})", "Tipo": tipo})

    df_barras = pd.DataFrame(barras)
    df_corr = pd.DataFrame(correlaciones)
    df_etq = pd.DataFrame(etiquetas)

    escala_color = alt.Scale(
        domain=["Ocupado (π)", "HOMO", "LUMO", "Virtual (π*)", "Orbital Atómico"],
        range=["#38BDF8", "#84CC16", "#F43F5E", "#A3E635", "#D1FAE5"],
    )

    capa_corr = (
        alt.Chart(df_corr)
        .mark_line(strokeDash=[4, 4], strokeWidth=1.2, opacity=0.65)
        .encode(
            x=alt.X("x:Q", title="", scale=alt.Scale(domain=[0, 4.1]), axis=None),
            y=alt.Y("y:Q", title="Energía (eV)", axis=alt.Axis(labelColor="#D1FAE5", titleColor="#84CC16", gridColor="rgba(132,204,22,0.08)")),
            detail="id:N",
            color=alt.Color("Tipo:N", scale=escala_color, legend=alt.Legend(labelColor="#E6F4EA", titleColor="#84CC16", orient="bottom")),
        )
    )

    capa_barras = (
        alt.Chart(df_barras)
        .mark_line(strokeWidth=3.2)
        .encode(
            x="x:Q",
            y="y:Q",
            detail="id:N",
            color=alt.Color("Tipo:N", scale=escala_color),
        )
    )

    capa_txt = (
        alt.Chart(df_etq)
        .mark_text(dy=-9, fontSize=11, fontWeight="bold")
        .encode(
            x="x:Q",
            y="y:Q",
            text="texto:N",
            color=alt.Color("Tipo:N", scale=escala_color),
        )
    )

    return (capa_corr + capa_barras + capa_txt).properties(height=255, background="transparent").configure_view(stroke=None)


def calcular_tcc(d_electrons: int, geometria: str, campo: str):
    if "Octaédrica" in geometria:
        if "Alto" in campo:
            t2g = min(3, d_electrons) + max(0, d_electrons - 5)
            eg = max(0, min(2, d_electrons - 3)) + max(0, d_electrons - 8)
        else:
            t2g = min(6, d_electrons)
            eg = max(0, d_electrons - 6)
        eecc = round(-0.4 * t2g + 0.6 * eg, 2)
        unpaired = (min(3, t2g) - max(0, t2g - 3)) + (min(2, eg) - max(0, eg - 2))
        niveles = [
            {"Nivel": "t2g (dxy, dxz, dyz)", "Energía": -0.4, "Electrones": t2g},
            {"Nivel": "eg (dx²-y², dz²)", "Energía": 0.6, "Electrones": eg},
        ]
        unidad = "Δo"
    elif "Tetraédrica" in geometria:
        e_orb = min(2, d_electrons) + max(0, min(2, d_electrons - 5))
        t2_orb = max(0, min(3, d_electrons - 2)) + max(0, d_electrons - 7)
        eecc = round(-0.6 * e_orb + 0.4 * t2_orb, 2)
        unpaired = (min(2, e_orb) - max(0, e_orb - 2)) + (min(3, t2_orb) - max(0, t2_orb - 3))
        niveles = [
            {"Nivel": "e (dx²-y², dz²)", "Energía": -0.6, "Electrones": e_orb},
            {"Nivel": "t2 (dxy, dxz, dyz)", "Energía": 0.4, "Electrones": t2_orb},
        ]
        unidad = "Δt"
    else:
        rem = d_electrons
        if "Bajo" in campo:
            n_xz_yz = min(4, rem); rem -= n_xz_yz
            n_z2 = min(2, rem); rem -= n_z2
            n_xy = min(2, rem); rem -= n_xy
            n_x2y2 = rem
        else:
            n_xz_yz = min(2, rem); rem -= n_xz_yz
            n_z2 = min(1, rem); rem -= n_z2
            n_xy = min(1, rem); rem -= n_xy
            n_x2y2 = min(1, rem); rem -= n_x2y2
            if rem > 0:
                add = min(2, rem); n_xz_yz += add; rem -= add
            if rem > 0:
                add = min(1, rem); n_z2 += add; rem -= add
            if rem > 0:
                add = min(1, rem); n_xy += add; rem -= add
            if rem > 0:
                n_x2y2 += rem

        eecc = round(-0.514 * n_xz_yz - 0.428 * n_z2 + 0.228 * n_xy + 1.228 * n_x2y2, 3)
        unpaired = (min(2, n_xz_yz) - max(0, n_xz_yz - 2)) + (n_z2 % 2) + (n_xy % 2) + (n_x2y2 % 2)
        niveles = [
            {"Nivel": "eg (dxz, dyz)", "Energía": -0.514, "Electrones": n_xz_yz},
            {"Nivel": "a1g (dz²)", "Energía": -0.428, "Electrones": n_z2},
            {"Nivel": "b2g (dxy)", "Energía": 0.228, "Electrones": n_xy},
            {"Nivel": "b1g (dx²-y²)", "Energía": 1.228, "Electrones": n_x2y2},
        ]
        unidad = "Δo"

    mu_b = round(float(np.sqrt(unpaired * (unpaired + 2))), 2)
    return {"eecc": eecc, "unidad": unidad, "unpaired": unpaired, "mu_b": mu_b, "niveles": niveles}


@st.cache_data(show_spinner=False)
def analizar_molecula(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    smiles_can = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    pm = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    anillos_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    qed_score = QED.qed(mol)

    n_heavy = mol.GetNumHeavyAtoms()
    n_arom_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    ap = (n_arom_atoms / n_heavy) if n_heavy > 0 else 0.0
    log_s = round(0.16 - 0.63 * logp - 0.0062 * pm + 0.066 * rot_bonds - 0.74 * ap, 2)
    sol_mg_ml = (10**log_s) * pm

    catalogo = obtener_catalogo_alertas()
    alertas = [m.GetDescription() for m in catalogo.GetMatches(mol)]

    mol_h = Chem.AddHs(mol)
    AllChem.ComputeGasteigerCharges(mol_h)

    cargas_esp, datos_atomos = [], []
    for atom in mol_h.GetAtoms():
        try:
            q = float(atom.GetProp("_GasteigerCharge"))
            q_safe = 0.0 if (np.isnan(q) or np.isinf(q)) else round(q, 4)
        except Exception:
            q_safe = 0.0
        cargas_esp.append(q_safe)

        if atom.GetSymbol() != "H":
            datos_atomos.append(
                {
                    "ID Átomo": atom.GetIdx(),
                    "Elemento": atom.GetSymbol(),
                    "Hibridación": str(atom.GetHybridization()),
                    "Aromático": "Sí" if atom.GetIsAromatic() else "No",
                    "Carga (e)": q_safe,
                }
            )

    embed_status = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    if embed_status == -1:
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDG(useRandomCoords=True))

    if AllChem.MMFFHasAllMoleculeParams(mol_h):
        AllChem.MMFFOptimizeMolecule(mol_h)
    else:
        AllChem.UFFOptimizeMolecule(mol_h)

    mol_block = Chem.MolToMolBlock(mol_h)

    smarts_checks = {
        "has_oh": mol.HasSubstructMatch(Chem.MolFromSmarts("[OX2H]")),
        "has_nh": mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")),
        "has_co": mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3]=[OX1]")),
        "has_cn": mol.HasSubstructMatch(Chem.MolFromSmarts("[CX2]#[NX1]")),
        "has_c_o": mol.HasSubstructMatch(Chem.MolFromSmarts("[CX4]-[OX2]")),
    }

    return {
        "smiles_can": smiles_can,
        "formula": formula,
        "pm": pm,
        "logp": logp,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "rot_bonds": rot_bonds,
        "anillos_arom": anillos_arom,
        "qed": qed_score,
        "log_s": log_s,
        "sol_mg_ml": sol_mg_ml,
        "alertas": alertas,
        "cargas_esp": cargas_esp,
        "datos_atomos": datos_atomos,
        "mol_block": mol_block,
        "smarts": smarts_checks,
    }


@st.cache_data(show_spinner=False)
def generar_svg_2d(smiles: str, highlight_idx: int = -1) -> str:
    mol_2d = Chem.MolFromSmiles(smiles)
    if mol_2d is None:
        return ""

    AllChem.Compute2DCoords(mol_2d)
    drawer = rdMolDraw2D.MolDraw2DSVG(420, 220)
    opts = drawer.drawOptions()

    if hasattr(rdMolDraw2D, "SetDarkMode"):
        rdMolDraw2D.SetDarkMode(opts)
    opts.setBackgroundColour((0.0, 0.0, 0.0, 0.0))
    opts.bondLineWidth = 2.0

    highlight_atoms = [highlight_idx] if 0 <= highlight_idx < mol_2d.GetNumAtoms() else []
    highlight_colors = {highlight_idx: (1.0, 0.0, 0.5, 0.65)} if highlight_atoms else {}

    drawer.DrawMolecule(mol_2d, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r"width='[^']+'\s+height='[^']+'", "style='width:100%; height:auto; max-height:200px;'", svg, count=1)
    return " ".join(line.strip() for line in svg.splitlines() if line.strip())


def calcular_tanimoto(smiles_a: str, smiles_b: str) -> float:
    m_a, m_b = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if m_a is None or m_b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(m_a, 2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(m_b, 2, nBits=2048)
    return round(DataStructs.TanimotoSimilarity(fp_a, fp_b), 3)


def top_similares_catalogo(smiles_objetivo: str, catalogo: dict, top_n: int = 2):
    m_obj = Chem.MolFromSmiles(smiles_objetivo)
    if m_obj is None:
        return []
    can_obj = Chem.MolToSmiles(m_obj)
    fp_obj = AllChem.GetMorganFingerprintAsBitVect(m_obj, 2, nBits=2048)

    resultados = []
    for sm, (nombre, _, formula) in catalogo.items():
        m_cat = Chem.MolFromSmiles(sm)
        if m_cat is None or Chem.MolToSmiles(m_cat) == can_obj:
            continue
        fp_cat = AllChem.GetMorganFingerprintAsBitVect(m_cat, 2, nBits=2048)
        resultados.append((nombre, formula, round(DataStructs.TanimotoSimilarity(fp_obj, fp_cat), 3)))

    resultados.sort(key=lambda x: x[2], reverse=True)
    return resultados[:top_n]


def construir_boiled_egg(tpsa_val: float, wlogp_val: float, nombre_mol: str):
    t = np.linspace(0, 2 * np.pi, 120)
    df_hia = pd.DataFrame({"TPSA": 71.0 + 71.0 * np.cos(t), "WLOGP": 2.3 + 4.1 * np.sin(t), "order": np.arange(len(t)), "Zona": "HIA (Absorción GI)"})
    df_bbb = pd.DataFrame({"TPSA": 38.0 + 41.0 * np.cos(t), "WLOGP": 3.1 + 2.6 * np.sin(t), "order": np.arange(len(t)), "Zona": "BBB (Cruce Cerebral)"})
    df_elipses = pd.concat([df_hia, df_bbb], ignore_index=True)
    df_punto = pd.DataFrame([{"TPSA": tpsa_val, "WLOGP": wlogp_val, "Compuesto": nombre_mol}])

    capa_elipses = (
        alt.Chart(df_elipses)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("TPSA:Q", title="TPSA (Å²)", scale=alt.Scale(domain=[0, 180]), axis=alt.Axis(labelColor="#D1FAE5", titleColor="#84CC16", gridColor="rgba(132,204,22,0.1)")),
            y=alt.Y("WLOGP:Q", title="WLOGP", scale=alt.Scale(domain=[-3, 7.5]), axis=alt.Axis(labelColor="#D1FAE5", titleColor="#84CC16", gridColor="rgba(132,204,22,0.1)")),
            color=alt.Color("Zona:N", scale=alt.Scale(domain=["HIA (Absorción GI)", "BBB (Cruce Cerebral)"], range=["#D1FAE5", "#facc15"]), legend=alt.Legend(labelColor="#E6F4EA", titleColor="#84CC16")),
            order="order:Q",
        )
    )
    capa_punto = (
        alt.Chart(df_punto)
        .mark_point(size=180, filled=True, color="#84CC16", stroke="#ffffff", strokeWidth=1.5)
        .encode(x="TPSA:Q", y="WLOGP:Q", tooltip=["Compuesto", "TPSA", "WLOGP"])
    )
    return (capa_elipses + capa_punto).properties(height=210, background="transparent").configure_view(stroke=None)


def construir_html_3d(
    mol_block: str,
    modo: str,
    cargas_esp: list,
    max_abs_q: float,
    selected_atom_id: int = -1,
    spin: bool = False,
    height: int = 360,
) -> str:
    viewer = py3Dmol.view(width="100%", height=height)
    viewer.addModel(mol_block, "mol")

    if "Esferas" in modo:
        viewer.setStyle({"stick": {"radius": 0.16, "colorscheme": "greenCarbon"}, "sphere": {"scale": 0.28, "colorscheme": "greenCarbon"}})
        if selected_atom_id != -1:
            serial_3dmol = selected_atom_id + 1
            viewer.addStyle({"serial": serial_3dmol}, {"sphere": {"scale": 0.58, "color": "#ff007f"}, "stick": {"radius": 0.24, "color": "#ff007f"}})
            viewer.addLabel(f"Atom #{selected_atom_id}", {"position": {"serial": serial_3dmol}, "backgroundColor": "#064E3B", "fontColor": "#84CC16", "fontSize": 13})
            viewer.zoomTo({"serial": serial_3dmol})
        else:
            viewer.zoomTo()
    elif "ESP" in modo:
        viewer.setStyle({"stick": {"radius": 0.14, "colorscheme": "greenCarbon"}})
        viewer.zoomTo()
    else:
        viewer.setStyle({"stick": {"radius": 0.15, "color": "#556B2F"}})
        viewer.addStyle({"elem": ["N", "O", "S", "F"]}, {"sphere": {"scale": 0.42, "color": "#84CC16"}})
        viewer.zoomTo()

    if spin:
        viewer.spin(True)

    viewer.setBackgroundColor("#021712")
    html_code = viewer._make_html()

    if "ESP" in modo:
        cargas_json = json.dumps(cargas_esp)
        esp_js = f"""
        var _cargas = {cargas_json};
        var _m = \\g<1>.getModel(0);
        if (_m) {{
            var _atoms = _m.selectedAtoms({{}});
            for (var i = 0; i < _atoms.length && i < _cargas.length; i++) {{
                _atoms[i].partialCharge = _cargas[i];
                if (!_atoms[i].properties) _atoms[i].properties = {{}};
                _atoms[i].properties.partialCharge = _cargas[i];
            }}
        }}
        \\g<1>.addSurface($3Dmol.SurfaceType.VDW, {{
            opacity: 0.78,
            colorscheme: {{
                prop: 'partialCharge',
                gradient: 'rwb',
                min: -{max_abs_q:.3f},
                max: {max_abs_q:.3f}
            }}
        }});
        \\g<1>.render();
        """
        html_code = re.sub(r"(viewer_\w+)\.render\(\);", esp_js, html_code, count=1)

    return html_code


# 6. HERO BAR (ESTILO 3D GLASSMORPHISM)
st.markdown(
    f"<div class='hero-bar'>"
    f"<div class='hero-left'>{svg_hero_emblem()}"
    f"<div><div class='brand-title'>PharmaChem AI Studio</div>"
    f"<div class='hero-sub'>Química Cuántica (TOM / TCC / L), Mapa ESP Gasteiger, BOILED-Egg y Comparador Estructural</div>"
    f"</div></div>"
    f"<div class='badge-pill'>Green Lab &amp; Quantum Core v8.5</div>"
    f"</div>",
    unsafe_allow_html=True,
)

# 7. BASE DE DATOS DE COMPUESTOS
base_moleculas = {
    "CC(=O)OC1=CC=CC=C1C(=O)O": ("Aspirina", "Analgésico / Antiinflamatorio", "C9H8O4"),
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": ("Cafeína", "Alcaloide / Estimulante SNC", "C8H10N4O2"),
    "CC(=O)NC1=CC=C(O)C=C1": ("Paracetamol", "Antipirético / Analgésico", "C8H9NO2"),
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": ("Ibuprofeno", "AINE / Inhibidor COX", "C13H18O2"),
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C": ("Testosterona", "Hormona Esteroidea", "C19H28O2"),
    "CN[C@@H](C[C@@H](C1=CC=C(C=C1)O)O)O": ("Adrenalina (Epinefrina)", "Neurotransmisor", "C9H13NO3"),
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CC=CN=C5": (
        "Imatinib",
        "Inhibidor Quinasa",
        "C29H31N7O",
    ),
    "CC(=O)O": ("Ácido Acético", "Disolvente Verde", "C2H4O2"),
    "C1=CC=C(C=C1)O": ("Fenol", "Precursor Orgánico", "C6H6O"),
    "CN1CCC[C@H]1C2=CN=CC=C2": ("Nicotina", "Alcaloide Natural", "C10H14N2"),
}

# 8. BARRA LATERAL CON BOTONES ESTILO MOCKUP
if "vista_nav" not in st.session_state:
    st.session_state.vista_nav = "dashboard"

st.sidebar.markdown(
    f"<div class='sec-header'>{svg_section_icon('search')}<span>Explorer</span></div>",
    unsafe_allow_html=True,
)

nav_items = [
    ("dashboard", "nav_dash", "Dashboard"),
    ("mol", "nav_mol", "Molécula y Cargas"),
    ("qm", "nav_qm", "Orbitales (TOM / TCC)"),
    ("spec", "nav_spec", "Espectros IR / UV"),
    ("adme", "nav_adme", "ADME y Seguridad"),
]
for vista_id, key_btn, label_btn in nav_items:
    btn_type = "primary" if st.session_state.vista_nav == vista_id else "secondary"
    if st.sidebar.button(label_btn, key=key_btn, type=btn_type, use_container_width=True):
        st.session_state.vista_nav = vista_id
        st.rerun()

st.sidebar.markdown("<hr style='border-color: rgba(132, 204, 22, 0.15); margin: 12px 0;'>", unsafe_allow_html=True)
modo_comparador = st.sidebar.toggle("Modo Comparador (A vs B)", value=False)

query_busqueda = (
    st.sidebar.text_input("Filtrar catálogo local:", placeholder="Ej: Cafeína, Imatinib...")
    .strip()
    .lower()
)

filtradas = {
    k: v
    for k, v in base_moleculas.items()
    if query_busqueda in v[0].lower() or query_busqueda in v[1].lower() or query_busqueda in k.lower()
}
if not filtradas:
    filtradas = base_moleculas

seleccion = st.sidebar.selectbox(
    "Espécimen A:",
    options=list(filtradas.keys()) + ["PUBCHEM", "CUSTOM"],
    format_func=lambda x: (
        f"{filtradas[x][0]} ({filtradas[x][2]})"
        if x in filtradas
        else ("Buscar en PubChem (Online)..." if x == "PUBCHEM" else "Dibujar / SMILES personalizado...")
    ),
)

nombre_pubchem = None
if seleccion == "PUBCHEM":
    termino_pubchem = st.sidebar.text_input("Nombre del compuesto (EN/ES):", "Dopamine")
    with st.sidebar:
        with st.spinner("Consultando PubChem..."):
            dato_pc = buscar_pubchem(termino_pubchem)
    if dato_pc:
        smiles_activo = dato_pc["smiles"]
        nombre_pubchem = (termino_pubchem.title(), f"PubChem CID {dato_pc['cid']}", dato_pc["formula"])
        st.sidebar.success(f"Encontrado: {dato_pc['formula']}")
    else:
        st.sidebar.warning("No encontrado en PubChem. Mostrando Etanol.")
        smiles_activo = "CCO"
elif seleccion == "CUSTOM":
    smiles_activo = st.session_state.get("custom_smiles", "CCO")
else:
    smiles_activo = seleccion

if seleccion == "CUSTOM":
    with st.expander("Editor Molecular Interactivo (Ketcher / SMILES)", expanded=True):
        if KETCHER_AVAILABLE:
            drawn_smiles = st_ketcher(value=smiles_activo, height=340)
            if drawn_smiles:
                smiles_activo = drawn_smiles
                st.session_state["custom_smiles"] = drawn_smiles
            st.caption(f"SMILES generado: `{smiles_activo}`")
        else:
            smiles_input = st.text_input("Ingresa estructura SMILES:", value=smiles_activo).strip()
            smiles_activo = smiles_input
            st.session_state["custom_smiles"] = smiles_input
            st.info("Instala `streamlit-ketcher` para habilitar el editor visual de dibujo molecular.")

smiles_b = None
if modo_comparador:
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div class='sec-header'>{svg_section_icon('atom')}<span>Espécimen B</span></div>",
        unsafe_allow_html=True,
    )
    keys_b = list(base_moleculas.keys())
    default_b_idx = 3 if len(keys_b) > 3 else 0
    sel_b = st.sidebar.selectbox(
        "Comparar contra:",
        options=keys_b + ["CUSTOM_B"],
        index=default_b_idx,
        format_func=lambda x: f"{base_moleculas[x][0]} ({base_moleculas[x][2]})"
        if x in base_moleculas
        else "SMILES personalizado B...",
    )
    smiles_b = st.sidebar.text_input("SMILES B:", "CC(=O)O").strip() if sel_b == "CUSTOM_B" else sel_b


# 9. VISTA MODO COMPARADOR (A vs B)
if modo_comparador and smiles_b:
    res_a = analizar_molecula(smiles_activo)
    res_b = analizar_molecula(smiles_b)

    if res_a is None or res_b is None:
        st.error("Verifica que ambos SMILES sean válidos para comparar.")
    else:
        info_a = nombre_pubchem or base_moleculas.get(smiles_activo, ("Compuesto A", "Personalizado", res_a["formula"]))
        info_b = base_moleculas.get(smiles_b, ("Compuesto B", "Personalizado", res_b["formula"]))
        tanimoto = calcular_tanimoto(smiles_activo, smiles_b)

        st.markdown(
            f"<div class='specimen-bar'>"
            f"<div><div style='font-size:0.75rem; color:#84CC16; text-transform:uppercase; letter-spacing:0.08em;'>Comparación Estructural A vs B</div>"
            f"<div style='font-size:1.45rem; font-weight:700; color:#FFFFFF;'>{info_a[0]} ({info_a[2]}) <span style='color:#84CC16;'>vs</span> {info_b[0]} ({info_b[2]})</div></div>"
            f"<div class='smiles-tag'>Similitud Tanimoto (Morgan): {tanimoto:.3f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        spin_comp = st.toggle("Rotación 3D simultánea", value=False)
        ca, cb = st.columns(2, gap="large")

        with ca:
            st.markdown(f"<div class='sec-header'>{svg_section_icon('cube')}<span>A: {info_a[0]}</span></div>", unsafe_allow_html=True)
            components.html(construir_html_3d(res_a["mol_block"], "Esferas", res_a["cargas_esp"], 0.35, spin=spin_comp, height=310), height=325)
            svg_a = generar_svg_2d(smiles_activo)
            if svg_a:
                st.markdown(f"<div style='background:rgba(2,22,17,0.75); border:1px solid rgba(132,204,22,0.25); border-radius:12px; padding:10px;'>{svg_a}</div>", unsafe_allow_html=True)

        with cb:
            st.markdown(f"<div class='sec-header'>{svg_section_icon('cube')}<span>B: {info_b[0]}</span></div>", unsafe_allow_html=True)
            components.html(construir_html_3d(res_b["mol_block"], "Esferas", res_b["cargas_esp"], 0.35, spin=spin_comp, height=310), height=325)
            svg_b = generar_svg_2d(smiles_b)
            if svg_b:
                st.markdown(f"<div style='background:rgba(2,22,17,0.75); border:1px solid rgba(132,204,22,0.25); border-radius:12px; padding:10px;'>{svg_b}</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='sec-header' style='margin-top:18px;'>{svg_section_icon('atom')}<span>Comparativa Fisicoquímica</span></div>", unsafe_allow_html=True)
        df_comp = pd.DataFrame(
            [
                {"Propiedad": "Peso Molecular (g/mol)", "Espécimen": info_a[0], "Valor": round(res_a["pm"], 2)},
                {"Propiedad": "Peso Molecular (g/mol)", "Espécimen": info_b[0], "Valor": round(res_b["pm"], 2)},
                {"Propiedad": "LogP", "Espécimen": info_a[0], "Valor": round(res_a["logp"], 2)},
                {"Propiedad": "LogP", "Espécimen": info_b[0], "Valor": round(res_b["logp"], 2)},
                {"Propiedad": "TPSA (Å²)", "Espécimen": info_a[0], "Valor": round(res_a["tpsa"], 1)},
                {"Propiedad": "TPSA (Å²)", "Espécimen": info_b[0], "Valor": round(res_b["tpsa"], 1)},
                {"Propiedad": "QED x 100", "Espécimen": info_a[0], "Valor": round(res_a["qed"] * 100, 1)},
                {"Propiedad": "QED x 100", "Espécimen": info_b[0], "Valor": round(res_b["qed"] * 100, 1)},
            ]
        )
        chart_comp = (
            alt.Chart(df_comp)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusTopLeft=4)
            .encode(
                x=alt.X("Propiedad:N", title="", axis=alt.Axis(labelColor="#D1FAE5", labelAngle=0)),
                y=alt.Y("Valor:Q", title="Valor", axis=alt.Axis(labelColor="#D1FAE5", titleColor="#84CC16", gridColor="rgba(132,204,22,0.1)")),
                color=alt.Color("Espécimen:N", scale=alt.Scale(range=["#84CC16", "#38bdf8"]), legend=alt.Legend(labelColor="#E6F4EA", titleColor="#84CC16")),
                xOffset="Espécimen:N",
                tooltip=["Propiedad", "Espécimen", "Valor"],
            )
            .properties(height=260, background="transparent")
            .configure_view(stroke=None)
        )
        st.altair_chart(chart_comp, use_container_width=True)

# 10. VISTA INDIVIDUAL COMPLETA
else:
    res = analizar_molecula(smiles_activo)

    if res is None:
        st.error("Error de sintaxis SMILES: ingresa una cadena química válida.")
    else:
        info_comp = nombre_pubchem or base_moleculas.get(smiles_activo, ("Compuesto Investigacional", "Personalizado", res["formula"]))
        nombre_mol, clase_mol, formula_mol = info_comp

        pm, logp, tpsa = res["pm"], res["logp"], res["tpsa"]
        hbd, hba, rot_bonds = res["hbd"], res["hba"], res["rot_bonds"]
        anillos_arom, qed_score = res["anillos_arom"], res["qed"]
        log_s, sol_mg_ml, alertas = res["log_s"], res["sol_mg_ml"], res["alertas"]
        cargas_esp = res["cargas_esp"]
        max_abs_q = max([abs(q) for q in cargas_esp] + [0.25])

        tom_res = calcular_huckel_tom(smiles_activo)

        st.markdown(
            f"<div class='specimen-bar'>"
            f"<div><div style='font-size:0.75rem; color:#84CC16; text-transform:uppercase; letter-spacing:0.08em;'>Espécimen Activo • {clase_mol}</div>"
            f"<div style='font-size:1.55rem; font-weight:700; color:#FFFFFF;'>{nombre_mol} <span style='font-size:1rem; font-weight:400; color:#D1FAE5;'>({formula_mol})</span></div></div>"
            f"<div class='smiles-tag'>SMILES: {smiles_activo}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # 4 TARJETAS KPI CON ICONOS 3D GLASS Y SPARKLINES
        st.markdown(
            f"<div class='kpi-grid'>"
            f"<div class='kpi-card'><div class='kpi-top'><span class='kpi-label'>Peso Molecular</span>{svg_kpi_badge('mw')}</div>"
            f"<div class='kpi-bottom'><div><div class='kpi-val'>{pm:.2f}</div><div class='kpi-unit'>g/mol (&lt; 500)</div></div>{svg_sparkline('#84CC16', 'up')}</div></div>"
            f"<div class='kpi-card'><div class='kpi-top'><span class='kpi-label'>LogP (WLOGP)</span>{svg_kpi_badge('logp')}</div>"
            f"<div class='kpi-bottom'><div><div class='kpi-val'>{logp:.2f}</div><div class='kpi-unit'>Octanol / Agua</div></div>{svg_sparkline('#84CC16', 'up')}</div></div>"
            f"<div class='kpi-card'><div class='kpi-top'><span class='kpi-label'>Solubilidad ESOL</span>{svg_kpi_badge('esol')}</div>"
            f"<div class='kpi-bottom'><div><div class='kpi-val'>{log_s:.2f}</div><div class='kpi-unit'>{sol_mg_ml:.2f} mg/mL</div></div>{svg_sparkline('#38BDF8', 'down')}</div></div>"
            f"<div class='kpi-card'><div class='kpi-top'><span class='kpi-label'>Índice QED</span>{svg_kpi_badge('qed')}</div>"
            f"<div class='kpi-bottom'><div><div class='kpi-val'>{qed_score:.2f}</div><div class='kpi-unit'>Drug-likeness</div></div>{svg_sparkline('#84CC16', 'up')}</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        col_analisis, col_visor = st.columns([1.25, 1], gap="large")
        selected_atom_id = -1

        def render_cargas():
            nonlocal selected_atom_id
            st.markdown(
                "<div class='glass-panel'>"
                "<div style='font-weight:600; color:#84CC16; font-size:0.95rem; margin-bottom:4px;'>Sincronización Tabla → Visor 3D y 2D</div>"
                "<p style='font-size:0.85rem; color:#D1FAE5; margin:0;'>Selecciona cualquier fila para enfocar y resaltar ese átomo en ambos visores.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            df_atomos = pd.DataFrame(res["datos_atomos"])
            event = st.dataframe(
                df_atomos, use_container_width=True, height=225, hide_index=True, on_select="rerun", selection_mode="single-row"
            )
            if event.selection.rows:
                row_idx = event.selection.rows[0]
                if row_idx < len(df_atomos):
                    fila = df_atomos.iloc[row_idx]
                    selected_atom_id = int(fila["ID Átomo"])
                    st.success(
                        f"Átomo seleccionado: ID #{selected_atom_id} | Elemento: {fila['Elemento']} | "
                        f"Hibridación: {fila['Hibridación']} | Carga: {fila['Carga (e)']} e"
                    )
            else:
                st.caption("Haz clic en el margen izquierdo de una fila para enfocar el átomo en los visores.")

            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "Descargar CSV de cargas",
                data=df_atomos.to_csv(index=False).encode("utf-8"),
                file_name=f"{nombre_mol.lower().replace(' ', '_')}_cargas.csv",
                mime="text/csv",
                use_container_width=True,
            )
            dl2.download_button(
                "Descargar estructura 3D (.mol)",
                data=res["mol_block"],
                file_name=f"{nombre_mol.lower().replace(' ', '_')}.mol",
                mime="chemical/x-mdl-molfile",
                use_container_width=True,
            )

        def render_cuantica():
            st.markdown(
                f"<div class='sec-header'>{svg_section_icon('atom')}<span>Ecuación de Schrödinger y Diagrama de Correlación TOM (Hückel π)</span></div>",
                unsafe_allow_html=True,
            )
            st.latex(r"\hat{H}\Psi_i = E_i\Psi_i \qquad \det(\mathbf{H} - E\mathbf{S}) = 0 \qquad \Psi_i = \sum_{\mu} c_{i\mu}\phi_\mu")

            if tom_res is None:
                st.info("Molécula saturada sin sistema π conjugado sp²/sp. Selecciona un compuesto aromático o conjugado para resolver sus orbitales π.")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("HOMO", f"{tom_res['homo_ev']} eV")
                m2.metric("LUMO", f"{tom_res['lumo_ev']} eV")
                m3.metric("Gap ΔE", f"{tom_res['gap_ev']} eV", f"{tom_res['gap_beta']} |β|")
                m4.metric("Electrones π", f"{tom_res['pi_electrons']} e⁻")

                st.altair_chart(construir_diagrama_tom_correlacion(tom_res), use_container_width=True)

                st.caption(
                    f"Descriptores DFT (Koopmans): Potencial químico $\\mu = {tom_res['mu_chem']}$ eV • "
                    f"Dureza $\\eta = {tom_res['eta_hard']}$ eV • Electrofilicidad $\\omega = {tom_res['omega_elec']}$ eV"
                )

                with st.expander("Tabla de Orbitales Moleculares y Densidad LCAO |cᵢ|²"):
                    st.dataframe(pd.DataFrame(tom_res["orbitales"]), use_container_width=True, hide_index=True, height=150)
                    st.dataframe(pd.DataFrame(tom_res["densidad_atoms"]), use_container_width=True, hide_index=True, height=150)

            st.markdown("<hr style='border-color: rgba(132, 204, 22, 0.18); margin: 14px 0;'>", unsafe_allow_html=True)
            st.markdown(f"<div class='sec-header'>{svg_section_icon('cube')}<span>Teoría del Campo Cristalino (TCC)</span></div>", unsafe_allow_html=True)

            tcc_c1, tcc_c2, tcc_c3 = st.columns(3)
            d_elec = tcc_c1.slider("Configuración dⁿ:", 1, 9, 6)
            geom_tcc = tcc_c2.selectbox("Geometría:", ["Octaédrica (Oh)", "Tetraédrica (Td)", "Cuadrada Plana (D4h)"])
            campo_tcc = tcc_c3.radio("Campo:", ["Alto Spin", "Bajo Spin"], horizontal=True)

            tcc_res = calcular_tcc(d_elec, geom_tcc, campo_tcc)
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("EECC (CFSE)", f"{tcc_res['eecc']} {tcc_res['unidad']}")
            tc2.metric("e⁻ Desapareados", f"{tcc_res['unpaired']}")
            tc3.metric("Momento μ_eff", f"{tcc_res['mu_b']} μB")

            chart_tcc = (
                alt.Chart(pd.DataFrame(tcc_res["niveles"]))
                .mark_bar(cornerRadius=4, color="#84CC16")
                .encode(
                    x=alt.X("Nivel:N", title="Orbitales d desdoblados", axis=alt.Axis(labelColor="#D1FAE5", labelAngle=0, titleColor="#84CC16")),
                    y=alt.Y("Energía:Q", title=f"Energía ({tcc_res['unidad']})", axis=alt.Axis(labelColor="#D1FAE5", titleColor="#84CC16", gridColor="rgba(132,204,22,0.1)")),
                    tooltip=["Nivel", "Energía", "Electrones"],
                )
                .properties(height=150, background="transparent")
                .configure_view(stroke=None)
            )
            st.altair_chart(chart_tcc, use_container_width=True)

            st.markdown("<hr style='border-color: rgba(132, 204, 22, 0.18); margin: 14px 0;'>", unsafe_allow_html=True)
            st.markdown(f"<div class='sec-header'>{svg_section_icon('atom')}<span>Momento Angular Orbital (Cuantización Espacial)</span></div>", unsafe_allow_html=True)
            st.latex(r"\hat{L}^2 Y_l^{m_l} = \hbar^2 l(l+1) Y_l^{m_l} \qquad \hat{L}_z Y_l^{m_l} = \hbar m_l Y_l^{m_l} \qquad \cos\theta = \frac{m_l}{\sqrt{l(l+1)}}")

            subcapas = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
            mao_c1, mao_c2, mao_c3 = st.columns(3)
            n_q = mao_c1.slider("Número principal (n):", 1, 5, 3, key="mao_n")
            l_q = mao_c2.selectbox("Azimutal (l):", options=list(range(n_q)), index=min(2, n_q - 1), format_func=lambda x: f"{x} (subcapa {subcapas[x]})", key="mao_l")
            m_opciones = list(range(-l_q, l_q + 1))
            ml_q = mao_c3.selectbox("Magnético (mₗ):", options=m_opciones, index=len(m_opciones) - 1, key="mao_ml")

            l_mag = float(np.sqrt(l_q * (l_q + 1)))
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Orbital", f"{n_q}{subcapas[l_q]}")
            k2.metric("|L| (ħ)", f"{l_mag:.3f} ħ")
            k3.metric("L_z (ħ)", f"{ml_q:+d} ħ")
            if l_q > 0:
                angulo_deg = float(np.degrees(np.arccos(np.clip(ml_q / l_mag, -1.0, 1.0))))
                k4.metric("Ángulo θ", f"{angulo_deg:.1f}°")
            else:
                k4.metric("Ángulo θ", "Esférico")

            st.caption(
                f"Nodos: {n_q - l_q - 1} radiales ($n-l-1$) • {l_q} angulares ($l$) • "
                f"Momento magnético orbital z: $\\mu_{{L,z}} = {-ml_q:+d}\\,\\mu_B$"
            )

            if l_q == 0:
                st.info("Para $l = 0$ (orbital $s$), $|\\mathbf{L}| = 0$ y la nube electrónica tiene simetría esférica.")
            else:
                filas_vec = []
                for m_val in m_opciones:
                    l_perp = float(np.sqrt(max(0.0, l_mag**2 - m_val**2)))
                    estado = "Seleccionado" if m_val == ml_q else "Permitido"
                    filas_vec.append({"m": f"mₗ={m_val:+d}", "x": 0.0, "y": 0.0, "Estado": estado})
                    filas_vec.append({"m": f"mₗ={m_val:+d}", "x": l_perp, "y": float(m_val), "Estado": estado})

                df_vec = pd.DataFrame(filas_vec)
                df_pts = df_vec[df_vec["x"] > 0].copy()
                if df_pts.empty:
                    df_pts = pd.DataFrame([{"m": f"mₗ={m:+d}", "x": float(np.sqrt(max(0.0, l_mag**2 - m**2))), "y": float(m), "Estado": "Seleccionado" if m == ml_q else "Permitido"} for m in m_opciones])

                limite = l_mag + 0.5
                lineas_l = (
                    alt.Chart(df_vec)
                    .mark_line(strokeWidth=2.5)
                    .encode(
                        x=alt.X("x:Q", title="Componente transversal L⊥ (ħ)", scale=alt.Scale(domain=[-0.3, limite]), axis=alt.Axis(labelColor="#D1FAE5", titleColor="#84CC16", gridColor="rgba(132,204,22,0.1)")),
                        y=alt.Y("y:Q", title="Proyección Lz (ħ)", scale=alt.Scale(domain=[-limite, limite]), axis=alt.Axis(labelColor="#D1FAE5", titleColor="#84CC16", gridColor="rgba(132,204,22,0.1)")),
                        detail="m:N",
                        color=alt.Color("Estado:N", scale=alt.Scale(domain=["Seleccionado", "Permitido"], range=["#84CC16", "#38bdf8"]), legend=alt.Legend(labelColor="#E6F4EA", titleColor="#84CC16")),
                    )
                )
                puntos_l = (
                    alt.Chart(df_pts)
                    .mark_point(size=100, filled=True)
                    .encode(x="x:Q", y="y:Q", color=alt.Color("Estado:N", scale=alt.Scale(domain=["Seleccionado", "Permitido"], range=["#84CC16", "#38bdf8"])), tooltip=["m", "y", "x"])
                )
                st.altair_chart((lineas_l + puntos_l).properties(height=210, background="transparent").configure_view(stroke=None), use_container_width=True)

        def render_espectros():
            st.markdown(f"<div class='sec-header'>{svg_section_icon('atom')}<span>Espectroscopia Simulada (SMARTS)</span></div>", unsafe_allow_html=True)

            ondas_ir = np.linspace(600, 3900, 320)
            transmitancia = np.ones_like(ondas_ir) * 95.0 - 20 * np.exp(-((ondas_ir - 2940) / 55) ** 2)
            sm = res["smarts"]
            if sm["has_oh"]:
                transmitancia -= 42 * np.exp(-((ondas_ir - 3360) / 130) ** 2)
            if sm["has_nh"]:
                transmitancia -= 28 * np.exp(-((ondas_ir - 3310) / 55) ** 2)
            if sm["has_cn"]:
                transmitancia -= 38 * np.exp(-((ondas_ir - 2240) / 22) ** 2)
            if sm["has_co"]:
                transmitancia -= 60 * np.exp(-((ondas_ir - 1715) / 28) ** 2)
            if anillos_arom > 0:
                transmitancia -= 34 * np.exp(-((ondas_ir - 1595) / 32) ** 2)
            if sm["has_c_o"]:
                transmitancia -= 30 * np.exp(-((ondas_ir - 1120) / 45) ** 2)

            df_ir = pd.DataFrame({"Onda": ondas_ir, "Transmitancia": np.clip(transmitancia, 5, 100)})
            chart_ir = (
                alt.Chart(df_ir)
                .mark_line(color="#84CC16", strokeWidth=2)
                .encode(
                    x=alt.X("Onda:Q", title="Número de Onda (cm⁻¹)", scale=alt.Scale(reverse=True), axis=alt.Axis(grid=False, labelColor="#D1FAE5", titleColor="#84CC16")),
                    y=alt.Y("Transmitancia:Q", title="% Transmitancia", axis=alt.Axis(grid=True, gridColor="rgba(132, 204, 22, 0.1)", labelColor="#D1FAE5", titleColor="#84CC16")),
                )
                .properties(height=155, background="transparent")
                .configure_view(stroke=None)
            )
            st.caption("Espectro Infrarrojo (FT-IR por grupos funcionales SMARTS):")
            st.altair_chart(chart_ir, use_container_width=True)

            long_onda = np.linspace(200, 550, 180)
            lambda_max = 215 + (anillos_arom * 40) + (hba * 5)
            df_uv = pd.DataFrame({"Lambda": long_onda, "Absorbancia": 1.3 * np.exp(-((long_onda - lambda_max) / 25) ** 2)})
            chart_uv = (
                alt.Chart(df_uv)
                .mark_area(
                    color=alt.Gradient(gradient="linear", stops=[alt.GradientStop(color="#84CC16", offset=0), alt.GradientStop(color="rgba(6,78,59,0.1)", offset=1)], x1=1, y1=1, x2=1, y2=0),
                    line={"color": "#84CC16", "strokeWidth": 2},
                )
                .encode(
                    x=alt.X("Lambda:Q", title="Longitud de Onda (nm)", axis=alt.Axis(grid=False, labelColor="#D1FAE5", titleColor="#84CC16")),
                    y=alt.Y("Absorbancia:Q", title="Absorbancia (UA)", axis=alt.Axis(grid=True, gridColor="rgba(132, 204, 22, 0.1)", labelColor="#D1FAE5", titleColor="#84CC16")),
                )
                .properties(height=135, background="transparent")
                .configure_view(stroke=None)
            )
            st.caption(f"Espectro UV-Vis estimado (λ max: {lambda_max} nm):")
            st.altair_chart(chart_uv, use_container_width=True)

        def render_adme():
            st.markdown(f"<div class='sec-header'>{svg_section_icon('shield')}<span>Mapa BOILED-Egg (HIA vs BBB)</span></div>", unsafe_allow_html=True)
            st.altair_chart(construir_boiled_egg(tpsa, logp, nombre_mol), use_container_width=True)

            sol_cat = "Muy soluble" if log_s > -2 else ("Soluble" if log_s >= -4 else ("Moderada" if log_s >= -6 else "Poco soluble"))
            reglas = [
                ("Peso Molecular < 500 g/mol (Lipinski)", pm < 500, f"{pm:.1f}"),
                ("Lipofilicidad LogP < 5 (Lipinski)", logp < 5, f"{logp:.2f}"),
                ("Donantes H (HBD ≤ 5) / Aceptores H (HBA ≤ 10)", hbd <= 5 and hba <= 10, f"{hbd} / {hba}"),
                ("TPSA ≤ 140 Å² & Rotables ≤ 10 (Veber)", tpsa <= 140 and rot_bonds <= 10, f"{tpsa:.1f} Å² | {rot_bonds}"),
                (f"Solubilidad ESOL ({sol_cat})", log_s >= -4, f"LogS {log_s:.2f}"),
            ]
            for nombre, cumple, val in reglas:
                color = "#84CC16" if cumple else "#f59e0b"
                estado_txt = "CUMPLE" if cumple else "ALERTA"
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; background:rgba(2,22,17,0.72); padding:8px 12px; border-radius:8px; margin-bottom:6px; border-left:4px solid {color};'>"
                    f"<span style='color:#E6F4EA; font-size:0.85rem;'>{nombre}</span>"
                    f"<span style='font-family:\"JetBrains Mono\", monospace; color:{color}; font-weight:600; font-size:0.82rem;'>{val} · {estado_txt}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if alertas:
                st.warning(f"Alertas PAINS / Brenk ({len(alertas)}): " + ", ".join(alertas))
            else:
                st.success("Filtros PAINS / Brenk: Sin alertas estructurales detectadas.")

            similares = top_similares_catalogo(smiles_activo, base_moleculas, top_n=2)
            if similares:
                texto_sim = " • ".join([f"**{n}** ({f}) — `{s}`" for n, f, s in similares])
                st.caption(f"Más similares en catálogo (Tanimoto): {texto_sim}")

        with col_analisis:
            vista = st.session_state.vista_nav
            if vista == "mol":
                render_cargas()
            elif vista == "qm":
                render_cuantica()
            elif vista == "spec":
                render_espectros()
            elif vista == "adme":
                render_adme()
            else:
                tab_quant, tab_qm, tab_spec, tab_adme = st.tabs(
                    ["Cargas y Tabla", "Cuántica (TOM / TCC / L)", "Espectros", "ADME y BOILED-Egg"]
                )
                with tab_quant:
                    render_cargas()
                with tab_qm:
                    render_cuantica()
                with tab_spec:
                    render_espectros()
                with tab_adme:
                    render_adme()

        with col_visor:
            st.markdown(f"<div class='sec-header'>{svg_section_icon('cube')}<span>Visor Molecular 3D y 2D</span></div>", unsafe_allow_html=True)

            v_col1, v_col2 = st.columns([2.2, 1])
            with v_col1:
                modo_3d = st.radio("Modo de renderizado:", ["Esferas 3D", "Mapa ESP", "Farmacóforo"], horizontal=True, label_visibility="collapsed")
            with v_col2:
                spin_3d = st.toggle("Rotar 3D", value=False)

            try:
                html_3d_code = construir_html_3d(
                    res["mol_block"], modo_3d, cargas_esp, max_abs_q, selected_atom_id=selected_atom_id, spin=spin_3d, height=355
                )
                components.html(html_3d_code, height=370)

                if "ESP" in modo_3d:
                    st.markdown(
                        f"<div style='display:flex; align-items:center; justify-content:space-between; font-size:0.75rem; font-family:\"JetBrains Mono\", monospace; margin-top:-6px; padding:6px 10px; background:rgba(2,22,17,0.75); border-radius:8px; border:1px solid rgba(132,204,22,0.22);'>"
                        f"<span style='color:#f87171;'>-{max_abs_q:.2f} e (δ⁻)</span>"
                        f"<span style='color:#e2e8f0;'>0.00 e</span>"
                        f"<span style='color:#60a5fa;'>+{max_abs_q:.2f} e (δ⁺)</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Arrastra para rotar, usa la rueda para zoom o selecciona una fila en la tabla atómica.")

                svg_2d = generar_svg_2d(smiles_activo, selected_atom_id)
                if svg_2d:
                    st.markdown(
                        f"<div style='margin-top:10px; background:rgba(2,22,17,0.78); border:1px solid rgba(132,204,22,0.22); border-radius:14px; padding:12px;'>"
                        f"<div style='font-size:0.72rem; color:#84CC16; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px; font-family:\"JetBrains Mono\", monospace;'>Estructura 2D (Esquelética)</div>"
                        f"{svg_2d}</div>",
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Error en renderizado 3D: {e}")

# 11. TARJETAS INFERIORES CON HOLOGRAMAS ISOMÉTRICOS 3D
st.markdown("<hr style='border-color: rgba(132, 204, 22, 0.18); margin: 24px 0 18px 0;'>", unsafe_allow_html=True)
st.markdown(f"<div class='sec-header'>{svg_section_icon('shield')}<span>Protocolos de Laboratorio y Química Sostenible</span></div>", unsafe_allow_html=True)

p1, p2, p3 = st.columns(3, gap="medium")

with p1:
    st.markdown(
        f"<div class='proto-card'>"
        f"<div><div class='proto-head'>Bioseguridad y EPP</div>"
        f"<p class='proto-text'>Operación bajo campana de extracción con guantes de nitrilo y protección ocular certificada.</p></div>"
        f"{svg_hologram('shield')}</div>",
        unsafe_allow_html=True,
    )

with p2:
    st.markdown(
        f"<div class='proto-card'>"
        f"<div><div class='proto-head'>Economía Atómica</div>"
        f"<p class='proto-text'>Diseño sintético enfocado en la minimización de residuos peligrosos y disolventes bio-basados.</p></div>"
        f"{svg_hologram('cube')}</div>",
        unsafe_allow_html=True,
    )

with p3:
    st.markdown(
        f"<div class='proto-card'>"
        f"<div><div class='proto-head'>Control Espectroscópico</div>"
        f"<p class='proto-text'>Validación estructural mediante cromatografía, espectroscopia FT-IR y densidad electrónica.</p></div>"
        f"{svg_hologram('wave')}</div>",
        unsafe_allow_html=True,
    )
