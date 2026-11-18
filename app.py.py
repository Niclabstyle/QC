import base64
import json
import os
import re

import altair as alt
import numpy as np
import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="PharmaChem AI | Quantum & Green Cheminformatics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. INYECCIÓN DE TIPOGRAFÍA PERSONALIZADA (NCL Wicky Bejari)
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

# 3. SISTEMA DE DISEÑO CSS ULTRA-SMOOTH & GLASSMORPHISM
st.markdown(
    f"""
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
        color: var(--mint-pale);
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
        color: #84CC16 !important;
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
        grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
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
        opacity: 0.85;
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
        opacity: 0.7;
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

    .stMarkdown p, .stMarkdown li, label[data-testid="stWidgetLabel"] p {{
        color: #E6F4EA;
    }}
</style>
""",
    unsafe_allow_html=True,
)


# 4. PROCESAMIENTO QUIMIOINFORMÁTICO CACHEADO
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

    mol_h = Chem.AddHs(mol)
    AllChem.ComputeGasteigerCharges(mol_h)

    cargas_esp = []
    datos_atomos = []
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
    drawer = rdMolDraw2D.MolDraw2DSVG(420, 240)
    opts = drawer.drawOptions()

    if hasattr(rdMolDraw2D, "SetDarkMode"):
        rdMolDraw2D.SetDarkMode(opts)
    opts.setBackgroundColour((0.0, 0.0, 0.0, 0.0))
    opts.bondLineWidth = 2.0

    highlight_atoms = [highlight_idx] if 0 <= highlight_idx < mol_2d.GetNumAtoms() else []
    highlight_colors = {highlight_idx: (1.0, 0.0, 0.5, 0.65)} if highlight_atoms else {}

    drawer.DrawMolecule(
        mol_2d,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Limpiar cabecera XML y saltos de línea para que st.markdown no rompa el HTML
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(
        r"width='[^']+'\s+height='[^']+'",
        "style='width:100%; height:auto; max-height:220px;'",
        svg,
        count=1,
    )
    svg = " ".join(line.strip() for line in svg.splitlines() if line.strip())
    return svg


# 5. HERO SECTION
st.markdown(
    """
<div class="hero-box">
    <div class="badge-pill">🌿 Green Lab & Quantum Core v5.5</div>
    <div class="brand-title">PharmaChem AI Studio</div>
    <div class="hero-desc">
        Entorno quimioinformático de alto rendimiento con mapa de potencial electrostático (ESP) por cargas de Gasteiger,
        estructura 2D SVG, espectroscopia SMARTS y sincronización directa entre la tabla atómica y los visores.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# 6. BASE DE DATOS DE COMPUESTOS
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

# 7. BARRA LATERAL
st.sidebar.markdown('<div class="sec-header">🔍 Explorador</div>', unsafe_allow_html=True)
query_busqueda = (
    st.sidebar.text_input("Filtrar compuesto o fórmula:", placeholder="Ej: Cafeína, Imatinib...")
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
    "Espécimen activo:",
    options=list(filtradas.keys()) + ["CUSTOM"],
    format_func=lambda x: f"{filtradas[x][0]} ({filtradas[x][2]})" if x in filtradas else "✍️ SMILES personalizado...",
)

smiles_activo = st.sidebar.text_input("Estructura SMILES:", "CCO").strip() if seleccion == "CUSTOM" else seleccion

# 8. ANÁLISIS Y RENDERIZADO
res = analizar_molecula(smiles_activo)

if res is None:
    st.error("❌ Error de sintaxis SMILES: Ingrese una cadena química válida.")
else:
    info_comp = base_moleculas.get(
        smiles_activo, ("Compuesto Investigacional", "Personalizado", res["formula"])
    )
    nombre_mol, clase_mol, formula_mol = info_comp

    pm = res["pm"]
    logp = res["logp"]
    tpsa = res["tpsa"]
    hbd = res["hbd"]
    hba = res["hba"]
    rot_bonds = res["rot_bonds"]
    anillos_arom = res["anillos_arom"]
    qed_score = res["qed"]
    cargas_esp = res["cargas_esp"]
    max_abs_q = max([abs(q) for q in cargas_esp] + [0.25])
    carga_max = max(cargas_esp) if cargas_esp else 0.0

    # Estimación heurística rápida (no DFT)
    homo_est = round(-6.85 + (anillos_arom * 0.32) + (hba * 0.08), 2)
    lumo_est = round(-1.15 - (anillos_arom * 0.28) - (hbd * 0.06), 2)
    band_gap = round(abs(lumo_est - homo_est), 2)

    # Header del espécimen
    st.markdown(
        f"""
    <div class="specimen-bar">
        <div>
            <div style="font-size: 0.78rem; color: #84CC16; text-transform: uppercase; letter-spacing: 0.08em;">Espécimen Activo • {clase_mol}</div>
            <div style="font-family: 'WickyBejari', 'Special Elite', serif; font-size: 1.7rem; color: #E6F4EA;">{nombre_mol} <span style="font-size: 1.05rem; color: #D1FAE5;">({formula_mol})</span></div>
        </div>
        <div class="smiles-tag">SMILES: {smiles_activo}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Tarjetas KPI
    st.markdown(
        f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Peso Molecular</div>
            <div class="kpi-val">{pm:.2f}</div>
            <div class="kpi-sub">g/mol (&lt; 500)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">LogP (Lipofilicidad)</div>
            <div class="kpi-val">{logp:.2f}</div>
            <div class="kpi-sub">Octanol / Agua</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Área TPSA</div>
            <div class="kpi-val">{tpsa:.1f} Å²</div>
            <div class="kpi-sub">Superficie Polar</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Índice QED</div>
            <div class="kpi-val">{qed_score:.2f}</div>
            <div class="kpi-sub">Drug-likeness (0–1)</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_analisis, col_visor = st.columns([1.25, 1], gap="large")
    selected_atom_id = -1

    with col_analisis:
        tab_quant, tab_spec, tab_adme = st.tabs(
            [
                "⚛️ Cargas & Tabla Atómica",
                "📈 Espectros Cohesionados",
                "💊 Parámetros ADME",
            ]
        )

        with tab_quant:
            st.markdown(
                """
            <div class="glass-panel">
                <div class="sec-header" style="font-size: 1.1rem;">⚡ Sincronización Tabla → Visor 3D y 2D</div>
                <p style="font-size: 0.88rem; color: #D1FAE5; line-height: 1.45; margin: 0;">
                    Selecciona cualquier fila de la tabla inferior para enfocar y resaltar ese átomo tanto en el modelo 3D como en el diagrama 2D.
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            q1, q2, q3 = st.columns(3)
            q1.metric("HOMO (Est. empírica)", f"{homo_est} eV")
            q2.metric("LUMO (Est. empírica)", f"{lumo_est} eV")
            q3.metric("Band Gap / Carga Máx.", f"{band_gap} eV / +{carga_max:.3f} e")

            df_atomos = pd.DataFrame(res["datos_atomos"])
            event = st.dataframe(
                df_atomos,
                use_container_width=True,
                height=195,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
            )

            if event.selection.rows:
                row_idx = event.selection.rows[0]
                if row_idx < len(df_atomos):
                    fila = df_atomos.iloc[row_idx]
                    selected_atom_id = int(fila["ID Átomo"])
                    st.success(
                        f"🎯 **Átomo Seleccionado:** ID #{selected_atom_id} | "
                        f"Elemento: **{fila['Elemento']}** | "
                        f"Hibridación: **{fila['Hibridación']}** | "
                        f"Carga: **{fila['Carga (e)']} e**"
                    )
            else:
                st.caption("ℹ️ Haz clic en el margen izquierdo de una fila para enfocar el átomo en los visores.")

            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "⬇️ Descargar CSV de cargas",
                data=df_atomos.to_csv(index=False).encode("utf-8"),
                file_name=f"{nombre_mol.lower().replace(' ', '_')}_cargas.csv",
                mime="text/csv",
                use_container_width=True,
            )
            dl2.download_button(
                "⬇️ Descargar estructura 3D (.mol)",
                data=res["mol_block"],
                file_name=f"{nombre_mol.lower().replace(' ', '_')}.mol",
                mime="chemical/x-mdl-molfile",
                use_container_width=True,
            )

        with tab_spec:
            st.markdown(
                '<div class="sec-header" style="font-size: 1.1rem;">📉 Espectroscopia Simulada (SMARTS)</div>',
                unsafe_allow_html=True,
            )

            ondas_ir = np.linspace(600, 3900, 320)
            transmitancia = np.ones_like(ondas_ir) * 95.0
            transmitancia -= 20 * np.exp(-((ondas_ir - 2940) / 55) ** 2)

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
                    x=alt.X(
                        "Onda:Q",
                        title="Número de Onda (cm⁻¹)",
                        scale=alt.Scale(reverse=True),
                        axis=alt.Axis(grid=False, labelColor="#D1FAE5", titleColor="#84CC16"),
                    ),
                    y=alt.Y(
                        "Transmitancia:Q",
                        title="% Transmitancia",
                        axis=alt.Axis(
                            grid=True,
                            gridColor="rgba(132, 204, 22, 0.1)",
                            labelColor="#D1FAE5",
                            titleColor="#84CC16",
                        ),
                    ),
                )
                .properties(height=155, background="transparent")
                .configure_view(stroke=None)
            )

            st.caption("Espectro Infrarrojo (FT-IR por grupos funcionales SMARTS):")
            st.altair_chart(chart_ir, use_container_width=True)

            long_onda = np.linspace(200, 550, 180)
            lambda_max = 215 + (anillos_arom * 40) + (hba * 5)
            absorbancia = 1.3 * np.exp(-((long_onda - lambda_max) / 25) ** 2)
            df_uv = pd.DataFrame({"Lambda": long_onda, "Absorbancia": absorbancia})

            chart_uv = (
                alt.Chart(df_uv)
                .mark_area(
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[
                            alt.GradientStop(color="#84CC16", offset=0),
                            alt.GradientStop(color="rgba(6,78,59,0.1)", offset=1),
                        ],
                        x1=1,
                        y1=1,
                        x2=1,
                        y2=0,
                    ),
                    line={"color": "#84CC16", "strokeWidth": 2},
                )
                .encode(
                    x=alt.X(
                        "Lambda:Q",
                        title="Longitud de Onda (nm)",
                        axis=alt.Axis(grid=False, labelColor="#D1FAE5", titleColor="#84CC16"),
                    ),
                    y=alt.Y(
                        "Absorbancia:Q",
                        title="Absorbancia (UA)",
                        axis=alt.Axis(
                            grid=True,
                            gridColor="rgba(132, 204, 22, 0.1)",
                            labelColor="#D1FAE5",
                            titleColor="#84CC16",
                        ),
                    ),
                )
                .properties(height=135, background="transparent")
                .configure_view(stroke=None)
            )

            st.caption(f"Espectro UV-Vis estimado (λ max: {lambda_max} nm):")
            st.altair_chart(chart_uv, use_container_width=True)

        with tab_adme:
            st.markdown(
                '<div class="sec-header" style="font-size: 1.1rem;">💊 Perfil Farmacológico (Lipinski & Veber)</div>',
                unsafe_allow_html=True,
            )
            reglas = [
                ("Peso Molecular < 500 g/mol (Lipinski)", pm < 500, f"{pm:.1f}"),
                ("Lipofilicidad LogP < 5 (Lipinski)", logp < 5, f"{logp:.2f}"),
                ("Donantes H (HBD ≤ 5)", hbd <= 5, f"{hbd}"),
                ("Aceptores H (HBA ≤ 10)", hba <= 10, f"{hba}"),
                ("Área Polar TPSA ≤ 140 Å² (Veber)", tpsa <= 140, f"{tpsa:.1f} Å²"),
                ("Enlaces Rotables ≤ 10 (Veber)", rot_bonds <= 10, f"{rot_bonds}"),
            ]
            for nombre, cumple, val in reglas:
                color = "#84CC16" if cumple else "#f59e0b"
                st.markdown(
                    f"""
                <div style="display:flex; justify-content:space-between; background:rgba(2,22,17,0.7); padding:9px 12px; border-radius:8px; margin-bottom:6px; border-left:4px solid {color};">
                    <span style="color:#E6F4EA; font-size:0.9rem;">{'✅' if cumple else '⚠️'} {nombre}</span>
                    <span style="font-family:'JetBrains Mono'; color:{color}; font-weight:600; font-size:0.9rem;">{val}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    with col_visor:
        st.markdown('<div class="sec-header">⚛️ Motor 3D & 2D Interactivo</div>', unsafe_allow_html=True)

        modo_3d = st.radio(
            "Modo de renderizado:",
            [
                "🔬 Esferas y Varillas",
                "⚡ Superficie Potencial (ESP)",
                "🔗 Farmacóforo (Donadores / Aceptores)",
            ],
            horizontal=True,
            label_visibility="collapsed",
        )

        try:
            viewer = py3Dmol.view(width="100%", height=360)
            viewer.addModel(res["mol_block"], "mol")

            if "Esferas" in modo_3d:
                viewer.setStyle(
                    {
                        "stick": {"radius": 0.16, "colorscheme": "greenCarbon"},
                        "sphere": {"scale": 0.28, "colorscheme": "greenCarbon"},
                    }
                )
                if selected_atom_id != -1:
                    serial_3dmol = selected_atom_id + 1
                    viewer.addStyle(
                        {"serial": serial_3dmol},
                        {
                            "sphere": {"scale": 0.58, "color": "#ff007f"},
                            "stick": {"radius": 0.24, "color": "#ff007f"},
                        },
                    )
                    viewer.addLabel(
                        f"Atom #{selected_atom_id}",
                        {
                            "position": {"serial": serial_3dmol},
                            "backgroundColor": "#064E3B",
                            "fontColor": "#84CC16",
                            "fontSize": 13,
                        },
                    )
                    viewer.zoomTo({"serial": serial_3dmol})
                else:
                    viewer.zoomTo()

            elif "ESP" in modo_3d:
                viewer.setStyle({"stick": {"radius": 0.14, "colorscheme": "greenCarbon"}})
                viewer.zoomTo()

            else:
                viewer.setStyle({"stick": {"radius": 0.15, "color": "#556B2F"}})
                viewer.addStyle(
                    {"elem": ["N", "O", "S", "F"]},
                    {"sphere": {"scale": 0.42, "color": "#84CC16"}},
                )
                viewer.zoomTo()

            viewer.setBackgroundColor("#021712")
            html_3d_code = viewer._make_html()

            if "ESP" in modo_3d:
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
                html_3d_code = re.sub(r"(viewer_\w+)\.render\(\);", esp_js, html_3d_code, count=1)

            components.html(html_3d_code, height=375)

            if "ESP" in modo_3d:
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; justify-content:space-between; font-size:0.75rem; font-family:'JetBrains Mono', monospace; margin-top:-6px; padding:6px 10px; background:rgba(2,22,17,0.7); border-radius:8px; border:1px solid rgba(132,204,22,0.2);">
                        <span style="color:#f87171;">🔴 -{max_abs_q:.2f} e (δ⁻)</span>
                        <span style="color:#e2e8f0;">⚪ 0.00 e</span>
                        <span style="color:#60a5fa;">🔵 +{max_abs_q:.2f} e (δ⁺)</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.caption("💡 Arrastra para rotar, usa la rueda para zoom o selecciona una fila en la tabla atómica.")

            svg_2d = generar_svg_2d(smiles_activo, selected_atom_id)
            if svg_2d:
                html_2d = (
                    "<div style='margin-top:10px; background:rgba(2,22,17,0.75); "
                    "border:1px solid rgba(132,204,22,0.25); border-radius:12px; padding:12px;'>"
                    "<div style='font-size:0.75rem; color:#84CC16; text-transform:uppercase; "
                    "letter-spacing:0.08em; margin-bottom:6px; font-family:\"JetBrains Mono\", monospace;'>"
                    "Estructura 2D (Esquelética)</div>"
                    f"{svg_2d}</div>"
                )
                st.markdown(html_2d, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error en renderizado 3D: {e}")

    st.markdown("<hr style='border-color: rgba(132, 204, 22, 0.2); margin: 25px 0;'>", unsafe_allow_html=True)

    # 9. PROTOCOLOS VERDES
    st.markdown('<div class="sec-header">🛡️ Protocolos de Laboratorio & Química Sostenible</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(
            """
        <div class="proto-card">
            <div class="proto-head">⚠️ Bioseguridad y EPP</div>
            <p style="font-size: 0.85rem; color: #E6F4EA; line-height: 1.45; margin: 0;">
                Operación bajo campana de extracción con guantes de nitrilo y protección ocular certificada.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with p2:
        st.markdown(
            """
        <div class="proto-card">
            <div class="proto-head">🌱 Economía Atómica</div>
            <p style="font-size: 0.85rem; color: #E6F4EA; line-height: 1.45; margin: 0;">
                Diseño sintético enfocado en la minimización de residuos peligrosos y uso de disolventes bio-basados.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with p3:
        st.markdown(
            """
        <div class="proto-card">
            <div class="proto-head">🔬 Control Analítico</div>
            <p style="font-size: 0.85rem; color: #E6F4EA; line-height: 1.45; margin: 0;">
                Verificación estructural mediante cromatografía y espectroscopia infrarroja de alta resolución.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
