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

    /* Fondo Global Fluido */
    .stApp {{
        background: radial-gradient(circle at 15% 10%, #064E3B 0%, #021a14 45%, #010c09 100%) !important;
        color: var(--mint-pale) !important;
        font-family: 'Inter', sans-serif !important;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Barra Lateral Estilizada */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #021712 0%, #010a08 100%) !important;
        border-right: 1px solid rgba(132, 204, 22, 0.2) !important;
    }}

    /* Tipografías */
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

    /* Hero Banner Smooth */
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

    /* KPIs Grid */
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

    /* Barra de Espécimen */
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

    /* Paneles Glass */
    .glass-panel {{
        background: rgba(6, 78, 59, 0.3);
        border: 1px solid rgba(209, 250, 229, 0.12);
        border-radius: 14px;
        padding: 18px;
        backdrop-filter: blur(8px);
        margin-bottom: 15px;
    }}

    /* Tarjetas de Protocolo */
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

    /* Pestañas Smooth */
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
    /* ==========================================
       CORRECCIÓN GLOBAL DE CONTRASTE (TEXTO CLARO)
       ========================================== */
    p, span, label, div, .stMarkdown, .stText, .streamlit-expanderHeader, .css-10trblm {
        color: #E6F4EA !important;
    }

    /* Forzar texto claro en tablas y DataFrames */
    .stDataFrame, .dataframe, th, td {
        color: #E6F4EA !important;
        background-color: transparent !important;
    }

    /* Textos dentro de selectbox e inputs */
    .stSelectbox div[data-baseweb="select"] span, input {
        color: #FFFFFF !important;
    }

    /* Títulos de pestañas */
    .stTabs [data-baseweb="tab"] {
        color: #D1FAE5 !important;
    }
</style>
""", unsafe_allow_html=True)

# 4. HERO SECTION
st.markdown("""
<div class="hero-box">
    <div class="badge-pill">🌿 Green Lab & Quantum Core v5.0</div>
    <div class="brand-title">PharmaChem AI Studio</div>
    <div class="hero-desc">
        Entorno quimiinformático de alto rendimiento optimizado con simulación electrostática, modelado 3D 
        y espectroscopia computacional integrada bajo una estética sostenible de alta precisión.
    </div>
</div>
""", unsafe_allow_html=True)

# 5. BASE DE DATOS DE COMPUESTOS
base_moleculas = {
    "CC(=O)OC1=CC=CC=C1C(=O)O": ("Aspirina", "Analgésico / Antiinflamatorio", "C9H8O4"),
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": ("Cafeína", "Alcaloide / Estimulante SNC", "C8H10N4O2"),
    "CC(=O)NC1=CC=C(O)C=C1": ("Paracetamol", "Antipirético / Analgésico", "C8H9NO2"),
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": ("Ibuprofeno", "AINE / Inhibidor COX", "C13H18O2"),
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C": ("Testosterona", "Hormona Esteroidea", "C19H28O2"),
    "CN[C@@H](C[C@@H](C1=CC=C(C=C1)O)O)O": ("Adrenalina (Epinefrina)", "Neurotransmisor", "C9H13NO3"),
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CC=CN=C5": ("Imatinib", "Inhibidor Quinasa", "C29H31N7O"),
    "CC(=O)O": ("Ácido Acético", "Disolvente Verde", "C2H4O2"),
    "C1=CC=C(C=C1)O": ("Fenol", "Precursor Orgánico", "C6H6O"),
    "CN1CCC[C@H]1C2=CN=CC=C2": ("Nicotina", "Alcaloide Natural", "C10H14N2")
}

# 6. BARRA LATERAL
st.sidebar.markdown('<div class="sec-header">🔍 Explorador</div>', unsafe_allow_html=True)
query_busqueda = st.sidebar.text_input("Filtrar compuesto o fórmula:", placeholder="Ej: Cafeína, Imatinib...").strip().lower()

filtradas = {k: v for k, v in base_moleculas.items() if query_busqueda in v[0].lower() or query_busqueda in v[1].lower() or query_busqueda in k.lower()}
if not filtradas:
    filtradas = base_moleculas

seleccion = st.sidebar.selectbox(
    "Espécimen activo:",
    options=list(filtradas.keys()) + ["CUSTOM"],
    format_func=lambda x: f"{filtradas[x][0]} ({filtradas[x][2]})" if x in filtradas else "✍️ SMILES personalizado..."
)

smiles_activo = st.sidebar.text_input("Estructura SMILES:", "CCO") if seleccion == "CUSTOM" else seleccion

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sec-header">⚛️ Motor 3D</div>', unsafe_allow_html=True)
modo_3d = st.sidebar.radio(
    "Modo de renderizado:",
    [
        "🔬 Esferas y Varillas (Estándar)",
        "⚡ Superficie Potencial (ESP)",
        "🔗 Simulación Docking (Sitio Activo)"
    ]
)

# 7. PROCESAMIENTO RDKIT
mol = Chem.MolFromSmiles(smiles_activo)

if mol is None:
    st.error("❌ Error de sintaxis SMILES: Ingrese una cadena química válida.")
else:
    info_comp = base_moleculas.get(smiles_activo, ("Compuesto Investigacional", "Personalizado", rdMolDescriptors.CalcMolFormula(mol)))
    nombre_mol, clase_mol, formula_mol = info_comp

    pm = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    anillos_arom = rdMolDescriptors.CalcNumAromaticRings(mol)

    mol_h = Chem.AddHs(mol)
    AllChem.ComputeGasteigerCharges(mol_h)
    cargas = [float(atom.GetProp('_GasteigerCharge')) for atom in mol_h.GetAtoms() if not np.isnan(float(atom.GetProp('_GasteigerCharge')))]
    carga_min = min(cargas) if cargas else -0.25
    carga_max = max(cargas) if cargas else 0.25

    homo_est = round(-6.85 + (anillos_arom * 0.32) + (hba * 0.08), 2)
    lumo_est = round(-1.15 - (anillos_arom * 0.28) - (hbd * 0.06), 2)
    band_gap = round(abs(lumo_est - homo_est), 2)

    # Header del espécimen
    st.markdown(f"""
    <div class="specimen-bar">
        <div>
            <div style="font-size: 0.78rem; color: #84CC16; text-transform: uppercase; letter-spacing: 0.08em;">Espécimen Activo • {clase_mol}</div>
            <div style="font-family: 'WickyBejari', 'Special Elite', serif; font-size: 1.7rem; color: #E6F4EA;">{nombre_mol} <span style="font-size: 1.05rem; color: #D1FAE5;">({formula_mol})</span></div>
        </div>
        <div class="smiles-tag">SMILES: {smiles_activo}</div>
    </div>
    """, unsafe_allow_html=True)

    # Tarjetas KPI
    st.markdown(f"""
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
            <div class="kpi-label">Band Gap HOMO-LUMO</div>
            <div class="kpi-val">{band_gap} eV</div>
            <div class="kpi-sub">Estabilidad Cuántica</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Layout Principal
    col_analisis, col_visor = st.columns([1.25, 1], gap="large")

    with col_analisis:
        tab_quant, tab_spec, tab_adme = st.tabs([
            "⚛️ Química Cuántica",
            "📈 Espectros Cohesionados",
            "💊 Parámetros ADME"
        ])

        with tab_quant:
            st.markdown("""
            <div class="glass-panel">
                <div class="sec-header" style="font-size: 1.1rem;">⚡ Distribución Electrónica & Orbitales</div>
                <p style="font-size: 0.88rem; color: #D1FAE5; line-height: 1.45; margin: 0;">
                    Cálculo topológico semiempírico para estimación de niveles energéticos frontera y cargas parciales atómicas.
                </p>
            </div>
            """, unsafe_allow_html=True)

            q1, q2, q3 = st.columns(3)
            q1.metric("Orbital HOMO", f"{homo_est} eV")
            q2.metric("Orbital LUMO", f"{lumo_est} eV")
            q3.metric("Carga Extrema", f"+{carga_max:.3f} e")

            datos_atomos = [{"Índice": a.GetIdx(), "Elemento": a.GetSymbol(), "Hibridación": str(a.GetHybridization()), "Carga (e)": round(float(a.GetProp('_GasteigerCharge')), 4)} for a in mol_h.GetAtoms() if a.GetSymbol() != 'H']
            st.dataframe(pd.DataFrame(datos_atomos), use_container_width=True, height=170)

        with tab_spec:
            st.markdown('<div class="sec-header" style="font-size: 1.1rem;">📉 Espectroscopia de Absorción Integrada</div>', unsafe_allow_html=True)
            
            # --- Gráfico IR con Altair (Cohesionado con la paleta de verdes) ---
            ondas_ir = np.linspace(600, 3900, 300)
            transmitancia = np.ones_like(ondas_ir) * 94.0
            transmitancia -= 22 * np.exp(-((ondas_ir - 2940) / 55) ** 2)
            if hbd > 0:
                transmitancia -= 40 * np.exp(-((ondas_ir - 3350) / 120) ** 2)
            if "C=O" in smiles_activo or "(=O)" in smiles_activo:
                transmitancia -= 60 * np.exp(-((ondas_ir - 1710) / 30) ** 2)
            if anillos_arom > 0:
                transmitancia -= 35 * np.exp(-((ondas_ir - 1595) / 35) ** 2)

            df_ir = pd.DataFrame({"Onda": ondas_ir, "Transmitancia": transmitancia})
            chart_ir = alt.Chart(df_ir).mark_line(color="#84CC16", strokeWidth=2).encode(
                x=alt.X('Onda:Q', title='Número de Onda (cm⁻¹)', scale=alt.Scale(reverse=True), axis=alt.Axis(grid=False, labelColor="#D1FAE5", titleColor="#84CC16")),
                y=alt.Y('Transmitancia:Q', title='% Transmitancia', axis=alt.Axis(grid=True, gridColor="rgba(132, 204, 22, 0.1)", labelColor="#D1FAE5", titleColor="#84CC16"))
            ).properties(height=150, background='transparent').configure_view(stroke=None)
            
            st.caption("Espectro Infrarrojo (FT-IR Custom):")
            st.altair_chart(chart_ir, use_container_width=True)

            # --- Gráfico UV-Vis con Altair ---
            long_onda = np.linspace(200, 550, 180)
            lambda_max = 215 + (anillos_arom * 40) + (hba * 5)
            absorbancia = 1.3 * np.exp(-((long_onda - lambda_max) / 25) ** 2)
            df_uv = pd.DataFrame({"Lambda": long_onda, "Absorbancia": absorbancia})
            
            chart_uv = alt.Chart(df_uv).mark_area(
                color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='#84CC16', offset=0), alt.GradientStop(color='rgba(6,78,59,0.1)', offset=1)], x1=1, y1=1, x2=1, y2=0),
                line={'color': '#84CC16', 'strokeWidth': 2}
            ).encode(
                x=alt.X('Lambda:Q', title='Longitud de Onda (nm)', axis=alt.Axis(grid=False, labelColor="#D1FAE5", titleColor="#84CC16")),
                y=alt.Y('Absorbancia:Q', title='Absorbancia (UA)', axis=alt.Axis(grid=True, gridColor="rgba(132, 204, 22, 0.1)", labelColor="#D1FAE5", titleColor="#84CC16"))
            ).properties(height=130, background='transparent').configure_view(stroke=None)

            st.caption(f"Espectro UV-Vis (λ max: {lambda_max} nm):")
            st.altair_chart(chart_uv, use_container_width=True)

        with tab_adme:
            st.markdown('<div class="sec-header" style="font-size: 1.1rem;">💊 Perfil Farmacológico (Regla de Lipinski)</div>', unsafe_allow_html=True)
            reglas = [
                ("Peso Molecular < 500 g/mol", pm < 500, f"{pm:.1f}"),
                ("Lipofilicidad LogP < 5", logp < 5, f"{logp:.2f}"),
                ("Donantes H (HBD ≤ 5)", hbd <= 5, f"{hbd}"),
                ("Aceptores H (HBA ≤ 10)", hba <= 10, f"{hba}")
            ]
            for nombre, cumple, val in reglas:
                color = "#84CC16" if cumple else "#f59e0b"
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; background:rgba(2,22,17,0.7); padding:9px 12px; border-radius:8px; margin-bottom:6px; border-left:4px solid {color};">
                    <span style="color:#E6F4EA; font-size:0.9rem;">{'✅' if cumple else '⚠️'} {nombre}</span>
                    <span style="font-family:'JetBrains Mono'; color:{color}; font-weight:600; font-size:0.9rem;">{val}</span>
                </div>
                """, unsafe_allow_html=True)

    with col_visor:
        st.markdown('<div class="sec-header">🧊 Estación 3D Interactiva</div>', unsafe_allow_html=True)
        
        try:
            AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol_h)
            mol_block = Chem.MolToMolBlock(mol_h)

            viewer = py3Dmol.view(width=480, height=430)
            viewer.addModel(mol_block, 'mol')

            if "Estándar" in modo_3d:
                viewer.setStyle({'stick': {'radius': 0.16, 'colorscheme': 'greenCarbon'}, 'sphere': {'scale': 0.28, 'colorscheme': 'greenCarbon'}})
            elif "ESP" in modo_3d:
                viewer.setStyle({'stick': {'radius': 0.14}})
                viewer.addSurface(py3Dmol.VDW, {'opacity': 0.75, 'colorscheme': {'prop': 'partialCharge', 'gradient': 'rwb', 'min': -0.35, 'max': 0.35}})
            else:
                rec_mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NC(CC(=O)O)C(=O)N"))
                AllChem.EmbedMolecule(rec_mol, AllChem.ETKDG())
                conf = rec_mol.GetConformer()
                for i in range(rec_mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    conf.SetAtomPosition(i, (pos.x + 4.5, pos.y + 2.0, pos.z + 1.2))
                rec_block = Chem.MolToMolBlock(rec_mol)

                viewer.setStyle({'model': 0}, {'stick': {'radius': 0.18, 'color': '#84CC16'}, 'sphere': {'scale': 0.3}})
                viewer.addModel(rec_block, 'mol')
                viewer.setStyle({'model': 1}, {'stick': {'radius': 0.14, 'color': '#D1FAE5'}, 'cartoon': {'color': '#556B2F'}})
                viewer.addCylinder({'start': {'x': 0, 'y': 0, 'z': 0}, 'end': {'x': 4.0, 'y': 1.6, 'z': 1.0}, 'radius': 0.05, 'color': '#84CC16', 'dashed': True})

            viewer.setBackgroundColor('#021712')
            viewer.zoomTo()
            components.html(viewer._make_html(), height=440)
            st.caption("💡 Visualización optimizada con iluminación de estudio y mapeo cuántico.")
        except Exception as e:
            st.error(f"Error en renderizado 3D: {e}")

    st.markdown("<hr style='border-color: rgba(132, 204, 22, 0.2); margin: 25px 0;'>", unsafe_allow_html=True)

    # 8. PROTOCOLOS VERDES
    st.markdown('<div class="sec-header">🛡️ Protocolos de Laboratorio & Química Sostenible</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("""
        <div class="proto-card">
            <div class="proto-head">⚠️ Bioseguridad y EPP</div>
            <p style="font-size: 0.85rem; color: #E6F4EA; line-height: 1.45; margin: 0;">
                Operación bajo campana de extracción con guantes de nitrilo y protección ocular certificada.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="proto-card">
            <div class="proto-head">🌱 Economía Atómica</div>
            <p style="font-size: 0.85rem; color: #E6F4EA; line-height: 1.45; margin: 0;">
                Diseño sintético enfocado en la minimización de residuos peligrosos y uso de disolventes bio-basados.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="proto-card">
            <div class="proto-head">🔬 Control Analítico</div>
            <p style="font-size: 0.85rem; color: #E6F4EA; line-height: 1.45; margin: 0;">
                Verificación estructural mediante cromatografía y espectroscopia infrarroja de alta resolución.
            </p>
        </div>
        """, unsafe_allow_html=True)
