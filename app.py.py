import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
import py3Dmol
import numpy as np
import pandas as pd
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

# 3. SISTEMA DE DISEÑO CSS AVANZADO (Paleta Green Lab & Glassmorphism)
st.markdown(f"""
<style>
    {font_face_rule}
    @import url('https://fonts.googleapis.com/css2?family=Special+Elite&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {{
        --bg-deep: #02140f;
        --pine-dark: #042f24;
        --pine-main: #064E3B;
        --olive-mid: #556B2F;
        --olive-light: #6B8E23;
        --lime-neon: #84CC16;
        --mint-soft: #D1FAE5;
        --mint-pale: #E6F4EA;
    }}

    /* Fondo Global */
    .stApp {{
        background: radial-gradient(circle at 20% 10%, #064E3B 0%, #03221a 40%, #010d0a 100%) !important;
        color: #E6F4EA !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* Ocultar decoración superior nativa de Streamlit */
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Barra Lateral */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #021a14 0%, #010d0a 100%) !important;
        border-right: 1px solid rgba(132, 204, 22, 0.25) !important;
    }}

    /* Tipografías Old Print */
    .brand-title {{
        font-family: 'WickyBejari', 'Special Elite', serif !important;
        font-size: 3.1rem !important;
        background: linear-gradient(90deg, #84CC16 0%, #D1FAE5 60%, #6B8E23 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.05em;
        line-height: 1.1;
        margin-bottom: 8px;
    }}

    .sec-header {{
        font-family: 'WickyBejari', 'Special Elite', serif !important;
        font-size: 1.45rem !important;
        color: #84CC16 !important;
        letter-spacing: 0.04em;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* Hero Banner */
    .hero-box {{
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.65) 0%, rgba(85, 107, 47, 0.25) 100%);
        border: 1px solid rgba(132, 204, 22, 0.35);
        border-radius: 18px;
        padding: 26px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(209, 250, 229, 0.15);
    }}

    .hero-desc {{
        color: #D1FAE5 !important;
        font-size: 1.02rem;
        line-height: 1.6;
        max-width: 900px;
    }}

    .badge-pill {{
        display: inline-block;
        background: rgba(132, 204, 22, 0.15);
        border: 1px solid #84CC16;
        color: #84CC16;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    /* Tarjetas de Métricas Custom HTML */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 20px;
    }}

    .kpi-card {{
        background: linear-gradient(160deg, rgba(6, 78, 59, 0.75) 0%, rgba(2, 26, 20, 0.9) 100%);
        border: 1px solid rgba(209, 250, 229, 0.18);
        border-top: 3px solid #84CC16;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}

    .kpi-card:hover {{
        transform: translateY(-3px);
        border-color: #84CC16;
    }}

    .kpi-label {{
        color: #D1FAE5;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.85;
        margin-bottom: 6px;
    }}

    .kpi-val {{
        font-family: 'WickyBejari', 'Special Elite', serif;
        font-size: 1.95rem;
        color: #84CC16;
        line-height: 1.1;
    }}

    .kpi-sub {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #E6F4EA;
        opacity: 0.7;
        margin-top: 4px;
    }}

    /* Contenedor de Espécimen Activo */
    .specimen-bar {{
        background: rgba(2, 26, 20, 0.85);
        border: 1px solid rgba(132, 204, 22, 0.3);
        border-left: 6px solid #84CC16;
        border-radius: 12px;
        padding: 16px 22px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }}

    .smiles-tag {{
        font-family: 'JetBrains Mono', monospace;
        background: rgba(209, 250, 229, 0.1);
        color: #D1FAE5;
        padding: 6px 14px;
        border-radius: 8px;
        border: 1px solid rgba(209, 250, 229, 0.25);
        font-size: 0.9rem;
    }}

    /* Paneles Glassmorphism */
    .glass-panel {{
        background: rgba(6, 78, 59, 0.38);
        border: 1px solid rgba(209, 250, 229, 0.15);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 18px;
    }}

    /* Tarjetas de Protocolo */
    .proto-card {{
        background: linear-gradient(145deg, rgba(6, 78, 59, 0.65), rgba(85, 107, 47, 0.25));
        border: 1px solid rgba(132, 204, 22, 0.25);
        border-radius: 14px;
        padding: 18px;
        height: 100%;
    }}

    .proto-head {{
        font-family: 'WickyBejari', 'Special Elite', serif;
        color: #84CC16;
        font-size: 1.15rem;
        margin-bottom: 8px;
    }}

    /* Estilización de Pestañas de Streamlit */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(2, 20, 15, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(132, 204, 22, 0.2);
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        color: #D1FAE5 !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: #064E3B !important;
        color: #84CC16 !important;
        border: 1px solid #84CC16 !important;
    }}

    /* Inputs y Selectboxes */
    div[data-baseweb="select"] > div, input {{
        background-color: #03221a !important;
        color: #E6F4EA !important;
        border-color: rgba(132, 204, 22, 0.4) !important;
    }}

    label, .stRadio label {{
        color: #D1FAE5 !important;
    }}
</style>
""", unsafe_allow_html=True)

# 4. HERO SECTION (ABREBOCAS)
st.markdown("""
<div class="hero-box">
    <div class="badge-pill">⚡ Quantum Cheminformatics & Green Lab v4.0</div>
    <div class="brand-title">🌿 PharmaChem AI Studio</div>
    <div class="hero-desc">
        Estación de trabajo computacional para el diseño molecular sostenible, simulación de potencial electrostático (ESP), 
        acoplamiento bimolecular (Docking) y predicción espectroscópica en tiempo real.
    </div>
</div>
""", unsafe_allow_html=True)

# 5. BASE DE DATOS AMPLIADA DE COMPUESTOS
base_moleculas = {
    "CC(=O)OC1=CC=CC=C1C(=O)O": ("Aspirina", "Analgésico / Antiinflamatorio", "C9H8O4"),
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": ("Cafeína", "Alcaloide / Estimulante SNC", "C8H10N4O2"),
    "CC(=O)NC1=CC=C(O)C=C1": ("Paracetamol", "Antipirético / Analgésico", "C8H9NO2"),
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": ("Ibuprofeno", "AINE / Inhibidor COX", "C13H18O2"),
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C": ("Testosterona", "Hormona Esteroidea", "C19H28O2"),
    "CN[C@@H](C[C@@H](C1=CC=C(C=C1)O)O)O": ("Adrenalina (Epinefrina)", "Neurotransmisor / Catecolamina", "C9H13NO3"),
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CC=CN=C5": ("Imatinib", "Inhibidor Tirosina Quinasa", "C29H31N7O"),
    "CC(=O)O": ("Ácido Acético", "Disolvente Verde / Metabolito", "C2H4O2"),
    "C1=CC=C(C=C1)O": ("Fenol", "Precursor Polimérico", "C6H6O"),
    "CN1CCC[C@H]1C2=CN=CC=C2": ("Nicotina", "Alcaloide Agonista nAChR", "C10H14N2"),
    "CC(C)(C)NCC(O)C1=CC(CO)=C(O)C=C1": ("Salbutamol", "Agonista Beta-2 Adrenérgico", "C13H21NO3"),
    "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F": ("Celecoxib", "Inhibidor Selectivo COX-2", "C17H14F3N3O2S")
}

# 6. BARRA LATERAL CON BUSCADOR INTELIGENTE Y CONTROLES CUÁNTICOS
st.sidebar.markdown('<div class="sec-header">🔍 Explorador Molecular</div>', unsafe_allow_html=True)

query_busqueda = st.sidebar.text_input("Buscar por nombre, clase o SMILES:", placeholder="Ej: Cafeína, Imatinib, C=O...").strip().lower()

filtradas = {
    k: v for k, v in base_moleculas.items()
    if query_busqueda in v[0].lower() or query_busqueda in v[1].lower() or query_busqueda in k.lower()
}
if not filtradas:
    st.sidebar.warning("Sin coincidencias exactas. Mostrando catálogo completo.")
    filtradas = base_moleculas

seleccion = st.sidebar.selectbox(
    "Espécimen en estudio:",
    options=list(filtradas.keys()) + ["CUSTOM"],
    format_func=lambda x: f"{filtradas[x][0]} ({filtradas[x][2]})" if x in filtradas else "✍️ Ingresar SMILES personalizado..."
)

if seleccion == "CUSTOM":
    smiles_activo = st.sidebar.text_input("Estructura SMILES:", "C1=CC=C2C(=C1)C=CC=C2")
else:
    smiles_activo = seleccion

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sec-header">⚛️ Motor de Renderizado 3D</div>', unsafe_allow_html=True)

modo_3d = st.sidebar.radio(
    "Modo de visualización molecular:",
    [
        "🔬 Esferas y Varillas (Estándar)",
        "⚡ Superficie Potencial Electrostático (ESP)",
        "🔗 Simulación Docking (Interacción Ligando-Sitio)"
    ]
)

st.sidebar.markdown("""
<div style="background: rgba(209, 250, 229, 0.08); border: 1px solid #84CC16; border-radius: 12px; padding: 14px; margin-top: 16px;">
    <span style="color: #84CC16; font-weight: 700; font-size: 0.85rem;">🌿 QUÍMICA COMPUTACIONAL</span>
    <p style="color: #E6F4EA; font-size: 0.8rem; margin: 6px 0 0 0; line-height: 1.4;">
        Arrastra con el clic izquierdo para rotar en 360°, usa la rueda del ratón para hacer zoom y explora las densidades electrónicas.
    </p>
</div>
""", unsafe_allow_html=True)

# 7. PROCESAMIENTO QUÍMICO Y CUÁNTICO CON RDKIT
mol = Chem.MolFromSmiles(smiles_activo)

if mol is None:
    st.error("❌ Error de sintaxis SMILES: La cadena ingresada no corresponde a una estructura química válida.")
else:
    info_comp = base_moleculas.get(smiles_activo, ("Compuesto Investigacional", "Síntesis Personalizada", rdMolDescriptors.CalcMolFormula(mol)))
    nombre_mol, clase_mol, formula_mol = info_comp

    # Descriptores Fisicoquímicos
    pm = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    anillos_arom = rdMolDescriptors.CalcNumAromaticRings(mol)

    # Cálculos Cuánticos / Electrónicos Semiempíricos (Gasteiger & Topología Pi)
    mol_h = Chem.AddHs(mol)
    AllChem.ComputeGasteigerCharges(mol_h)
    cargas = [float(atom.GetProp('_GasteigerCharge')) for atom in mol_h.GetAtoms() if not np.isnan(float(atom.GetProp('_GasteigerCharge')))]
    carga_min = min(cargas) if cargas else -0.25
    carga_max = max(cargas) if cargas else 0.25

    # Estimación heurística de orbitales frontera (HOMO/LUMO) basada en conjugación aromática y heteroátomos
    homo_est = round(-6.85 + (anillos_arom * 0.32) + (hba * 0.08), 2)
    lumo_est = round(-1.15 - (anillos_arom * 0.28) - (hbd * 0.06), 2)
    band_gap = round(abs(lumo_est - homo_est), 2)

    # Banner del Espécimen Activo
    st.markdown(f"""
    <div class="specimen-bar">
        <div>
            <div style="font-size: 0.8rem; color: #84CC16; text-transform: uppercase; letter-spacing: 0.08em;">Espécimen Activo • {clase_mol}</div>
            <div style="font-family: 'WickyBejari', 'Special Elite', serif; font-size: 1.8rem; color: #E6F4EA;">{nombre_mol} <span style="font-size: 1.1rem; color: #D1FAE5;">({formula_mol})</span></div>
        </div>
        <div class="smiles-tag">SMILES: {smiles_activo}</div>
    </div>
    """, unsafe_allow_html=True)

    # Tarjetas KPI Superiores (HTML Puro)
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Peso Molecular</div>
            <div class="kpi-val">{pm:.2f}</div>
            <div class="kpi-sub">g/mol (Umbral &lt; 500)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Coeficiente LogP</div>
            <div class="kpi-val">{logp:.2f}</div>
            <div class="kpi-sub">Lipofilicidad Octanol/Agua</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Área Polar (TPSA)</div>
            <div class="kpi-val">{tpsa:.1f} Å²</div>
            <div class="kpi-sub">Permeabilidad Celular</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Brecha HOMO-LUMO</div>
            <div class="kpi-val">{band_gap} eV</div>
            <div class="kpi-sub">Estabilidad Cuántica</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 8. ÁREA PRINCIPAL DIVIDIDA (ANÁLISIS MULTIDISCIPLINARIO + VISOR 3D)
    col_analisis, col_visor = st.columns([1.25, 1], gap="large")

    with col_analisis:
        tab_quant, tab_spec, tab_adme = st.tabs([
            "⚛️ Química Cuántica & Orbitales",
            "📈 Espectroscopia IR / UV-Vis",
            "💊 Farmacocinética & Lipinski"
        ])

        with tab_quant:
            st.markdown("""
            <div class="glass-panel">
                <div class="sec-header" style="font-size: 1.15rem;">⚡ Estructura Electrónica y Cargas de Gasteiger</div>
                <p style="font-size: 0.9rem; color: #D1FAE5; line-height: 1.5;">
                    Análisis de distribución de densidad electrónica y reactividad de orbitales frontera para predecir sitios de ataque nucleofílico y electrofílico.
                </p>
            </div>
            """, unsafe_allow_html=True)

            q1, q2, q3 = st.columns(3)
            q1.metric("Orbital HOMO", f"{homo_est} eV", "Donador e⁻")
            q2.metric("Orbital LUMO", f"{lumo_est} eV", "Aceptor e⁻")
            q3.metric("Dipolo / Carga Max", f"+{carga_max:.3f} e", f"Min: {carga_min:.3f} e")

            # Tabla de átomos y cargas parciales
            datos_atomos = []
            for atom in mol_h.GetAtoms():
                if atom.GetSymbol() != 'H':
                    datos_atomos.append({
                        "Índice": atom.GetIdx(),
                        "Elemento": atom.GetSymbol(),
                        "Hibridación": str(atom.GetHybridization()),
                        "Aromático": "Sí" if atom.GetIsAromatic() else "No",
                        "Carga Parcial (e)": round(float(atom.GetProp('_GasteigerCharge')), 4)
                    })
            st.markdown("**Distribución Atómica y Estados de Hibridación:**")
            st.dataframe(pd.DataFrame(datos_atomos), use_container_width=True, height=190)

        with tab_spec:
            st.markdown('<div class="sec-header" style="font-size: 1.15rem;">📉 Simulación Espectral (FT-IR y UV-Vis)</div>', unsafe_allow_html=True)
            
            # Generación dinámica de espectro IR basado en grupos funcionales reales de la molécula
            ondas_ir = np.linspace(500, 4000, 350)
            transmitancia = np.ones_like(ondas_ir) * 95.0
            
            # Picos base C-H
            transmitancia -= 25 * np.exp(-((ondas_ir - 2950) / 60) ** 2)
            # Si tiene oxígeno/hidroxilo
            if hbd > 0:
                transmitancia -= 45 * np.exp(-((ondas_ir - 3380) / 140) ** 2)
            # Si tiene carbonilos o dobles enlaces
            if "C=O" in smiles_activo or "(=O)" in smiles_activo:
                transmitancia -= 65 * np.exp(-((ondas_ir - 1715) / 35) ** 2)
            # Si tiene anillos aromáticos
            if anillos_arom > 0:
                transmitancia -= 40 * np.exp(-((ondas_ir - 1600) / 40) ** 2)
                transmitancia -= 35 * np.exp(-((ondas_ir - 1490) / 35) ** 2)

            df_ir = pd.DataFrame({"Número de Onda (cm⁻¹)": ondas_ir, "Transmitancia (%)": transmitancia}).set_index("Número de Onda (cm⁻¹)")
            st.caption("Espectro Infrarrojo Simulado (FT-IR - % Transmitancia):")
            st.line_chart(df_ir, color="#84CC16", height=180)

            # Espectro UV-Vis
            long_onda = np.linspace(200, 600, 200)
            lambda_max = 220 + (anillos_arom * 38) + (hba * 6)
            absorbancia = 1.4 * np.exp(-((long_onda - lambda_max) / 28) ** 2) + 0.35 * np.exp(-((long_onda - (lambda_max - 45)) / 20) ** 2)
            df_uv = pd.DataFrame({"Longitud de Onda (nm)": long_onda, "Absorbancia (UA)": absorbancia}).set_index("Longitud de Onda (nm)")
            st.caption(f"Espectro de Absorción Electrónica UV-Vis (λ max estimado: {lambda_max} nm):")
            st.area_chart(df_uv, color="#D1FAE5", height=150)

        with tab_adme:
            st.markdown('<div class="sec-header" style="font-size: 1.15rem;">💊 Evaluación Farmacocinética (ADME)</div>', unsafe_allow_html=True)
            
            reglas = [
                ("Peso Molecular < 500 g/mol", pm < 500, f"{pm:.1f} g/mol"),
                ("Lipofilicidad LogP < 5", logp < 5, f"{logp:.2f}"),
                ("Donantes Puente H (HBD ≤ 5)", hbd <= 5, f"{hbd}"),
                ("Aceptores Puente H (HBA ≤ 10)", hba <= 10, f"{hba}"),
                ("Enlaces Rotables ≤ 10", rot_bonds <= 10, f"{rot_bonds}")
            ]

            for nombre_regla, cumple, valor in reglas:
                estado_color = "#84CC16" if cumple else "#f59e0b"
                icono = "✅" if cumple else "⚠️"
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; background:rgba(2,26,20,0.7); padding:10px 14px; border-radius:8px; margin-bottom:8px; border-left:4px solid {estado_color};">
                    <span style="color:#E6F4EA;">{icono} {nombre_regla}</span>
                    <span style="font-family:'JetBrains Mono', monospace; color:{estado_color}; font-weight:600;">{valor}</span>
                </div>
                """, unsafe_allow_html=True)

    with col_visor:
        st.markdown('<div class="sec-header">🧊 Modelamiento Molecular 3D</div>', unsafe_allow_html=True)
        
        try:
            AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol_h)
            mol_block = Chem.MolToMolBlock(mol_h)

            viewer = py3Dmol.view(width=480, height=420)
            viewer.addModel(mol_block, 'mol')

            if "Estándar" in modo_3d:
                viewer.setStyle({'stick': {'radius': 0.16, 'colorscheme': 'greenCarbon'}, 'sphere': {'scale': 0.30, 'colorscheme': 'greenCarbon'}})
            elif "ESP" in modo_3d:
                viewer.setStyle({'stick': {'radius': 0.14}})
                viewer.addSurface(py3Dmol.VDW, {'opacity': 0.72, 'colorscheme': {'prop': 'partialCharge', 'gradient': 'rwb', 'min': -0.35, 'max': 0.35}})
            else:
                # Simulación de Docking: Añadimos un residuo receptor simulado (ej. fragmento peptídico en el sitio activo)
                receptor_smiles = "CC(=O)NC(CC(=O)O)C(=O)N"
                rec_mol = Chem.AddHs(Chem.MolFromSmiles(receptor_smiles))
                AllChem.EmbedMolecule(rec_mol, AllChem.ETKDG())
                # Desplazar el receptor cerca del ligando para visualizar la interacción
                conf = rec_mol.GetConformer()
                for i in range(rec_mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    conf.SetAtomPosition(i, (pos.x + 4.8, pos.y + 2.2, pos.z + 1.5))
                rec_block = Chem.MolToMolBlock(rec_mol)

                viewer.setStyle({'model': 0}, {'stick': {'radius': 0.18, 'color': '#84CC16'}, 'sphere': {'scale': 0.3}})
                viewer.addModel(rec_block, 'mol')
                viewer.setStyle({'model': 1}, {'stick': {'radius': 0.14, 'color': '#D1FAE5'}, 'cartoon': {'color': '#556B2F'}})
                # Línea de interacción intermolecular (Puente de hidrógeno simulado)
                viewer.addCylinder({'start': {'x': 0, 'y': 0, 'z': 0}, 'end': {'x': 4.2, 'y': 1.8, 'z': 1.2}, 'radius': 0.06, 'color': '#84CC16', 'dashed': True})

            viewer.setBackgroundColor('#021a14')
            viewer.zoomTo()
            components.html(viewer._make_html(), height=430)

            st.caption("💡 **Controles Activos:** Rotación libre 3D • Optimización de campo de fuerza MMFF94 aplicada.")
        except Exception as e:
            st.error(f"Error en renderizado 3D: {e}")

    st.markdown("<hr style='border-color: rgba(132, 204, 22, 0.25); margin: 28px 0;'>", unsafe_allow_html=True)

    # 9. PROTOCOLOS DE LABORATORIO VERDE
    st.markdown('<div class="sec-header">🛡️ Protocolos de Síntesis Verde & Seguridad</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("""
        <div class="proto-card">
            <div class="proto-head">⚠️ Bioseguridad & EPP</div>
            <p style="font-size: 0.88rem; color: #E6F4EA; line-height: 1.5; margin: 0;">
                Manipulación bajo campana de extracción laminar con guantes de nitrilo de alto calibre, gafas herméticas y bata ignífuga.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="proto-card">
            <div class="proto-head">🌱 Economía Atómica</div>
            <p style="font-size: 0.88rem; color: #E6F4EA; line-height: 1.5; margin: 0;">
                Priorizar rutas catalíticas limpias y disolventes biodegradables (etanol, agua o fluidos supercríticos) reduciendo residuos tóxicos.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="proto-card">
            <div class="proto-head">🔬 Validación Analítica</div>
            <p style="font-size: 0.88rem; color: #E6F4EA; line-height: 1.5; margin: 0;">
                Confirmación de pureza mediante HPLC-MS y verificación de grupos funcionales por espectroscopia infrarroja FT-IR.
            </p>
        </div>
        """, unsafe_allow_html=True)
