import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import py3Dmol
from stmol import showmol

# 1. Configuración de página
st.set_page_config(
    page_title="PharmaChem AI | Molecular Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS Personalizado (Dark-Tech / Glassmorphism)
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 15% 15%, #16213e 0%, #0b0f19 45%, #07090f 100%);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background-color: #0d121f;
        border-right: 1px solid rgba(0, 245, 212, 0.15);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f5d4 0%, #7b2cbf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(20, 28, 46, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 245, 212, 0.25);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
    }
    .protocol-card {
        background: rgba(15, 23, 42, 0.8);
        border-left: 4px solid #00f5d4;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .protocol-warning {
        border-left-color: #f59e0b;
    }
    .protocol-title {
        font-weight: 700;
        color: #00f5d4;
        font-size: 0.95rem;
        margin-bottom: 4px;
        text-transform: uppercase;
    }
    .protocol-warning .protocol-title {
        color: #fbbf24;
    }
</style>
""", unsafe_allow_html=True)

# 3. Encabezado Principal
st.markdown('<div class="hero-title">🧬 PharmaChem AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Plataforma de Inteligencia Molecular, Descriptores Fisicoquímicos y Visualización Tridimensional</div>', unsafe_allow_html=True)

# Catálogo de moléculas
nombres_conocidos = {
    "CC(=O)OC1=CC=CC=C1C(=O)O": "Aspirina (Ácido acetilsalicílico)",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": "Cafeína",
    "CC(=O)NC1=CC=C(O)C=C1": "Paracetamol (Acetaminofén)",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": "Ibuprofeno",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C": "Testosterona"
}

# 4. Panel Lateral
st.sidebar.markdown("### ⚙️ PARÁMETROS DE ENTRADA")
opcion = st.sidebar.selectbox(
    "Seleccionar espécimen molecular:",
    options=list(nombres_conocidos.keys()) + ["Personalizado"],
    format_func=lambda x: nombres_conocidos.get(x, "✍️ Ingresar código SMILES manual")
)

if opcion == "Personalizado":
    smiles_input = st.sidebar.text_input("Cadena SMILES:", "CCO")
else:
    smiles_input = opcion

st.sidebar.markdown("---")
st.sidebar.caption("Motor de cálculo: **RDKit & py3Dmol**")

# 5. Procesamiento
mol = Chem.MolFromSmiles(smiles_input)

if mol is None:
    st.error("❌ Error de sintaxis molecular: El código SMILES ingresado no es válido.")
else:
    nombre_molecula = nombres_conocidos.get(smiles_input, "Compuesto Personalizado")
    
    # Cálculos fisicoquímicos
    pm = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    st.markdown(f"#### 🔬 Espécimen Activo: **{nombre_molecula}**")
    st.code(f"SMILES // {smiles_input}", language="text")

    col_izq, col_der = st.columns([1.3, 1], gap="large")

    with col_izq:
        m1, m2, m3 = st.columns(3)
        m1.metric("Peso Molecular", f"{pm:.2f} g/mol")
        m2.metric("LogP (Lipofilicidad)", f"{logp:.2f}")
        m3.metric("TPSA (Área Polar)", f"{tpsa:.2f} Å²")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🎯 Predicción de Aplicación & Biodisponibilidad")
        lipinski_ok = (pm < 500 and logp < 5 and hbd <= 5 and hba <= 10)
        
        if lipinski_ok and tpsa < 140:
            st.success("✅ **Candidato Farmacológico Óptimo:** Cumple estrictamente con la Regla de los 5 de Lipinski y umbral TPSA para absorción oral.")
        elif logp > 2 and pm < 400:
            st.info("🌾 **Perfil Agroquímico / Penetración Lipídica:** Alta afinidad por membranas no polares.")
        else:
            st.warning("🧪 **Materiales Avanzados / Síntesis Especializada:** Estructura compleja fuera del espacio químico oral estándar.")

    with col_der:
        st.markdown("##### ⚛️ Visor Estructural 3D Interactivo")
        
        # Preparar geometría 3D con RDKit (añadir hidrógenos y calcular coordenadas espaciales)
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol_3d)
        mol_block = Chem.MolToMolBlock(mol_3d)
        
        # Renderizar visor 3D con py3Dmol dentro de Streamlit
        viewer = py3Dmol.view(width=400, height=300)
        viewer.addModel(mol_block, 'mol')
        viewer.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.3}})
        viewer.setBackgroundColor('#0b0f19')
        viewer.zoomTo()
        showmol(viewer, height=300, width=400)

    st.markdown("---")

    # 6. Protocolos de Laboratorio
    st.markdown("#### 🛡️ Protocolo Táctico de Laboratorio & Control Analítico")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("""
        <div class="protocol-card protocol-warning">
            <div class="protocol-title">⚠️ Bioseguridad & EPP</div>
            Manipulación obligatoria bajo campana de extracción de gases con guantes de nitrilo y protección ocular.
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="protocol-card">
            <div class="protocol-title">📊 Control Analítico</div>
            Monitoreo por Cromatografía en Capa Fina (CCF / Sílica) y verificación de grupos funcionales por FT-IR.
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="protocol-card">
            <div class="protocol-title">🔄 Estrategia de Purificación</div>
            Recristalización controlada por gradiente térmico o cromatografía flash en columna según polaridad.
        </div>
        """, unsafe_allow_html=True)
