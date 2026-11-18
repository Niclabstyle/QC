import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

# 1. Configuración de página (Ancha y profesional)
st.set_page_config(
    page_title="PharmaChem AI | Molecular Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS Personalizado (Estética Dark-Tech / Glassmorphism)
st.markdown("""
<style>
    /* Fondo principal oscuro estilo Obsidian con destellos sutiles */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #16213e 0%, #0b0f19 45%, #07090f 100%);
        color: #e2e8f0;
    }
    
    /* Barra lateral oscura */
    [data-testid="stSidebar"] {
        background-color: #0d121f;
        border-right: 1px solid rgba(0, 245, 212, 0.15);
    }

    /* Títulos con gradiente tecnológico */
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

    /* Tarjetas Glassmorphism para Métricas */
    div[data-testid="stMetric"] {
        background: rgba(20, 28, 46, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 245, 212, 0.25);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        border-color: #00f5d4;
        transform: translateY(-2px);
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
    }

    /* Cajas de protocolo tácticas */
    .protocol-card {
        background: rgba(15, 23, 42, 0.8);
        border-left: 4px solid #00f5d4;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border-top: 1px solid rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.05);
        border-bottom: 1px solid rgba(255,255,255,0.05);
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
        letter-spacing: 0.05em;
    }
    
    .protocol-warning .protocol-title {
        color: #fbbf24;
    }
</style>
""", unsafe_allow_html=True)

# 3. Encabezado Principal
st.markdown('<div class="hero-title">🧬 PharmaChem AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Plataforma de Inteligencia Molecular, Descriptores Fisicoquímicos y Protocolos de Síntesis</div>', unsafe_allow_html=True)

# Diccionario de compuestos
nombres_conocidos = {
    "CC(=O)OC1=CC=CC=C1C(=O)O": "Aspirina (Ácido acetilsalicílico)",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": "Cafeína",
    "CC(=O)NC1=CC=C(O)C=C1": "Paracetamol (Acetaminofén)",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": "Ibuprofeno",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C": "Testosterona"
}

# 4. Panel Lateral (Inputs)
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
st.sidebar.caption("Motor de cálculo: **RDKit Cheminformatics Engine**")

# 5. Procesamiento y Renderizado
mol = Chem.MolFromSmiles(smiles_input)

if mol is None:
    st.error("❌ Error de sintaxis molecular: El código SMILES ingresado no es válido.")
else:
    nombre_molecula = nombres_conocidos.get(smiles_input, "Compuesto Personalizado")
    
    # Cálculos con RDKit
    pm = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    # Cabecera del espécimen
    st.markdown(f"#### 🔬 Espécimen Activo: **{nombre_molecula}**")
    st.code(f"SMILES // {smiles_input}", language="text")

    # Diseño en 2 columnas principales: Izquierda (Métricas y Reglas) | Derecha (Visor Molecular)
    col_izq, col_der = st.columns([1.4, 1], gap="large")

    with col_izq:
        # Fila de 3 métricas principales
        m1, m2, m3 = st.columns(3)
        m1.metric("Peso Molecular", f"{pm:.2f} g/mol")
        m2.metric("LogP (Lipofilicidad)", f"{logp:.2f}")
        m3.metric("TPSA (Área Polar)", f"{tpsa:.2f} Å²")

        st.markdown("<br>", unsafe_allow_html=True)

        # Evaluación de Aplicación Industrial
        st.markdown("##### 🎯 Predicción de Aplicación & Biodisponibilidad")
        lipinski_ok = (pm < 500 and logp < 5 and hbd <= 5 and hba <= 10)
        
        if lipinski_ok and tpsa < 140:
            st.success("✅ **Candidato Farmacológico Óptimo:** Cumple estrictamente con la Regla de los 5 de Lipinski y umbral TPSA para absorción oral.")
        elif logp > 2 and pm < 400:
            st.info("🌾 **Perfil Agroquímico / Penetración Lipídica:** Alta afinidad por membranas no polares.")
        else:
            st.warning("🧪 **Materiales Avanzados / Síntesis Especializada:** Estructura compleja fuera del espacio químico oral estándar.")

    with col_der:
        st.markdown("##### ⚛️ Estructura Molecular 2D")
        # Generar imagen de la molécula con fondo blanco limpio en tarjeta redondeada
        img = Draw.MolToImage(mol, size=(420, 300))
        st.image(img, use_container_width=True)

    st.markdown("---")

    # 6. Sección Inferior: Protocolo de Laboratorio estilo GenCherry
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
