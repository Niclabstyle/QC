import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import py3Dmol
import base64

# 1. Configuración de página
st.set_page_config(
    page_title="PharmaChem AI | Molecular Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar la fuente TTF local en Base64 para el CSS
def load_font_base64(font_path):
    try:
        with open(font_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return ""

font_base64 = load_font_base64("NCLWickyBejari-Regular.ttf")

# 2. CSS Avanzado envuelto correctamente en <style>
st.markdown(f"""
<style>
    @font-face {{
        font-family: 'WickyBejari';
        src: url(data:font/ttf;charset=utf-8;base64,{font_base64}) format('truetype');
        font-weight: normal;
        font-style: normal;
    }}

    .stApp {{
        background: radial-gradient(circle at 15% 15%, #16213e 0%, #0b0f19 45%, #07090f 100%);
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: #0d121f;
        border-right: 1px solid rgba(0, 245, 212, 0.15);
    }}
    
    /* Aplicar la fuente Wicky Bejari a los títulos principales */
    .hero-title, h4, h5, .protocol-title {{
        font-family: 'WickyBejari', serif !important;
        letter-spacing: 0.08em;
        font-size: 2.8rem;
    }}

    p, span, label, div, .stMarkdown {{
        color: #e2e8f0 !important;
    }}
    
    .hero-container {{
        background: linear-gradient(135deg, rgba(0, 245, 212, 0.08) 0%, rgba(123, 44, 191, 0.12) 100%);
        border: 1px solid rgba(0, 245, 212, 0.3);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
        backdrop-filter: blur(10px);
    }}
    
    .hero-title {{
        background: linear-gradient(90deg, #00f5d4 0%, #7b2cbf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }}
    
    .hero-subtitle {{
        color: #cbd5e1 !important;
        font-size: 1.05rem;
        line-height: 1.5;
    }}
    
    div[data-testid="stMetric"] {{
        background: rgba(20, 28, 46, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 245, 212, 0.3);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}
    
    div[data-testid="stMetricLabel"] {{
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    div[data-testid="stMetricValue"] {{
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        font-family: 'WickyBejari', serif !important;
    }}
    
    .protocol-card {{
        background: rgba(15, 23, 42, 0.9);
        border-left: 4px solid #00f5d4;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border-top: 1px solid rgba(255,255,255,0.1);
        border-right: 1px solid rgba(255,255,255,0.1);
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }}
    
    .protocol-warning {{
        border-left-color: #f59e0b;
    }}
    
    .protocol-title {{
        font-weight: 700;
        color: #00f5d4 !important;
        font-size: 1.1rem;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    .protocol-warning .protocol-title {{
        color: #fbbf24 !important;
    }}
</style>
""", unsafe_allow_html=True)

# 3. Abrebocas / Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧬 PharmaChem AI Platform</div>
    <div class="hero-subtitle">
        Plataforma quimiinformática de alto rendimiento diseñada para la automatización del análisis molecular, 
        evaluación de propiedades farmacocinéticas y generación de protocolos analíticos en tiempo real. 
        Ideal para investigación computacional y desarrollo farmacéutico avanzado.
    </div>
</div>
""", unsafe_allow_html=True)

# Catálogo inteligente de moléculas
nombres_conocidos = {
    "CC(=O)OC1=CC=CC=C1C(=O)O": "Aspirina (Ácido acetilsalicílico)",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": "Cafeína",
    "CC(=O)NC1=CC=C(O)C=C1": "Paracetamol (Acetaminofén)",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": "Ibuprofeno",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C": "Testosterona"
}

# 4. Panel Lateral de Control
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
st.sidebar.info("💡 **Consejo:** Puedes rotar, hacer zoom e inspeccionar la geometría tridimensional en tiempo real en el panel derecho.")

# 5. Procesamiento Principal
mol = Chem.MolFromSmiles(smiles_input)

if mol is None:
    st.error("❌ Error de sintaxis molecular: El código SMILES ingresado no es válido.")
else:
    nombre_molecula = nombres_conocidos.get(smiles_input, "Compuesto Personalizado / Investigacional")
    
    pm = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    st.markdown(f"#### 🔬 Espécimen Activo: **{nombre_molecula}**")
    st.code(f"SMILES // {smiles_input}", language="text")

    col_izq, col_der = st.columns([1.3, 1], gap="large")

    with col_izq:
        tab_metricas, tab_aplicacion = st.tabs(["📊 Métricas Fisicoquímicas", "🎯 Viabilidad & Lipinski"])
        
        with tab_metricas:
            m1, m2, m3 = st.columns(3)
            m1.metric("Peso Molecular", f"{pm:.2f} g/mol")
            m2.metric("LogP (Lipofilicidad)", f"{logp:.2f}")
            m3.metric("TPSA (Área Polar)", f"{tpsa:.2f} Å²")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Donantes de enlaces H (HBD):** {hbd} &nbsp;&nbsp;|&nbsp;&nbsp; **Aceptores (HBA):** {hba}")

        with tab_aplicacion:
            lipinski_ok = (pm < 500 and logp < 5 and hbd <= 5 and hba <= 10)
            if lipinski_ok and tpsa < 140:
                st.success("✅ **Candidato Farmacológico Óptimo:** Cumple con la Regla de Lipinski y umbrales óptimos de absorción oral.")
            elif logp > 2 and pm < 400:
                st.info("🌾 **Perfil Agroquímico:** Excelente afinidad lipídica y penetración de membranas no polares.")
            else:
                st.warning("🧪 **Materiales / Síntesis Avanzada:** Estructura compleja fuera del espacio farmacológico oral estándar.")

    with col_der:
        st.markdown("##### ⚛️ Visor Tridimensional Interactivo")
        
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol_3d)
        mol_block = Chem.MolToMolBlock(mol_3d)
        
        viewer = py3Dmol.view(width=420, height=310)
        viewer.addModel(mol_block, 'mol')
        viewer.setStyle({'stick': {'radius': 0.16}, 'sphere': {'scale': 0.3}})
        viewer.setBackgroundColor('#0b0f19')
        viewer.zoomTo()
        
        html_3d = viewer._make_html()
        components.html(html_3d, height=320, width=430)

    st.markdown("---")

    st.markdown("#### 🛡️ Protocolo Táctico de Laboratorio & Control Analítico")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("""
        <div class="protocol-card protocol-warning">
            <div class="protocol-title">⚠️ Bioseguridad & EPP</div>
            Manipulación obligatoria bajo campana extractora con equipo de protección integral (guantes de nitrilo y gafas).
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="protocol-card">
            <div class="protocol-title">📊 Control Analítico</div>
            Monitoreo por Cromatografía en Capa Fina (CCF) y confirmación estructural mediante espectroscopia FT-IR.
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="protocol-card">
            <div class="protocol-title">🔄 Estrategia de Purificación</div>
            Recristalización controlada por gradiente térmico o cromatografía flash en columna según polaridad.
        </div>
        """, unsafe_allow_html=True)
    
