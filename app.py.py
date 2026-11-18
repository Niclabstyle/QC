import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors

# Configuración de la página (Layout ancho para mejor aspecto de dashboard)
st.set_page_config(
    page_title="PharmaChem AI | Dashboard Quimiinformático", 
    page_icon="🧪", 
    layout="wide"
)

# Estilo personalizado básico con CSS para darle toque profesional
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2e6c80;
        font-weight: 700;
    }
    .sub-text {
        color: #555555;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Encabezado principal
st.markdown('<p class="main-header">🧪 PharmaChem AI Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Plataforma quimiinformática para análisis molecular, predicción de propiedades y diseño de protocolos de síntesis.</p>', unsafe_allow_html=True)
st.markdown("---")

# Diccionario inteligente de moléculas
nombres_conocidos = {
    "CC(=O)OC1=CC=CC=C1C(=O)O": "Aspirina (Ácido acetilsalicílico)",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": "Cafeína",
    "CC(=O)NC1=CC=C(O)C=C1": "Paracetamol (Acetaminofén)",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": "Ibuprofeno"
}

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.header("⚙️ Panel de Control")
st.sidebar.markdown("Selecciona o introduce la molécula a evaluar:")

opcion = st.sidebar.selectbox(
    "Molécula de referencia:",
    options=list(nombres_conocidos.keys()) + ["Personalizado"],
    format_func=lambda x: nombres_conocidos.get(x, "✍️ Introducir SMILES personalizado")
)

if opcion == "Personalizado":
    smiles_input = st.sidebar.text_input("Código SMILES:", "CCO")
else:
    smiles_input = opcion

analizar_btn = st.sidebar.button("🚀 Ejecutar Análisis", type="primary")

# --- CUERPO PRINCIPAL ---
if analizar_btn or smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    
    if mol is None:
        st.error("❌ Error crítico: El código SMILES ingresado no es válido. Verifica la estructura.")
    else:
        nombre_molecula = nombres_conocidos.get(smiles_input, "Molécula Personalizada / Investigacional")
        
        # Calcular propiedades
        pm = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # Mostrar título y SMILES
        st.subheader(f"📊 Reporte Analítico para: {nombre_molecula}")
        st.code(smiles_input, language="text")
        
        # Pestañas profesionales para organizar la información
        tab1, tab2, tab3 = st.tabs(["📈 Propiedades Físico-Químicas", "🔍 Campos de Aplicación", "🧪 Guía de Laboratorio"])
        
        with tab1:
            st.markdown("### Métricas Clave de RDKit")
            col1, col2, col3 = st.columns(3)
            col1.metric("Peso Molecular", f"{pm:.2f} g/mol", help="Masa molecular calculada")
            col2.metric("LogP (Lipofilicidad)", f"{logp:.2f}", help="Coeficiente de partición octanol/agua")
            col3.metric("TPSA (Área Polar)", f"{tpsa:.2f} Å²", help="Área superficial polar topológica")
            
        with tab2:
            st.markdown("### Evaluación de Viabilidad y Aplicación Industrial")
            campos = []
            if pm < 500 and logp < 5 and tpsa < 140:
                campos.append("💊 **Farmacéutica (Oral)**: Cumple con los filtros estándar de Lipinski (buena biodisponibilidad oral estimada).")
            if logp > 2 and pm < 400:
                campos.append("🌾 **Agroquímicos / Protección de Cultivos**: Presenta perfil lipofílico adecuado para penetración.")
            if tpsa > 100 or pm > 400:
                campos.append("🧪 **Materiales Avanzados / Polímeros**: Densidad de centros polares o tamaño molecular elevado.")
            
            if not campos:
                campos.append("🔬 **Investigación Exploratoria**: Molécula fuera de los rangos convencionales de fármacos, idónea para síntesis orgánica avanzada.")
                
            for c in campos:
                st.info(c)
                
        with tab3:
            st.markdown("### Protocolo Preliminar y Seguridad")
            st.warning("⚠️ **Normativa de Seguridad**: Manipular estrictamente bajo campana extractora de gases utilizando equipo de protección personal (EPP) completo.")
            st.success("📊 **Control Analítico Sugerido**: Cromatografía en Capa Fina (TCC) preliminar con fase estacionaria de sílica gel y eluyente optimizado, seguida de confirmación estructural mediante espectroscopia FT-IR.")
            st.info("🔄 **Ruta de Purificación**: Evaluar recristalización en disolvente adecuado o cromatografía en columna de media presión según polaridad.")
else:
    st.info("👈 Selecciona una molécula en la barra lateral y haz clic en **Ejecutar Análisis** para generar el reporte.")
