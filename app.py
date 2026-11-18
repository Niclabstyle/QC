import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors

# Configuración de la página web
st.set_page_config(page_title="Plataforma Quimiinformática", page_icon="🧪", layout="centered")

st.title("🧪 Asistente Quimiinformático Multipropósito")
st.write("Analiza propiedades moleculares, predice campos de aplicación y obtén guías preliminares de laboratorio al instante.")

# Diccionario inteligente de moléculas
nombres_conocidos = {
    "CC(=O)OC1=CC=CC=C1C(=O)O": "Aspirina (Ácido acetilsalicílico)",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": "Cafeína",
    "CC(=O)NC1=CC=C(O)C=C1": "Paracetamol (Acetaminofén)",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": "Ibuprofeno"
}

# Selector visual en la web
opcion = st.selectbox(
    "Selecciona una molécula de ejemplo o ingresa su código SMILES:",
    options=list(nombres_conocidos.keys()) + ["Personalizado"],
    format_func=lambda x: nombres_conocidos.get(x, "✍️ Escribir mi propio código SMILES")
)

if opcion == "Personalizado":
    smiles_input = st.text_input("Ingresa el código SMILES de tu molécula:", "CCO")
else:
    smiles_input = opcion

# Botón para ejecutar el análisis
if st.button("Analizar Molécula"):
    mol = Chem.MolFromSmiles(smiles_input)
    
    if mol is None:
        st.error("❌ Error: El código SMILES no es válido. Revisa la estructura.")
    else:
        nombre_molecula = nombres_conocidos.get(smiles_input, "Molécula Personalizada")
        
        # Calcular propiedades
        pm = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # Mostrar resultados
        st.subheader(f"📊 Reporte para: {nombre_molecula}")
        st.code(smiles_input, language="text")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Peso Molecular", f"{pm:.2f} g/mol")
        col2.metric("LogP (Lipofilicidad)", f"{logp:.2f}")
        col3.metric("TPSA (Área Polar)", f"{tpsa:.2f} Å²")
        
        st.markdown("### 🔍 Campos de Aplicación Sugeridos")
        campos = []
        if pm < 500 and logp < 5 and tpsa < 140:
            campos.append("💊 **Farmacéutica** (Alto potencial de biodisponibilidad oral)")
        if logp > 2 and pm < 400:
            campos.append("🌾 **Agroquímicos / Pesticidas** (Buena penetración)")
        if tpsa > 100 or pm > 400:
            campos.append("🧪 **Polímeros / Materiales Avanzados** o Ingredientes Especiales")
        
        if not campos:
            campos.append("🔬 **Investigación general** / Síntesis orgánica avanzada")
            
        for c in campos:
            st.write(f"- {c}")
            
        st.markdown("### 🧪 Guía Preliminar de Laboratorio")
        st.warning("⚠️ Precaución general: Manipular bajo campana extractora utilizando el EPP completo requerido.")
        st.info("📊 **Control Analítico Recomendado:** Cromatografía en Capa Fina (TCC) previa y confirmación por FT-IR.")
