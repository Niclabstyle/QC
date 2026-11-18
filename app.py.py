import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import py3Dmol
import base64
import os

# 1. CONFIGURACIÓN DE PÁGINA (Ancho y sin barras innecesarias)
st.set_page_config(
    page_title="PharmaChem AI | Molecular Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CARGA DE FUENTE PERSONALIZADA (NCLWickyBejari-Regular.ttf)
def get_font_base64(font_path):
    if os.path.exists(font_path):
        try:
            with open(font_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            st.sidebar.warning(f"Error al cargar fuente: {e}")
    return ""

font_base64 = get_font_base64("NCLWickyBejari-Regular.ttf")

# Configurar @font-face si la fuente base64 existe, si no, se usará un fallback elegante
font_face_rule = ""
if font_base64:
    font_face_rule = f"""
    @font-face {{
        font-family: 'WickyBejari';
        src: url(data:font/ttf;charset=utf-8;base64,{font_base64}) format('truetype');
        font-weight: normal;
        font-style: normal;
    }}
    """

# 3. CSS ULTRA-OPTIMIZADO (Efectos de neón, tipografías antiguas y modo cyber-obscuro)
st.markdown(f"""
<style>
    {font_face_rule}
    @import url('https://fonts.googleapis.com/css2?family=Special+Elite&family=Inter:wght@300;400;600;700&family=Playfair+Display:ital,wght@0,600;1,400&display=swap');

    /* Estilo de la App Principal */
    .stApp {{
        background: radial-gradient(circle at 50% 10%, #151a30 0%, #080b12 50%, #030406 100%) !important;
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
    }}
    
    /* Barra Lateral */
    [data-testid="stSidebar"] {{
        background-color: #06090e !important;
        border-right: 2px solid rgba(0, 245, 212, 0.2) !important;
    }}
    
    /* Reajuste de la tipografía "Old Print" en los Títulos */
    .hero-title {{
        font-family: 'WickyBejari', 'Special Elite', 'Playfair Display', serif !important;
        font-size: 3.2rem !important;
        font-weight: normal !important;
        background: linear-gradient(90deg, #00f5d4 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        letter-spacing: 0.05em;
        text-shadow: 0 0 15px rgba(0,245,212,0.1);
    }}
    
    .section-title, h4, h5 {{
        font-family: 'WickyBejari', 'Special Elite', 'Playfair Display', serif !important;
        color: #00f5d4 !important;
        letter-spacing: 0.04em;
        font-weight: normal !important;
        margin-bottom: 12px;
    }}
    
    /* Contenedor del Abrebocas / Hero */
    .hero-container {{
        background: linear-gradient(135deg, rgba(0, 245, 212, 0.05) 0%, rgba(168, 85, 247, 0.08) 100%);
        border: 1px solid rgba(0, 245, 212, 0.2);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }}
    
    .hero-subtitle {{
        color: #cbd5e1 !important;
        font-size: 1.05rem;
        line-height: 1.6;
        font-weight: 300;
    }}

    /* Tarjeta de Especificaciones Activas (SMILES) */
    .spec-banner {{
        background: rgba(13, 18, 30, 0.85);
        border-left: 5px solid #00f5d4;
        border-top: 1px solid rgba(255, 255, 255, 0.03);
        border-right: 1px solid rgba(255, 255, 255, 0.03);
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }}
    
    .smiles-text {{
        font-family: 'Special Elite', 'Courier New', monospace !important;
        color: #a855f7 !important;
        font-size: 1.05rem;
        background: rgba(0,0,0,0.4);
        padding: 8px 12px;
        border-radius: 4px;
        border: 1px solid rgba(168, 85, 247, 0.2);
        word-break: break-all;
    }}

    /* Estilización Premium para Métricas (Mesa de Datos) */
    div[data-testid="stMetric"] {{
        background: linear-gradient(145deg, rgba(16, 23, 38, 0.8) 0%, rgba(10, 14, 23, 0.9) 100%) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(0, 245, 212, 0.18) !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255,255,255,0.03) !important;
        transition: all 0.3s ease !important;
    }}
    
    div[data-testid="stMetric"]:hover {{
        border-color: rgba(0, 245, 212, 0.6) !important;
        box-shadow: 0 10px 30px rgba(0, 245, 212, 0.15) !important;
        transform: translateY(-3px);
    }}
    
    div[data-testid="stMetricLabel"] {{
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        font-weight: 600 !important;
    }}
    
    div[data-testid="stMetricValue"] {{
        color: #ffffff !important;
        font-size: 2.1rem !important;
        font-weight: normal !important;
        font-family: 'WickyBejari', 'Special Elite', 'Playfair Display', serif !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.1);
    }}

    /* Estilización de Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px 8px 0 0 !important;
        color: #94a3b8 !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        color: #00f5d4 !important;
        background-color: rgba(0, 245, 212, 0.05) !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: rgba(0, 245, 212, 0.1) !important;
        border-color: rgba(0, 245, 212, 0.3) !important;
        color: #00f5d4 !important;
    }}

    /* Tarjetas de Protocolos */
    .protocol-card {{
        background: rgba(13, 18, 30, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-left: 4px solid #00f5d4;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }}
    
    .protocol-warning {{
        border-left-color: #fbbf24;
    }}
    
    .protocol-title {{
        font-family: 'WickyBejari', 'Special Elite', serif !important;
        font-size: 1.15rem;
        margin-bottom: 8px;
        letter-spacing: 0.03em;
    }}

    /* Tarjetas para la Barra Lateral y Consejos */
    .sidebar-card {{
        background: rgba(0, 245, 212, 0.03);
        border: 1px solid rgba(0, 245, 212, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-top: 20px;
    }}

    /* Personalización para los Inputs en barra lateral */
    div[data-baseweb="select"] {{
        background-color: #0c0f16 !important;
    }}
    
    input {{
        background-color: #0c0f16 !important;
        color: #ffffff !important;
    }}
</style>
""", unsafe_allow_html=True)

# 4. ENCABEZADO Y PRESENTACIÓN
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧬 PharmaChem AI Platform</div>
    <div class="hero-subtitle">
        Plataforma quimiinformática de alto rendimiento diseñada para la automatización de la caracterización molecular, 
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

# 5. PANEL LATERAL (PARÁMETROS DE ENTRADA)
st.sidebar.markdown('<div class="section-title">⚙️ Parámetros</div>', unsafe_allow_html=True)

opcion = st.sidebar.selectbox(
    "Seleccionar espécimen molecular:",
    options=list(nombres_conocidos.keys()) + ["Personalizado"],
    format_func=lambda x: nombres_conocidos.get(x, "✍️ SMILES personalizado...")
)

if opcion == "Personalizado":
    smiles_input = st.sidebar.text_input("Cadena SMILES:", "CCO")
else:
    smiles_input = opcion

# Tarjeta de Consejo/Aviso en la barra lateral estilizada en sintonía
st.sidebar.markdown("""
<div class="sidebar-card">
    <span style="color: #00f5d4; font-weight: bold; font-family: 'Special Elite', monospace;">💡 CONSEJO</span><br>
    <p style="font-size: 0.85rem; color: #94a3b8; margin: 6px 0 0 0; line-height: 1.4;">
        Puedes rotar, hacer zoom e inspeccionar la geometría tridimensional e interactiva en tiempo real en el panel derecho de la aplicación.
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br><hr style='border-color: rgba(255,255,255,0.05)'>", unsafe_allow_html=True)
st.sidebar.caption("Motor de cálculo: **RDKit & py3Dmol Engine v2.0**")

# 6. CÁLCULO Y LOGÍSTICA
mol = Chem.MolFromSmiles(smiles_input)

if mol is None:
    st.error("❌ Error de sintaxis molecular: El código SMILES ingresado no es válido.")
else:
    nombre_molecula = nombres_conocidos.get(smiles_input, "Compuesto Personalizado")
    
    # Cálculos Fisicoquímicos
    pm = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    # Banner del compuesto seleccionado
    st.markdown(f"""
    <div class="spec-banner">
        <div class="section-title" style="margin-bottom: 2px;">🔬 Espécimen Activo: {nombre_molecula}</div>
        <div class="smiles-text">SMILES // {smiles_input}</div>
    </div>
    """, unsafe_allow_html=True)

    # 7. FILAS DE CONTENIDO (Dividido en 2 Columnas)
    col_izq, col_der = st.columns([1.2, 1], gap="large")

    with col_izq:
        # Pestañas de análisis a la izquierda
        tab_metricas, tab_aplicacion = st.tabs(["📊 Métricas Fisicoquímicas", "🎯 Viabilidad & Lipinski"])
        
        with tab_metricas:
            m1, m2, m3 = st.columns(3)
            m1.metric("PESO MOLECULAR", f"{pm:.2f} g/mol")
            m2.metric("LOGP (LIPOFILICIDAD)", f"{logp:.2f}")
            m3.metric("TPSA (ÁREA POLAR)", f"{tpsa:.2f} Å²")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.02); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                📝 <b>Donantes de enlaces de Hidrógeno (HBD):</b> {hbd} &nbsp;&nbsp;|&nbsp;&nbsp; 
                <b>Aceptores de Hidrógeno (HBA):</b> {hba}
            </div>
            """, unsafe_allow_html=True)

        with tab_aplicacion:
            lipinski_ok = (pm < 500 and logp < 5 and hbd <= 5 and hba <= 10)
            if lipinski_ok and tpsa < 140:
                st.markdown("""
                <div style="background: rgba(0, 245, 212, 0.08); border: 1px solid rgba(0, 245, 212, 0.3); border-radius: 8px; padding: 18px;">
                    <span style="color: #00f5d4; font-weight: bold; font-size: 1.1rem;">✅ Candidato Farmacológico Óptimo</span><br>
                    <p style="color: #cbd5e1; margin-top: 6px; font-size: 0.92rem; line-height: 1.5;">
                        Esta molécula cumple estrictamente con la Regla de Lipinski (Peso < 500 g/mol, LogP < 5) 
                        y tiene niveles de Área Superficial Polar Topológica (TPSA) idóneos para una excelente biodisponibilidad oral.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif logp > 2 and pm < 400:
                st.markdown("""
                <div style="background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 8px; padding: 18px;">
                    <span style="color: #c084fc; font-weight: bold; font-size: 1.1rem;">🌾 Idóneo para Agroquímica</span><br>
                    <p style="color: #cbd5e1; margin-top: 6px; font-size: 0.92rem; line-height: 1.5;">
                        La molécula presenta un perfil adecuado para la penetración en membranas celulares complejas y de plantas, 
                        con gran afinidad por entornos lipídicos, lo que la orienta hacia el área agroquímica o el desarrollo de pesticidas/defensivos.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(251, 191, 36, 0.08); border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 8px; padding: 18px;">
                    <span style="color: #fca5a5; font-weight: bold; font-size: 1.1rem;">🧪 Materiales & Síntesis Compleja</span><br>
                    <p style="color: #cbd5e1; margin-top: 6px; font-size: 0.92rem; line-height: 1.5;">
                        La geometría o los índices moleculares están fuera de los parámetros de accesibilidad oral tradicionales. 
                        Su valor reside en el desarrollo de biomateriales, polímeros avanzados o áreas de investigación biotecnológica específica.
                    </p>
                </div>
                """, unsafe_allow_html=True)

    with col_der:
        st.markdown('<div class="section-title">⚛️ Visor Tridimensional Interactivo</div>', unsafe_allow_html=True)
        
        # Preparar coordenadas 3D para renderizado dinámico
        try:
            mol_3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol_3d)
            mol_block = Chem.MolToMolBlock(mol_3d)
            
            # Generar el bloque interactivo con py3Dmol
            viewer = py3Dmol.view(width=480, height=310)
            viewer.addModel(mol_block, 'mol')
            viewer.setStyle({'stick': {'radius': 0.16}, 'sphere': {'scale': 0.32}})
            viewer.setBackgroundColor('#080b12') # Fondo coincidente con el gradiente de la UI
            viewer.zoomTo()
            
            html_3d = viewer._make_html()
            components.html(html_3d, height=320)
        except Exception as e:
            st.error(f"Error al calcular la geometría 3D: {e}")

    st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 30px 0;'>", unsafe_allow_html=True)

    # 8. SECCIÓN INFERIOR (TARJETAS DE PROTOCOLOS DE LABORATORIO)
    st.markdown('<div class="section-title" style="font-size: 1.4rem;">🛡️ Protocolo de Seguridad & Control de Laboratorio</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("""
        <div class="protocol-card protocol-warning">
            <div class="protocol-title" style="color: #fbbf24 !important;">⚠️ Bioseguridad & EPP</div>
            <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.5; margin: 0;">
                Manipulación estrictamente obligatoria bajo campana extractora certificada. 
                Utilizar el equipo de protección individual (EPP) completo: bata de algodón, 
                guantes de nitrilo y gafas de seguridad resistentes a salpicaduras.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="protocol-card">
            <div class="protocol-title">📊 Control Analítico</div>
            <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.5; margin: 0;">
                Monitoreo obligado por Cromatografía en Capa Fina (CCF / Fase estacionaria de Sílice) 
                con sistemas de elución específicos. Confirmación estructural requerida mediante espectroscopia 
                infrarroja FT-IR o resonancia RMN según la pureza.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="protocol-card">
            <div class="protocol-title">🔄 Purificación Sugerida</div>
            <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.5; margin: 0;">
                Evaluar el protocolo de recristalización en un sistema de disolventes apropiado según polaridad. 
                Si la mezcla presenta impurezas complejas, proceder con cromatografía en columna de media presión (flash).
            </p>
        </div>
        """, unsafe_allow_html=True)
