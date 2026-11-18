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


# 4. FUNCIONES DE CÁLCULO Y API CACHEADAS
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
    """Resuelve H*c = E*c (Método de Hückel con parámetros de Streitwieser) sobre el sistema pi."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    pi_atoms = []
    pi_electrons = 0
    h_params = []

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
            if atom.GetDegree() == 1:
                n_e, h_x = 1, 1.0
            else:
                n_e, h_x = 2, 2.0
        elif sym == "S":
            if atom.GetDegree() == 1:
                n_e, h_x = 1, 0.5
            else:
                n_e, h_x = 2, 1.0
        else:
            continue

        pi_atoms.append(atom.GetIdx())
        pi_electrons += n_e
        h_params.append(h_x)

    n = len(pi_atoms)
    if n < 2:
        return None

    idx_map = {atom_idx: i for i, atom_idx in enumerate(pi_atoms)}
    # Construimos M donde H = alpha*I + beta*M; como beta < 0, diagonalizamos -M
    mat = np.zeros((n, n), dtype=float)
    for i, h_val in enumerate(h_params):
        mat[i, i] = h_val

    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in idx_map and a2 in idx_map:
            i, j = idx_map[a1], idx_map[a2]
            s1 = mol.GetAtomWithIdx(a1).GetSymbol()
            s2 = mol.GetAtomWithIdx(a2).GetSymbol()
            k_xy = 1.0 if (s1 == "C" and s2 == "C") else 0.85
            mat[i, j] = k_xy
            mat[j, i] = k_xy

    # Autovalores en orden ascendente de energía (en unidades de |beta|, con alpha = -6.0 eV y |beta| = 2.4 eV)
    eigvals, eigvecs = np.linalg.eigh(-mat)

    alpha_ev = -6.0
    beta_abs_ev = 2.4

    n_occ = min(n, (pi_electrons + 1) // 2)
    homo_idx = max(0, n_occ - 1)
    lumo_idx = min(n - 1, homo_idx + 1)

    orbitales = []
    for k in range(n):
        e_beta = float(eigvals[k])
        e_ev = round(alpha_ev + e_beta * beta_abs_ev, 3)
        if k < homo_idx:
            occ = 2
            tipo = "Ocupado (σ/π)"
        elif k == homo_idx:
            occ = 1 if (pi_electrons % 2 == 1) else 2
            tipo = "HOMO"
        elif k == lumo_idx and lumo_idx > homo_idx:
            occ = 0
            tipo = "LUMO"
        else:
            occ = 0
            tipo = "Virtual (π*)"

        orbitales.append(
            {
                "Orbital": f"Ψ{k + 1}",
                "Índice": k + 1,
                "Energía (|β|)": round(e_beta, 3),
                "Energía (eV)": e_ev,
                "Ocupación": occ,
                "Tipo": tipo,
            }
        )

    homo_ev = orbitales[homo_idx]["Energía (eV)"]
    lumo_ev = orbitales[lumo_idx]["Energía (eV)"] if lumo_idx > homo_idx else homo_ev
    gap_ev = round(abs(lumo_ev - homo_ev), 3)
    gap_beta = round(abs(orbitales[lumo_idx]["Energía (|β|)"] - orbitales[homo_idx]["Energía (|β|)"]), 3)

    # Descriptores de DFT conceptual (Teorema de Koopmans)
    i_pot = -homo_ev
    e_aff = -lumo_ev
    mu_chem = round(-(i_pot + e_aff) / 2.0, 3)
    eta_hard = round(max(0.01, (i_pot - e_aff) / 2.0), 3)
    omega_elec = round((mu_chem**2) / (2.0 * eta_hard), 3)

    # Densidad electrónica HOMO (c_i^2) por átomo del sistema pi
    homo_vec = eigvecs[:, homo_idx]
    lumo_vec = eigvecs[:, lumo_idx]
    densidad_atoms = []
    for i, a_idx in enumerate(pi_atoms):
        atom = mol.GetAtomWithIdx(a_idx)
        densidad_atoms.append(
            {
                "ID Átomo": a_idx,
                "Átomo": f"{atom.GetSymbol()}#{a_idx}",
                "c² (HOMO)": round(float(homo_vec[i] ** 2), 4),
                "c² (LUMO)": round(float(lumo_vec[i] ** 2), 4),
            }
        )

    return {
        "n_pi_atoms": n,
        "pi_electrons": pi_electrons,
        "orbitales": orbitales,
        "homo_ev": homo_ev,
        "lumo_ev": lumo_ev,
        "gap_ev": gap_ev,
        "gap_beta": gap_beta,
        "mu_chem": mu_chem,
        "eta_hard": eta_hard,
        "omega_elec": omega_elec,
        "densidad_atoms": densidad_atoms,
    }


def calcular_tcc(d_electrons: int, geometria: str, campo: str):
    """Calcula ocupación, EECC (CFSE), electrones desapareados y momento magnético."""
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
            {"Nivel": "t2g (dxy, dxz, dyz)", "Energía (Δo)": -0.4, "Electrones": t2g},
            {"Nivel": "eg (dx²-y², dz²)", "Energía (Δo)": 0.6, "Electrones": eg},
        ]
        unidad = "Δo"
    elif "Tetraédrica" in geometria:
        # Tetraédrica siempre es campo débil (Delta_t ≈ 4/9 Delta_o)
        e_orb = min(2, d_electrons) + max(0, min(2, d_electrons - 5))
        t2_orb = max(0, min(3, d_electrons - 2)) + max(0, d_electrons - 7)
        eecc = round(-0.6 * e_orb + 0.4 * t2_orb, 2)
        unpaired = (min(2, e_orb) - max(0, e_orb - 2)) + (min(3, t2_orb) - max(0, t2_orb - 3))
        niveles = [
            {"Nivel": "e (dx²-y², dz²)", "Energía (Δo)": -0.6, "Electrones": e_orb},
            {"Nivel": "t2 (dxy, dxz, dyz)", "Energía (Δo)": 0.4, "Electrones": t2_orb},
        ]
        unidad = "Δt"
    else:
        # Cuadrada plana (D4h): 4 niveles
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
        unpaired = (
            (min(2, n_xz_yz) - max(0, n_xz_yz - 2))
            + (n_z2 % 2)
            + (n_xy % 2)
            + (n_x2y2 % 2)
        )
        niveles = [
            {"Nivel": "eg (dxz, dyz)", "Energía (Δo)": -0.514, "Electrones": n_xz_yz},
            {"Nivel": "a1g (dz²)", "Energía (Δo)": -0.428, "Electrones": n_z2},
            {"Nivel": "b2g (dxy)", "Energía (Δo)": 0.228, "Electrones": n_xy},
            {"Nivel": "b1g (dx²-y²)", "Energía (Δo)": 1.228, "Electrones": n_x2y2},
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

    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(
        r"width='[^']+'\s+height='[^']+'",
        "style='width:100%; height:auto; max-height:220px;'",
        svg,
        count=1,
    )
    svg = " ".join(line.strip() for line in svg.splitlines() if line.strip())
    return svg


def calcular_tanimoto(smiles_a: str, smiles_b: str) -> float:
    m_a = Chem.MolFromSmiles(smiles_a)
    m_b = Chem.MolFromSmiles(smiles_b)
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
    for sm, (nombre, clase, formula) in catalogo.items():
        m_cat = Chem.MolFromSmiles(sm)
        if m_cat is None or Chem.MolToSmiles(m_cat) == can_obj:
            continue
        fp_cat = AllChem.GetMorganFingerprintAsBitVect(m_cat, 2, nBits=2048)
        sim = DataStructs.TanimotoSimilarity(fp_obj, fp_cat)
        resultados.append((nombre, formula, round(sim, 3)))

    resultados.sort(key=lambda x: x[2], reverse=True)
    return resultados[:top_n]


def construir_boiled_egg(tpsa_val: float, wlogp_val: float, nombre_mol: str):
    t = np.linspace(0, 2 * np.pi, 120)
    df_hia = pd.DataFrame(
        {"TPSA": 71.0 + 71.0 * np.cos(t), "WLOGP": 2.3 + 4.1 * np.sin(t), "order": np.arange(len(t)), "Zona": "HIA (Absorción GI)"}
    )
    df_bbb = pd.DataFrame(
        {"TPSA": 38.0 + 41.0 * np.cos(t), "WLOGP": 3.1 + 2.6 * np.sin(t), "order": np.arange(len(t)), "Zona": "BBB (Cruce Cerebral)"}
    )
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
    elif "ESP" in modo:
        viewer.setStyle({"stick": {"radius": 0.14, "colorscheme": "greenCarbon"}})
        viewer.zoomTo()
    else:
        viewer.setStyle({"stick": {"radius": 0.15, "color": "#556B2F"}})
        viewer.addStyle(
            {"elem": ["N", "O", "S", "F"]},
            {"sphere": {"scale": 0.42, "color": "#84CC16"}},
        )
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


# 5. HERO SECTION
st.markdown(
    """
<div class="hero-box">
    <div class="badge-pill">🌿 Green Lab & Quantum Core v7.0</div>
    <div class="brand-title">PharmaChem AI Studio</div>
    <div class="hero-desc">
        Entorno quimioinformático interactivo con PubChem, editor molecular, comparador A vs B,
        química cuántica (Schrödinger / TOM / TCC), mapa BOILED-Egg y potencial electrostático (ESP).
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

modo_comparador = st.sidebar.toggle("⚖️ Modo Comparador (A vs B)", value=False)

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
        else ("🌐 Buscar en PubChem..." if x == "PUBCHEM" else "✍️ Dibujar / SMILES personalizado...")
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
    with st.expander("✏️ Editor Molecular Interactivo (Ketcher / SMILES)", expanded=True):
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
            st.info("💡 Instala `streamlit-ketcher` para dibujar moléculas visualmente con el ratón.")

smiles_b = None
if modo_comparador:
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sec-header" style="font-size:1.1rem;">⚖️ Espécimen B</div>', unsafe_allow_html=True)
    keys_b = list(base_moleculas.keys())
    default_b_idx = 3 if len(keys_b) > 3 else 0
    sel_b = st.sidebar.selectbox(
        "Comparar contra:",
        options=keys_b + ["CUSTOM_B"],
        index=default_b_idx,
        format_func=lambda x: f"{base_moleculas[x][0]} ({base_moleculas[x][2]})"
        if x in base_moleculas
        else "✍️ SMILES personalizado B...",
    )
    smiles_b = st.sidebar.text_input("SMILES B:", "CC(=O)O").strip() if sel_b == "CUSTOM_B" else sel_b


# 8. VISTA MODO COMPARADOR (A vs B)
if modo_comparador and smiles_b:
    res_a = analizar_molecula(smiles_activo)
    res_b = analizar_molecula(smiles_b)

    if res_a is None or res_b is None:
        st.error("❌ Verifica que ambos SMILES sean válidos para comparar.")
    else:
        info_a = nombre_pubchem or base_moleculas.get(
            smiles_activo, ("Compuesto A", "Personalizado", res_a["formula"])
        )
        info_b = base_moleculas.get(smiles_b, ("Compuesto B", "Personalizado", res_b["formula"]))
        tanimoto = calcular_tanimoto(smiles_activo, smiles_b)

        st.markdown(
            f"""
        <div class="specimen-bar">
            <div>
                <div style="font-size: 0.78rem; color: #84CC16; text-transform: uppercase; letter-spacing: 0.08em;">Comparación Estructural A vs B</div>
                <div style="font-family: 'WickyBejari', 'Special Elite', serif; font-size: 1.55rem; color: #E6F4EA;">
                    {info_a[0]} ({info_a[2]}) <span style="color:#84CC16;">vs</span> {info_b[0]} ({info_b[2]})
                </div>
            </div>
            <div class="smiles-tag">Similitud Tanimoto (Morgan): {tanimoto:.3f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        spin_comp = st.toggle("🔄 Rotación 3D simultánea", value=False)
        ca, cb = st.columns(2, gap="large")

        with ca:
            st.markdown(f'<div class="sec-header">🧪 A: {info_a[0]}</div>', unsafe_allow_html=True)
            html_a = construir_html_3d(
                res_a["mol_block"], "Esferas", res_a["cargas_esp"], 0.35, spin=spin_comp, height=310
            )
            components.html(html_a, height=325)
            svg_a = generar_svg_2d(smiles_activo)
            if svg_a:
                st.markdown(
                    f"<div style='background:rgba(2,22,17,0.75); border:1px solid rgba(132,204,22,0.25); border-radius:12px; padding:10px;'>{svg_a}</div>",
                    unsafe_allow_html=True,
                )

        with cb:
            st.markdown(f'<div class="sec-header">🧪 B: {info_b[0]}</div>', unsafe_allow_html=True)
            html_b = construir_html_3d(
                res_b["mol_block"], "Esferas", res_b["cargas_esp"], 0.35, spin=spin_comp, height=310
            )
            components.html(html_b, height=325)
            svg_b = generar_svg_2d(smiles_b)
            if svg_b:
                st.markdown(
                    f"<div style='background:rgba(2,22,17,0.75); border:1px solid rgba(132,204,22,0.25); border-radius:12px; padding:10px;'>{svg_b}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="sec-header" style="margin-top:18px;">📊 Comparativa Fisicoquímica</div>', unsafe_allow_html=True)
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

# 9. VISTA INDIVIDUAL COMPLETA
else:
    res = analizar_molecula(smiles_activo)

    if res is None:
        st.error("❌ Error de sintaxis SMILES: Ingrese una cadena química válida.")
    else:
        info_comp = nombre_pubchem or base_moleculas.get(
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
        log_s = res["log_s"]
        sol_mg_ml = res["sol_mg_ml"]
        alertas = res["alertas"]
        cargas_esp = res["cargas_esp"]
        max_abs_q = max([abs(q) for q in cargas_esp] + [0.25])
        carga_max = max(cargas_esp) if cargas_esp else 0.0

        tom_res = calcular_huckel_tom(smiles_activo)

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

        st.markdown(
            f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Peso Molecular</div>
                <div class="kpi-val">{pm:.2f}</div>
                <div class="kpi-sub">g/mol (&lt; 500)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">LogP (WLOGP)</div>
                <div class="kpi-val">{logp:.2f}</div>
                <div class="kpi-sub">Octanol / Agua</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Solubilidad ESOL</div>
                <div class="kpi-val">{log_s:.2f}</div>
                <div class="kpi-sub">{sol_mg_ml:.2f} mg/mL</div>
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
            tab_quant, tab_qm, tab_spec, tab_adme = st.tabs(
                [
                    "⚛️ Cargas & Tabla",
                    "🌌 Cuántica (TOM / TCC)",
                    "📈 Espectros",
                    "💊 ADME & BOILED-Egg",
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

                df_atomos = pd.DataFrame(res["datos_atomos"])
                event = st.dataframe(
                    df_atomos,
                    use_container_width=True,
                    height=225,
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

            with tab_qm:
                st.markdown(
                    '<div class="sec-header" style="font-size: 1.1rem;">🌌 Ecuación de Schrödinger & TOM (Hückel π)</div>',
                    unsafe_allow_html=True,
                )
                st.latex(r"\hat{H}\Psi_i = E_i\Psi_i \qquad \det(\mathbf{H} - E\mathbf{S}) = 0 \qquad \Psi_i = \sum_{\mu} c_{i\mu}\phi_\mu")

                if tom_res is None:
                    st.info(
                        "Esta molécula es saturada (sin sistema π conjugado sp²/sp). "
                        "Prueba con Aspirina, Cafeína, Fenol o Imatinib para resolver sus orbitales moleculares π."
                    )
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("HOMO", f"{tom_res['homo_ev']} eV")
                    m2.metric("LUMO", f"{tom_res['lumo_ev']} eV")
                    m3.metric("Gap ΔE", f"{tom_res['gap_ev']} eV", f"{tom_res['gap_beta']} |β|")
                    m4.metric("Electrones π", f"{tom_res['pi_electrons']} e⁻")

                    df_orb = pd.DataFrame(tom_res["orbitales"])
                    chart_orb = (
                        alt.Chart(df_orb)
                        .mark_rule(size=3.5)
                        .encode(
                            x=alt.X("Índice:O", title="Orbital Molecular (Ψₖ)", axis=alt.Axis(labelColor="#D1FAE5", titleColor="#84CC16")),
                            y=alt.Y("Energía (eV):Q", title="Energía (eV)", axis=alt.Axis(labelColor="#D1FAE5", titleColor="#84CC16", gridColor="rgba(132,204,22,0.1)")),
                            color=alt.Color(
                                "Tipo:N",
                                scale=alt.Scale(
                                    domain=["Ocupado (σ/π)", "HOMO", "LUMO", "Virtual (π*)"],
                                    range=["#38bdf8", "#84CC16", "#f43f5e", "#94a3b8"],
                                ),
                                legend=alt.Legend(labelColor="#E6F4EA", titleColor="#84CC16"),
                            ),
                            tooltip=["Orbital", "Tipo", "Energía (eV)", "Energía (|β|)", "Ocupación"],
                        )
                        .properties(height=180, background="transparent")
                        .configure_view(stroke=None)
                    )
                    st.altair_chart(chart_orb, use_container_width=True)

                    st.caption(
                        f"**Descriptores DFT (Koopmans):** Potencial químico $\\mu = {tom_res['mu_chem']}$ eV • "
                        f"Dureza $\\eta = {tom_res['eta_hard']}$ eV • "
                        f"Electrofilicidad $\\omega = {tom_res['omega_elec']}$ eV"
                    )

                    with st.expander("🔍 Densidad Electrónica LCAO |cᵢ|² por Átomo (HOMO vs LUMO)"):
                        df_dens = pd.DataFrame(tom_res["densidad_atoms"])
                        st.dataframe(df_dens, use_container_width=True, hide_index=True, height=160)

                st.markdown("<hr style='border-color: rgba(132, 204, 22, 0.2); margin: 14px 0;'>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="sec-header" style="font-size: 1.1rem;">💎 Teoría del Campo Cristalino (TCC)</div>',
                    unsafe_allow_html=True,
                )
                tcc_c1, tcc_c2, tcc_c3 = st.columns(3)
                d_elec = tcc_c1.slider("Configuración dⁿ:", 1, 9, 6)
                geom_tcc = tcc_c2.selectbox("Geometría:", ["Octaédrica (Oh)", "Tetraédrica (Td)", "Cuadrada Plana (D4h)"])
                campo_tcc = tcc_c3.radio("Campo:", ["Alto Spin", "Bajo Spin"], horizontal=True)

                tcc_res = calcular_tcc(d_elec, geom_tcc, campo_tcc)
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("EECC (CFSE)", f"{tcc_res['eecc']} {tcc_res['unidad']}")
                tc2.metric("e⁻ Desapareados", f"{tcc_res['unpaired']}")
                tc3.metric("Momento μ_eff", f"{tcc_res['mu_b']} μB")

                df_tcc = pd.DataFrame(tcc_res["niveles"])
                chart_tcc = (
                    alt.Chart(df_tcc)
                    .mark_bar(cornerRadius=4, color="#84CC16")
                    .encode(
                        x=alt.X("Nivel:N", title="Orbitales d desdoblados", axis=alt.Axis(labelColor="#D1FAE5", labelAngle=0, titleColor="#84CC16")),
                        y=alt.Y("Energía (Δo):Q", title=f"Energía ({tcc_res['unidad']})", axis=alt.Axis(labelColor="#D1FAE5", titleColor="#84CC16", gridColor="rgba(132,204,22,0.1)")),
                        tooltip=["Nivel", "Energía (Δo)", "Electrones"],
                    )
                    .properties(height=150, background="transparent")
                    .configure_view(stroke=None)
                )
                st.altair_chart(chart_tcc, use_container_width=True)

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
                    '<div class="sec-header" style="font-size: 1.1rem;">🥚 Mapa BOILED-Egg (HIA vs BBB)</div>',
                    unsafe_allow_html=True,
                )
                st.altair_chart(construir_boiled_egg(tpsa, logp, nombre_mol), use_container_width=True)

                sol_cat = (
                    "Muy soluble"
                    if log_s > -2
                    else ("Soluble" if log_s >= -4 else ("Moderada" if log_s >= -6 else "Poco soluble"))
                )
                reglas = [
                    ("Peso Molecular < 500 g/mol (Lipinski)", pm < 500, f"{pm:.1f}"),
                    ("Lipofilicidad LogP < 5 (Lipinski)", logp < 5, f"{logp:.2f}"),
                    ("Donantes H (HBD ≤ 5) / Aceptores H (HBA ≤ 10)", hbd <= 5 and hba <= 10, f"{hbd} / {hba}"),
                    ("TPSA ≤ 140 Å² & Rotables ≤ 10 (Veber)", tpsa <= 140 and rot_bonds <= 10, f"{tpsa:.1f} Å² | {rot_bonds}"),
                    (f"Solubilidad ESOL ({sol_cat})", log_s >= -4, f"LogS {log_s:.2f}"),
                ]
                for nombre, cumple, val in reglas:
                    color = "#84CC16" if cumple else "#f59e0b"
                    st.markdown(
                        f"""
                    <div style="display:flex; justify-content:space-between; background:rgba(2,22,17,0.7); padding:8px 12px; border-radius:8px; margin-bottom:6px; border-left:4px solid {color};">
                        <span style="color:#E6F4EA; font-size:0.86rem;">{'✅' if cumple else '⚠️'} {nombre}</span>
                        <span style="font-family:'JetBrains Mono'; color:{color}; font-weight:600; font-size:0.86rem;">{val}</span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                if alertas:
                    st.warning(f"🚨 **Alertas PAINS / Brenk ({len(alertas)}):** " + ", ".join(alertas))
                else:
                    st.success("🛡️ **Filtros PAINS / Brenk:** Sin alertas estructurales detectadas.")

                similares = top_similares_catalogo(smiles_activo, base_moleculas, top_n=2)
                if similares:
                    texto_sim = " • ".join([f"**{n}** ({f}) — `{s}`" for n, f, s in similares])
                    st.caption(f"🧬 Más similares en catálogo (Tanimoto): {texto_sim}")

        with col_visor:
            st.markdown('<div class="sec-header">⚛️ Motor 3D & 2D Interactivo</div>', unsafe_allow_html=True)

            v_col1, v_col2 = st.columns([2.2, 1])
            with v_col1:
                modo_3d = st.radio(
                    "Modo de renderizado:",
                    ["🔬 Esferas", "⚡ ESP", "🔗 Farmacóforo"],
                    horizontal=True,
                    label_visibility="collapsed",
                )
            with v_col2:
                spin_3d = st.toggle("🔄 Rotar", value=False)

            try:
                html_3d_code = construir_html_3d(
                    res["mol_block"],
                    modo_3d,
                    cargas_esp,
                    max_abs_q,
                    selected_atom_id=selected_atom_id,
                    spin=spin_3d,
                    height=355,
                )
                components.html(html_3d_code, height=370)

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

# 10. PROTOCOLOS VERDES
st.markdown("<hr style='border-color: rgba(132, 204, 22, 0.2); margin: 25px 0;'>", unsafe_allow_html=True)
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
