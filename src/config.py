# ---------------------------------------------------------------------------
# Configuration for post-wildfire debris flow probability model
# Based on USGS Open-File Report 2016-1106
# ---------------------------------------------------------------------------

import os
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]

# --- Data ---
DATA_FILE = str(RELEASE_ROOT / "data/data.xlsx")
SHEET_NAME = "Appendix1_ModelData"

# --- Logistic regression coefficients (Equation 2) ---
INTERCEPT = -3.63
COEFF_X1  =  0.41   # Proportion high/mod burn severity & slope >= 23 deg
COEFF_X2  =  0.67   # Average dNBR / 1000
COEFF_X3  =  0.70   # Soil KF-Factor

# --- Classification threshold ---
THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Fusion model (LLM + sentence encoder + neural network)
# ---------------------------------------------------------------------------

# --- LLM provider: "claude" or "openai" ---
LLM_PROVIDER = "openai"

# --- Claude API ---
CLAUDE_MODEL      = "claude-opus-4-6"   # model used to generate descriptions
DESCRIPTIONS_FILE = "data/descriptions_gpt4o.json"  # cached LLM outputs (index -> text)

# --- OpenAI API ---
# Set this before re-running LLM generation:
#     export OPENAI_API_KEY="..."
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = "gpt-4o"              # swap to latest model as needed

# --- Sentence transformer encoder ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim; swap for all-mpnet-base-v2 (768-dim) for higher quality

# --- Neural network hyperparameters ---
NN_HIDDEN_TEXT = 64    # text branch first hidden layer
NN_HIDDEN_FUSE = 32    # text branch second hidden layer (feeds into fusion)
NN_FUSION_DIM  = 16    # fusion hidden layer
NN_DROPOUT     = 0.3
LEARNING_RATE  = 1e-3
EPOCHS         = 100
BATCH_SIZE     = 32

# --- Output ---
FUSION_MODEL_FILE = "checkpoints/fusion_model.pt"  # saved PyTorch weights

# --- Fire-name -> zone mapping ---
# Zones: OSD (Orange/San Diego), SGSBSJ (San Gabriel/San Bernardino/San
#        Jacinto), VEN (Ventura/Santa Barbara), CNM (Central NM),
#        FRCO (Front Range CO), SWCO (SW CO), WCO (Western CO/Utah),
#        MT (Montana/Idaho), NAZ (N Arizona), SAZ (S Arizona)
FIRE_TO_ZONE = {
    # OSD – Orange / San Diego
    "Harris":           "OSD",
    "Horse":            "OSD",
    "Witch":            "OSD",
    "Poomacha":         "OSD",
    "Santiago":         "OSD",
    "Canyon":           "OSD",
    # SGSBSJ – San Gabriel / San Bernardino / San Jacinto
    "Buckweed":         "SGSBSJ",
    "Day":              "SGSBSJ",
    "Gap":              "SGSBSJ",
    "Grand Prix - Old": "SGSBSJ",
    "Harvard":          "SGSBSJ",
    "Sayre":            "SGSBSJ",
    "School":           "SGSBSJ",
    "Sesnon":           "SGSBSJ",
    "Station":          "SGSBSJ",
    "Topanga":          "SGSBSJ",
    # VEN – Ventura / Santa Barbara
    "Blaisdell":        "VEN",
    "Freeway":          "VEN",
    "Jesusita":         "VEN",
    # CNM – Central New Mexico
    "Cerro Grande":     "CNM",
    "Little Bear":      "CNM",
    # FRCO – Front Range Colorado
    "Waldo Canyon":     "FRCO",
    # SWCO – Southwest Colorado
    "Missionary Ridge": "SWCO",
    # WCO – Western Colorado / Utah
    "Coal Seam":        "WCO",
    "Mollie":           "WCO",
    # MT – Montana / Idaho
    "Fridley Peak":     "MT",
    "Purdy":            "MT",
    "Sula":             "MT",
    "Bear":             "MT",
    # NAZ – Northern Arizona
    "Gladiator":        "NAZ",
    "Schultz":          "NAZ",
    "Wallow":           "NAZ",
    # SAZ – Southern Arizona
    "Horseshoe 2":      "SAZ",
    "Monument":         "SAZ",
}

# Display order for zone reports
ZONE_ORDER = ["OSD", "SGSBSJ", "VEN", "CNM", "FRCO", "MT", "NAZ", "SAZ", "SWCO", "WCO"]
