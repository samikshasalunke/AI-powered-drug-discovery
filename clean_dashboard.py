import streamlit as st
st.write("✅ Streamlit app started successfully")

import pandas as pd
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# === Correct model architecture (matches training model) ===
class MorganNet(nn.Module):
    def __init__(self, input_dim=2048, output_dim=12):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# === Load the trained weights ===
#st.cache_resource
def load_model():
    model = MorganNet(input_dim=2048, output_dim=12)
    state_dict = torch.load("morgannet_tox21_weights.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# === Task names (Tox21 endpoints) ===
task_names = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

# === Load model once ===
model = load_model()

# === SMILES to Morgan fingerprint ===
def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1)

# === Streamlit UI ===
st.set_page_config(page_title="Tox21 Toxicity Predictor", layout="centered")
st.title("🧬 AI-Powered Toxicity Predictor (Tox21)")
st.markdown("Enter a **SMILES** string to predict toxicity across 12 biological targets using your trained deep learning model.")

user_input = st.text_input("💊 Enter SMILES (e.g., `CCO`):", value="CCO")

if user_input:
    fp = smiles_to_morgan(user_input)
    if fp is None:
        st.error("❌ Invalid SMILES string! Please enter a valid structure.")
    else:
        X = torch.tensor(fp, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(X).numpy().flatten()

        # Apply threshold
        threshold = 0.2
        bin_pred = (y_pred >= threshold).astype(int)

        # Create DataFrame
        results_df = pd.DataFrame({
            "Target": task_names,
            "Probability": np.round(y_pred, 3),
            "Toxic (Yes/No)": ["✅" if val else "❌" for val in bin_pred]
        })

        st.success("✅ Prediction complete!")
        st.subheader("📊 Toxicity Predictions")
        st.dataframe(results_df)

        st.subheader("📈 Toxicity Probability Chart")
        st.bar_chart(results_df.set_index("Target")["Probability"])
