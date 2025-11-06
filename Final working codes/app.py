# app.py – HelixNet: Professional Protein Structure Predictor
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import base64
import os

# --------------------------------------------------------------
# Page Config
# --------------------------------------------------------------
st.set_page_config(
    page_title="HelixNet",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------
# Background Image
# --------------------------------------------------------------
def get_base64_of_image(image_path: str) -> str | None:
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

BG_IMAGE_PATH = "background_image3.jpg"
bg_base64 = get_base64_of_image(BG_IMAGE_PATH)

# --------------------------------------------------------------
# CSS – Beautiful, Modern, Centered
# --------------------------------------------------------------
if bg_base64:
    page_bg_css = f'''
    <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.58);
            z-index: -1;
        }}

        .main-header {{
            font-size: 4.5rem !important;
            font-weight: 900 !important;
            text-align: center;
            color: white !important;
            text-shadow: 0 0 12px rgba(0,0,0,0.8), 0 0 20px rgba(255,255,255,0.4);
            margin: 1rem 0 0.5rem 0 !important;
            letter-spacing: 2px;
        }}

        .sub-header {{
            text-align: center;
            color: #00d4ff !important;
            font-size: 1.4rem;
            margin-bottom: 2rem;
            font-weight: 600;
            text-shadow: 0 0 8px rgba(0,212,255,0.6);
        }}

        .stTextArea > div > div > textarea {{
            background: rgba(255,255,255,0.95) !important;
            border: 2px solid #dadce0 !important;
            border-radius: 16px !important;
            padding: 16px !important;
            font-size: 1.1rem !important;
            backdrop-filter: blur(4px);
        }}

        .button-container {{
            display: flex;
            justify-content: center;
            gap: 16px;
            margin: 20px 0;
            flex-wrap: nowrap;
        }}

        .stButton > button {{
            background: white !important;
            color: #1a73e8 !important;
            border: 2.5px solid #dadce0 !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
            padding: 12px 36px !important;
            box-shadow: 0 3px 10px rgba(0,0,0,0.15);
            min-width: 160px;
            font-size: 1rem;
        }}

        .stButton > button:hover {{
            border-color: #1a73e8 !important;
            box-shadow: 0 6px 16px rgba(26,115,232,0.3) !important;
            transform: translateY(-2px);
        }}

        .stButton > button[kind="primary"] {{
            background: #1a73e8 !important;
            color: white !important;
            border: none !important;
        }}

        .stButton > button[kind="primary"]:hover {{
            background: #1669c1 !important;
            box-shadow: 0 8px 20px rgba(26,115,232,0.4) !important;
        }}

        .plot-box {{
            padding: 24px;
            border-radius: 20px;
            background: rgba(255,255,255,0.97);
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
            margin: 24px 0;
            border: 1px solid #e0e0e0;
        }}

        .stMetric {{
            background: rgba(255,255,255,0.97);
            padding: 16px;
            border-radius: 16px;
            border: 1px solid #dadce0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .footer {{
            text-align: center;
            margin-top: 5rem;
            color: #ccc;
            font-size: 0.95rem;
            padding: 30px;
            font-weight: 500;
        }}
    </style>
    '''
    st.markdown(page_bg_css, unsafe_allow_html=True)
else:
    st.warning("`background_image3.jpg` not found. Using default style.")

# --------------------------------------------------------------
# Title
# --------------------------------------------------------------
st.markdown('<h1 class="main-header">HelixNet</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">AI-Powered 3D Protein Structure & Secondary Structure Prediction</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------
# Import Predictors
# --------------------------------------------------------------
from combined_graphs import (
    ProteinStructurePredictor,
    EnhancedProteinStructurePredictor,
    AA_LIST,
    TARGET_LEN,
)

@st.cache_resource
def get_3d_predictor():
    return ProteinStructurePredictor()

@st.cache_resource
def get_ensemble_predictor():
    return EnhancedProteinStructurePredictor()

# --------------------------------------------------------------
# Input + Buttons
# --------------------------------------------------------------
user_txt = st.text_area(
    "",
    height=120,
    placeholder="Paste FASTA or raw sequence here...\n>Example\nMKTIIALSYIFCLVFADYKDDDDK",
    label_visibility="collapsed",
    key="seq_input"
)

st.markdown("<div class='button-container'>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    use_sample = st.button("Use Sample Sequence", use_container_width=True)
with col2:
    run = st.button("Predict Structure", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------
# Process Input
# --------------------------------------------------------------
example_seq = "MKTIIALSYIFCLVFADYKDDDDK"

if use_sample:
    user_txt = example_seq
    run = True

clean_seq = None
if run and user_txt:
    if user_txt.lstrip().startswith(">"):
        seq = "".join(l.strip() for l in user_txt.split("\n")[1:] if not l.startswith(">"))
    else:
        seq = user_txt.strip().upper()

    clean_seq = "".join(a for a in seq if a in AA_LIST)

    if not clean_seq:
        st.error("No valid amino acids found. Using example.")
        clean_seq = example_seq
    if len(clean_seq) > TARGET_LEN:
        st.warning(f"Sequence too long. Truncated to {TARGET_LEN} residues.")
        clean_seq = clean_seq[:TARGET_LEN]

    st.success(f"**Ready: {len(clean_seq)} residues**")
else:
    st.info("Enter a protein sequence and click **Predict Structure**")

# --------------------------------------------------------------
# PREDICTION
# --------------------------------------------------------------
if clean_seq:
    st.markdown("---")

    with st.spinner("Generating 3D structure..."):
        pred_3d = get_3d_predictor()
        res_3d = pred_3d.predict_from_fasta(clean_seq)

    with st.spinner("Running ensemble models..."):
        ens_pred = get_ensemble_predictor()
        raw_pred, _ = ens_pred.predict_with_all_models(clean_seq)

    UI_CLASSES = ["Helix", "Sheet", "Coil", "Turn", "Bend", "Other1", "Other2", "Other3"]

    # Normalize logits to 8 classes
    model_class_counts = {"cnn": 3, "rnn": 3, "logistic": 3, "feedforward_nn": 3, "cnn_rnn_hybrid": 3}
    for name in raw_pred:
        if name == "ensemble": continue
        logits = np.array(raw_pred[name]["logits"])
        n = model_class_counts.get(name, len(logits))
        ui_logits = np.zeros(8)
        ui_logits[:n] = logits[:n]
        raw_pred[name]["logits"] = ui_logits.tolist()
        probs = np.exp(ui_logits) / (np.sum(np.exp(ui_logits)) + 1e-12)
        raw_pred[name]["confidence"] = float(probs.max())
        raw_pred[name]["class"] = int(probs.argmax())

    # Tabs
    t3d, tEns, tCNN, tRNN, tLog, tFF, tHyb = st.tabs([
        "3D Structure", "Ensemble Vote", "CNN", "RNN", "Logistic", "Feed-Forward", "Hybrid"
    ])

    # ==================== 3D TAB WITH RMSE ====================
    with t3d:
        colA, colB = st.columns([1, 2])
        with colA:
            st.metric("Secondary Structure", res_3d["secondary_structure"])
            st.metric("Confidence", f"{res_3d['ss_confidence']:.3f}")
            st.metric("Length", f"{len(clean_seq)} residues")

            # === RMSE vs Ideal α-Helix ===
            coords = res_3d["coordinates"]
            n_res = len(coords)
            t = np.arange(n_res)
            ideal_helix = np.stack([
                t * 1.5,
                2.3 * np.sin(t * 100 * np.pi / 180),
                2.3 * np.cos(t * 100 * np.pi / 180)
            ], axis=1)

            pred_c = coords - coords.mean(axis=0)
            ideal_c = ideal_helix - ideal_helix.mean(axis=0)
            H = pred_c.T @ ideal_c
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = Vt.T @ U.T
            aligned = pred_c @ R
            rmse = np.sqrt(np.mean(np.sum((aligned - ideal_c)**2, axis=1)))

            st.metric("**RMSE vs Ideal α-Helix**", f"**{rmse:.2f} Å**")

            if rmse < 2.0:
                st.success("Excellent helical fold! (Native-like)")
            elif rmse < 4.0:
                st.warning("Good prediction – helical character present")
            else:
                st.info("Likely coil/turn-rich or unfolded")

            # PDB Download
            center = coords.mean(axis=0)
            pdb_lines = [
                f"REMARK HelixNet Prediction",
                f"REMARK Sequence: {clean_seq}",
                f"REMARK SS: {res_3d['secondary_structure']} | RMSE: {rmse:.2f} Å",
            ]
            for i, (x, y, z) in enumerate(coords - center):
                aa = clean_seq[i]
                pdb_lines.append(f"ATOM  {i+1:5d}  CA  {aa} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")
            pdb_lines.append("END")
            st.download_button(
                "Download PDB",
                data="\n".join(pdb_lines),
                file_name=f"helixnet_{clean_seq[:10]}.pdb",
                mime="chemical/x-pdb"
            )

        with colB:
            st.subheader("Interactive 3D Structure")
            fig3d = go.Figure()
            c = res_3d["coordinates"]
            fig3d.add_trace(go.Scatter3d(
                x=c[:, 0], y=c[:, 1], z=c[:, 2],
                mode="lines+markers",
                line=dict(color="#00d4ff", width=8),
                marker=dict(size=7, color=np.arange(len(c)), colorscale="Viridis", showscale=True),
                hovertemplate="<b>%{text}</b><br>X: %{x:.2f} Y: %{y:.2f} Z: %{z:.2f}",
                text=[f"{i+1}:{clean_seq[i]}" for i in range(len(c))],
            ))
            fig3d.update_layout(
                scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
                height=650, margin=dict(l=0, r=0, t=50, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig3d, use_container_width=True)

        st.subheader("Torsion Angles (φ/ψ)")
        tor = res_3d["torsion_angles"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
        ax1.plot(tor[:, 0], "o-", color="#00d4ff", markersize=6, linewidth=2)
        ax1.set_ylabel("φ (degrees)", fontsize=12); ax1.set_title("Phi Angles", fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3); ax1.set_ylim(-180, 180)
        ax2.plot(tor[:, 1], "o-", color="#ff6b6b", markersize=6, linewidth=2)
        ax2.set_ylabel("ψ (degrees)", fontsize=12); ax2.set_xlabel("Residue", fontsize=12)
        ax2.set_title("Psi Angles", fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3); ax2.set_ylim(-180, 180)
        plt.tight_layout()
        st.pyplot(fig)

    # ==================== ENSEMBLE TAB ====================
    with tEns:
        ens = raw_pred.get("ensemble", {})
        if ens:
            st.success(f"**FINAL PREDICTION: {UI_CLASSES[ens['class']]}**")
            st.metric("Ensemble Confidence", f"{ens['confidence']:.3f}")
            vote_data = [{"Class": UI_CLASSES[int(k)], "Votes": v} for k, v in ens["vote_count"].items()]
            vote_df = pd.DataFrame(vote_data)
            st.bar_chart(vote_df.set_index("Class"))

        models = [m for m in raw_pred if m != "ensemble"]
        prob_mat = np.array([np.exp(raw_pred[m]["logits"]) / (np.sum(np.exp(raw_pred[m]["logits"])) + 1e-12) for m in models])
        fig, ax = plt.subplots(figsize=(11, len(models)*0.9))
        im = ax.imshow(prob_mat, cmap="plasma", vmin=0, vmax=1)
        ax.set_xticks(range(8)); ax.set_xticklabels(UI_CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(len(models))); ax.set_yticklabels([m.upper() for m in models])
        plt.colorbar(im, ax=ax, label="Probability")
        for i in range(len(models)):
            for j in range(8):
                ax.text(j, i, f"{prob_mat[i,j]:.2f}", ha="center", va="center",
                        color="white" if prob_mat[i,j] > 0.5 else "black", fontweight="bold")
        st.pyplot(fig)

    # ==================== INDIVIDUAL MODEL TABS ====================
    def draw_model_tab(model_name, tab):
        with tab:
            if model_name not in raw_pred:
                st.warning(f"{model_name.upper()} model not loaded.")
                return
            p = raw_pred[model_name]
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Prediction", UI_CLASSES[p["class"]])
                st.metric("Confidence", f"{p['confidence']:.3f}")
            with c2:
                fig, ax = plt.subplots(figsize=(9, 5))
                colors = ["#ff6b6b" if i == p["class"] else "#00d4ff" for i in range(8)]
                ax.bar(range(8), p["logits"], color=colors, edgecolor="black", alpha=0.8)
                ax.set_xticks(range(8)); ax.set_xticklabels(UI_CLASSES, rotation=45)
                ax.set_ylabel("Logits"); ax.set_title(f"{model_name.upper()} Model")
                for i, v in enumerate(p["logits"]):
                    ax.text(i, v + max(p["logits"])*0.02, f"{v:.2f}", ha="center", fontweight="bold")
                st.pyplot(fig)

    draw_model_tab("cnn", tCNN)
    draw_model_tab("rnn", tRNN)
    draw_model_tab("logistic", tLog)
    draw_model_tab("feedforward_nn", tFF)
    draw_model_tab("cnn_rnn_hybrid", tHyb)

else:
    st.info("Enter a sequence above and click **Predict Structure** to begin.")

# --------------------------------------------------------------
# Footer
# --------------------------------------------------------------
st.markdown("---")
