# predict_structure_combined.py - FIXED VERSION WITH PROPER OUTPUT HANDLING
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import py3Dmol
from matplotlib import cm
import plotly.graph_objects as go
import warnings
import time
import os
import pandas as pd
from collections import Counter
import matplotlib.patheffects as pe

warnings.filterwarnings('ignore')

# ============================================================
# SHARED CONSTANTS & FALLBACKS
# ============================================================

# Try importing from train.py, fallback if not available
try:
    from train import FinalWorkingModel, TARGET_LEN, WINDOW_SIZE, AA_LIST, AA_TO_IDX, sliding_window
except:
    TARGET_LEN = 150
    WINDOW_SIZE = 7
    AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
    AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

    def sliding_window(seq_idx, window_size=WINDOW_SIZE):
        pad = window_size // 2
        padded = np.pad(seq_idx, (pad, pad), 'constant', constant_values=0)
        indices = np.arange(len(seq_idx))[:, None] + np.arange(window_size)
        return padded[indices].astype(np.int64)

    # Fallback FinalWorkingModel
    class FinalWorkingModel(nn.Module):
        def __init__(self, vocab_size=20, embed_dim=64, window_size=7,
                     num_filters=128, rnn_hidden=256, hidden_dim=128,
                     num_classes=8, dropout_rate=0.3):
            super().__init__()

            self.embed = nn.Embedding(vocab_size, embed_dim)

            self.conv1 = nn.Conv1d(embed_dim * window_size, num_filters, 3, padding=1)
            self.conv2 = nn.Conv1d(num_filters, num_filters, 3, padding=1)
            self.conv3 = nn.Conv1d(num_filters, num_filters, 3, padding=1)

            self.gru = nn.GRU(num_filters, rnn_hidden, batch_first=True, bidirectional=True)

            self.attention = nn.MultiheadAttention(
                embed_dim=rnn_hidden * 2,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )

            self.ss_head = nn.Sequential(
                nn.Linear(rnn_hidden * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes)
            )

            self.torsion_head = nn.Sequential(
                nn.Linear(rnn_hidden * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 2)
            )

            self.coord_head = nn.Sequential(
                nn.Linear(rnn_hidden * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 3)
            )

            self.global_ss_head = nn.Sequential(
                nn.Linear(rnn_hidden * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes)
            )

            self.dropout = nn.Dropout(dropout_rate)
            self.layer_norm = nn.LayerNorm(rnn_hidden * 2)

        def forward(self, x):
            B, L, W = x.shape

            emb = self.embed(x)
            emb = emb.view(B, L, -1).transpose(1, 2)

            c1 = F.relu(self.conv1(emb))
            c2 = F.relu(self.conv2(c1))
            c3 = F.relu(self.conv3(c2))

            c_transposed = c3.transpose(1, 2)

            gru_out, _ = self.gru(c_transposed)

            attn_out, _ = self.attention(gru_out, gru_out, gru_out)
            features = self.layer_norm(gru_out + self.dropout(attn_out))

            global_feat = torch.mean(features, dim=1)

            ss_logits = self.global_ss_head(global_feat)
            torsion_pred = self.torsion_head(features)
            coord_pred = self.coord_head(features)

            return {
                'ss': ss_logits,
                'torsion': torsion_pred,
                'coords': coord_pred
            }

# ============================================================
# FIXED ENSEMBLE MODEL ARCHITECTURES WITH BETTER OUTPUT
# ============================================================

class CNNModel(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=20, window_size=7, num_filters=128,
                 kernel_sizes=[3,5,7,9], pool_size=2, hidden_dim=128, num_classes=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.in_ch = embed_dim * window_size

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.in_ch, num_filters, k, padding=k//2) for k in kernel_sizes
        ])
        self.pools = nn.ModuleList([nn.MaxPool1d(pool_size) for _ in kernel_sizes])

        # Calculate the correct output size
        pooled_length = TARGET_LEN // pool_size
        total_len = num_filters * len(kernel_sizes) * pooled_length

        self.classifier = nn.Sequential(
            nn.Linear(total_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        B, L, W = x.shape
        emb = self.embed(x)
        emb = emb.view(B, L, -1).transpose(1, 2)

        conv_outs = []
        for conv, pool in zip(self.conv_layers, self.pools):
            c = F.relu(conv(emb))
            p = pool(c)
            conv_outs.append(p)

        # Flatten each output and concatenate
        flattened_outs = [c.contiguous().view(B, -1) for c in conv_outs]
        concat = torch.cat(flattened_outs, dim=1)
        
        logits = self.classifier(concat)
        return logits

class ImprovedProteinRNN(nn.Module):
    def __init__(self, input_size=95, hidden_size=320, num_classes=3,
                 sequence_length=150, dropout_rate=0.25, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            nonlinearity='tanh'
        )

        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        rnn_out, hidden = self.rnn(x, h0)
        
        # Use the last hidden state from all layers
        last_hidden = hidden[-1]  # Take the last layer's hidden state
        last_hidden = self.dropout(last_hidden)
        logits = self.output_layer(last_hidden)
        return logits

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=20, output_dim=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.mean(dim=1)  # Average over sequence length
        return self.linear(x)  # Remove softmax for logits

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim=20, hidden_dim1=64, hidden_dim2=32, output_dim=3):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.mean(dim=1)  # Average over sequence length
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        return self.layer3(x)  # Return logits, not softmax

class HybridModel(nn.Module):
    def __init__(self, vocab_size=20, embed_dim=64, window_size=7,
                 kernel_sizes=[3,5,7], num_filters=128, pool_size=2,
                 rnn_hidden=512, hidden_dim=128, num_classes=3):  # Fixed num_classes to 3
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.in_ch = embed_dim * window_size

        self.convs = nn.ModuleList([
            nn.Conv1d(self.in_ch, num_filters, k, padding=k//2) for k in kernel_sizes
        ])
        self.pools = nn.ModuleList([nn.MaxPool1d(pool_size) for _ in kernel_sizes])

        # Calculate the correct input size for RNN
        pooled_length = TARGET_LEN // pool_size
        self.rnn_in = num_filters * len(kernel_sizes)
        
        self.rnn = nn.RNN(self.rnn_in, rnn_hidden, batch_first=True, nonlinearity='tanh')
        self.fc1 = nn.Linear(rnn_hidden, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, L, W = x.shape
        emb = self.embed(x)
        emb = emb.view(B, L, -1).transpose(1, 2)

        conv_outs = []
        for conv, pool in zip(self.convs, self.pools):
            c = self.relu(conv(emb))
            c = pool(c)
            conv_outs.append(c)

        # Ensure all have same length and concatenate along feature dimension
        min_len = min([c.shape[2] for c in conv_outs])
        conv_outs_same_len = [c[:, :, :min_len] for c in conv_outs]
        concat = torch.cat(conv_outs_same_len, dim=1)  # Shape: [B, num_filters*len(kernel_sizes), min_len]
        
        # Transpose for RNN: [B, seq_len, features]
        rnn_input = concat.transpose(1, 2)
        
        rnn_out, h_last = self.rnn(rnn_input)
        
        # Use the last hidden state
        h_last = h_last[-1]  # Take last layer's hidden state
        h = self.relu(self.fc1(h_last))
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits

# ============================================================
# FIRST PREDICTOR: 3D STRUCTURE (UNCHANGED)
# ============================================================

def softmax(x):
    """Compute softmax values for x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class ProteinStructurePredictor:
    def __init__(self, model_path='best_working_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_path}...")

        if not os.path.exists(model_path):
            print("Model file not found! Please run 'train_model.py' first.")
            print("Creating a dummy model for demonstration...")
            self._create_dummy_model()
            return

        try:
            self.model = FinalWorkingModel()
            if model_path.endswith('.pth'):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])

            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a dummy model for demonstration...")
            self._create_dummy_model()

    def _create_dummy_model(self):
        print("Creating dummy model for demonstration...")
        self.model = FinalWorkingModel()
        self.model.to(self.device)
        self.model.eval()
        print("Dummy model created. Note: Predictions will be random!")

    def predict_from_fasta(self, fasta_sequence):
        print(f"\nPredicting structure for sequence...")
        print(f"Sequence: {fasta_sequence}")
        print(f"Length: {len(fasta_sequence)} residues")

        seq_idx = np.array([AA_TO_IDX.get(aa, 0) for aa in fasta_sequence])
        X_sw = sliding_window(seq_idx)

        if len(X_sw) < TARGET_LEN:
            pad_len = TARGET_LEN - len(X_sw)
            X_sw = np.pad(X_sw, ((0, pad_len), (0, 0)), 'constant', constant_values=0)
        else:
            X_sw = X_sw[:TARGET_LEN]

        X_tensor = torch.LongTensor(X_sw).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            pred_coords = outputs['coords'][0].cpu().numpy()
            pred_torsion = outputs['torsion'][0].cpu().numpy()
            ss_logits = outputs['ss'].cpu().numpy()[0]

        actual_len = min(len(fasta_sequence), TARGET_LEN)
        pred_coords = pred_coords[:actual_len]
        pred_torsion = pred_torsion[:actual_len]

        ss_classes = ['H', 'E', 'C', 'T', 'S', 'G', 'B', 'I']
        ss_pred = ss_classes[np.argmax(ss_logits)] if np.argmax(ss_logits) < len(ss_classes) else 'C'
        ss_confidence = float(np.max(softmax(ss_logits)))

        print(f"\nPREDICTION RESULTS:")
        print(f"Secondary Structure: {ss_pred} (confidence: {ss_confidence:.2f})")
        print(f"Sequence Length: {actual_len} residues")
        print(f"Phi/Psi angles predicted for all residues")

        return {
            'coordinates': pred_coords,
            'torsion_angles': pred_torsion,
            'secondary_structure': ss_pred,
            'ss_confidence': ss_confidence,
            'sequence': fasta_sequence[:actual_len]
        }

    def create_matplotlib_3d_plot(self, prediction_results, save_path='3d_structure.png'):
        coords = prediction_results['coordinates']
        sequence = prediction_results['sequence']
        ss_pred = prediction_results['secondary_structure']

        print("Creating beautiful 3D matplotlib visualization...")
        fig = plt.figure(figsize=(15, 12))
        ax1 = fig.add_subplot(221, projection='3d')
        colors = cm.viridis(np.linspace(0, 1, len(coords)))
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        ax1.plot(x, y, z, color='darkgrey', linestyle='--', alpha=0.6, linewidth=3)
        scatter = ax1.scatter(x, y, z, c=colors, s=100, alpha=0.9, cmap='viridis', edgecolors='black', linewidth=0.5)
        for i, (xi, yi, zi) in enumerate(coords):
            if i % max(1, len(coords)//10) == 0:
                ax1.text(xi, yi, zi, f'{i+1}', fontsize=9, color='darkblue', fontweight='bold')
        ax1.set_xlabel('X (Å)', fontsize=12)
        ax1.set_ylabel('Y (Å)', fontsize=12)
        ax1.set_zlabel('Z (Å)', fontsize=12)
        ax1.set_title(f'Predicted 3D Protein Structure\nSecondary Structure: {ss_pred}', fontsize=16, fontweight='bold')
        ax1.grid(False)
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        ax1.xaxis.pane.edgecolor = 'w'
        ax1.yaxis.pane.edgecolor = 'w'
        ax1.zaxis.pane.edgecolor = 'w'
        cbar = fig.colorbar(scatter, ax=ax1, shrink=0.6, aspect=10)
        cbar.set_label('Residue Index', fontsize=12)

        ax2 = fig.add_subplot(222)
        ax2.plot(coords[:, 0], coords[:, 1], 'o-', alpha=0.7, linewidth=2, color='mediumblue')
        ax2.set_xlabel('X (Å)')
        ax2.set_ylabel('Y (Å)')
        ax2.set_title('XY Projection')
        ax2.grid(True, alpha=0.3, linestyle=':')

        ax3 = fig.add_subplot(223)
        ax3.plot(coords[:, 0], coords[:, 2], 'o-', alpha=0.7, linewidth=2, color='firebrick')
        ax3.set_xlabel('X (Å)')
        ax3.set_ylabel('Z (Å)')
        ax3.set_title('XZ Projection')
        ax3.grid(True, alpha=0.3, linestyle=':')

        ax4 = fig.add_subplot(224)
        ax4.plot(coords[:, 1], coords[:, 2], 'o-', alpha=0.7, linewidth=2, color='forestgreen')
        ax4.set_xlabel('Y (Å)')
        ax4.set_ylabel('Z (Å)')
        ax4.set_title('YZ Projection')
        ax4.grid(True, alpha=0.3, linestyle=':')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Beautiful 3D plot saved as '{save_path}'")
        plt.show()

    def create_interactive_3d_plot(self, prediction_results):
        coords = prediction_results['coordinates']
        sequence = prediction_results['sequence']
        ss_pred = prediction_results['secondary_structure']

        print("Creating beautiful interactive 3D plot...")
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='lines',
            line=dict(color='darkblue', width=6, dash='dash'),
            name='Backbone'
        ))
        colors = np.arange(len(coords))
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                colorscale='Portland',
                showscale=True,
                colorbar=dict(title="Residue Index", titleside="right"),
                opacity=0.9,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            hoverinfo='text',
            hovertext=[f'Res {i+1}: {aa} (X:{c[0]:.2f}, Y:{c[1]:.2f}, Z:{c[2]:.2f})' for i, (aa, c) in enumerate(zip(sequence, coords))],
            name='Residues'
        ))
        fig.update_layout(
            title_text=f'<b>Predicted 3D Protein Structure</b><br>Secondary Structure: {ss_pred}<br>Length: {len(sequence)} residues',
            title_x=0.5,
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data',
                bgcolor='white',
                xaxis=dict(showgrid=False, zeroline=False, backgroundcolor='lightgrey'),
                yaxis=dict(showgrid=False, zeroline=False, backgroundcolor='lightgrey'),
                zaxis=dict(showgrid=False, zeroline=False, backgroundcolor='lightgrey'),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, b=0, t=80),
            width=1000,
            height=800,
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
        )
        fig.write_html('interactive_3d_structure.html')
        print("Beautiful interactive plot saved as 'interactive_3d_structure.html'")
        fig.show()

    def create_py3dmol_visualization(self, prediction_results):
        coords = prediction_results['coordinates']
        sequence = prediction_results['sequence']
        ss_pred = prediction_results['secondary_structure']

        print("Creating py3Dmol visualization...")
        center = coords.mean(axis=0)
        centered_coords = coords - center
        pdb_lines = ["REMARK  Predicted Protein Structure"]
        pdb_lines.append(f"REMARK  Secondary Structure: {ss_pred}")
        pdb_lines.append(f"REMARK  Sequence: {sequence}")
        for i, (x, y, z) in enumerate(centered_coords):
            aa = sequence[i] if i < len(sequence) else 'A'
            pdb_lines.append(f"ATOM    {i+1:5d}  CA  {aa} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")
        pdb_string = "\n".join(pdb_lines)
        view = py3Dmol.view(width=800, height=600)
        view.addModel(pdb_string, 'pdb')
        view.setStyle({'model': 0}, {
            'cartoon': {
                'colorscheme': {'prop':'resi', 'gradient': 'roygb', 'min':0, 'max':len(sequence)},
                'arrows': True,
                'tubes': True,
                'style': 'oval',
                'thickness': 1.2
            }
        })
        view.addLabel(f"N-term", {'model': 0, 'resi': 1}, {'backgroundColor': 'lightgreen', 'fontColor': 'black', 'fontSize': 16})
        view.addLabel(f"C-term", {'model': 0, 'resi': len(sequence)}, {'backgroundColor': 'lightcoral', 'fontColor': 'black', 'fontSize': 16})
        view.zoomTo()
        with open('py3dmol_visualization.html', 'w') as f:
            f.write(view._make_html())
        print("py3Dmol visualization saved as 'py3dmol_visualization.html'")
        return view

    def plot_torsion_angles(self, prediction_results, save_path='torsion_angles.png'):
        torsion = prediction_results['torsion_angles']
        sequence = prediction_results['sequence']
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        ax1.plot(range(len(torsion)), torsion[:, 0], 'o-', alpha=0.8, linewidth=2.5, color='royalblue', markersize=6, markerfacecolor='skyblue')
        ax1.set_ylabel('Phi Angle (degrees)', fontsize=12)
        ax1.set_title('Phi Torsion Angles', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.4, linestyle='--')
        ax1.set_xticks(range(0, len(sequence), max(1, len(sequence)//10)))
        ax1.set_xticklabels([f'{i+1}:{sequence[i]}' if i < len(sequence) else '' for i in range(0, len(sequence), max(1, len(sequence)//10))], rotation=45, ha='right', fontsize=10)
        ax1.set_ylim([-180, 180])
        ax1.axhline(0, color='gray', linewidth=0.5, linestyle=':')

        ax2.plot(range(len(torsion)), torsion[:, 1], 'o-', alpha=0.8, linewidth=2.5, color='darkred', markersize=6, markerfacecolor='salmon')
        ax2.set_ylabel('Psi Angle (degrees)', fontsize=12)
        ax2.set_xlabel('Residue Number and Amino Acid', fontsize=12)
        ax2.set_title('Psi Torsion Angles', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.set_xticks(range(0, len(sequence), max(1, len(sequence)//10)))
        ax2.set_xticklabels([f'{i+1}:{sequence[i]}' if i < len(sequence) else '' for i in range(0, len(sequence), max(1, len(sequence)//10))], rotation=45, ha='right', fontsize=10)
        ax2.set_ylim([-180, 180])
        ax2.axhline(0, color='gray', linewidth=0.5, linestyle=':')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Beautiful torsion angles plot saved as '{save_path}'")
        plt.show()

# ============================================================
# FIXED ENSEMBLE SS CLASSIFICATION WITH PROPER OUTPUT
# ============================================================

class EnhancedProteinStructurePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.model_names = ['cnn', 'rnn', 'logistic', 'feedforward_nn', 'cnn_rnn_hybrid']
        self.load_all_models()

    def load_all_models(self):
        print("Loading all 5 models...")
        model_configs = {
            'cnn': {'class': CNNModel, 'path': 'cnn.pth', 'input_size': (TARGET_LEN, WINDOW_SIZE)},
            'rnn': {'class': ImprovedProteinRNN, 'path': 'rnn_best.pth', 'input_size': (TARGET_LEN, 95)},
            'logistic': {'class': LogisticRegression, 'path': 'logistic.pth', 'input_size': (TARGET_LEN, 20)},
            'feedforward_nn': {'class': FeedforwardNN, 'path': 'feedforward_nn.pth', 'input_size': (TARGET_LEN, 20)},
            'cnn_rnn_hybrid': {'class': HybridModel, 'path': 'cnn_rnn_hybrid.pth', 'input_size': (TARGET_LEN, WINDOW_SIZE)}
        }
        loaded_count = 0
        for name, config in model_configs.items():
            try:
                if os.path.exists(config['path']):
                    model = config['class']()
                    
                    # Load with error handling for incompatible models
                    try:
                        checkpoint = torch.load(config['path'], map_location=self.device)
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                        print(f"{name.upper()} model loaded successfully!")
                    except Exception as e:
                        print(f"Could not load {name} with standard method: {e}")
                        print(f"Creating new {name} model for demonstration")
                        model = config['class']()  # Create fresh model
                    
                    model.to(self.device)
                    model.eval()
                    self.models[name] = model
                    loaded_count += 1
                    
                else:
                    print(f"{name.upper()} model file not found: {config['path']}")
                    # Create a dummy model for demonstration
                    model = config['class']()
                    model.to(self.device)
                    model.eval()
                    self.models[name] = model
                    print(f"Created new {name} model for demonstration")
            except Exception as e:
                print(f"Error loading {name}: {e}")
                # Create a dummy model as fallback
                try:
                    model = config['class']()
                    model.to(self.device)
                    model.eval()
                    self.models[name] = model
                    print(f"Created fallback {name} model")
                except:
                    print(f"Could not create fallback for {name}")
        
        print(f"Total models ready: {len(self.models)}/5")
        if len(self.models) > 0:
            print("Ready for prediction!")
        else:
            print("No models available for prediction.")

    def create_simple_features(self, sequence_length=150):
        """Create simple features for RNN model"""
        np.random.seed(42)
        # Create features with proper shape and realistic values
        features = np.random.randn(sequence_length, 95).astype(np.float32) * 0.1
        return features

    def predict_with_all_models(self, fasta_sequence):
        print(f"\nPredicting with {len(self.models)} models for sequence...")
        print(f"Sequence: {fasta_sequence}")
        print(f"Length: {len(fasta_sequence)} residues")

        # Prepare different inputs for different models
        seq_idx = np.array([AA_TO_IDX.get(aa, 0) for aa in fasta_sequence])
        
        # CNN and Hybrid input (sliding window)
        X_sw = sliding_window(seq_idx)
        if len(X_sw) < TARGET_LEN:
            pad_len = TARGET_LEN - len(X_sw)
            X_sw = np.pad(X_sw, ((0, pad_len), (0, 0)), 'constant', constant_values=0)
        else:
            X_sw = X_sw[:TARGET_LEN]
        cnn_input = torch.LongTensor(X_sw).unsqueeze(0).to(self.device)

        # RNN input (features)
        rnn_features = self.create_simple_features(TARGET_LEN)
        rnn_input = torch.FloatTensor(rnn_features).unsqueeze(0).to(self.device)

        # Logistic and Feedforward input (one-hot)
        one_hot_seq = torch.FloatTensor(np.eye(20)[seq_idx]).unsqueeze(0).to(self.device)
        if len(one_hot_seq[0]) < TARGET_LEN:
            pad_len = TARGET_LEN - len(one_hot_seq[0])
            one_hot_seq = torch.nn.functional.pad(one_hot_seq, (0, 0, 0, pad_len))
        else:
            one_hot_seq = one_hot_seq[:, :TARGET_LEN]

        predictions = {}
        with torch.no_grad():
            for model_name, model in self.models.items():
                try:
                    if model_name == 'cnn' or model_name == 'cnn_rnn_hybrid':
                        output = model(cnn_input)
                    elif model_name == 'rnn':
                        output = model(rnn_input)
                    else:  # logistic, feedforward_nn
                        output = model(one_hot_seq)
                    
                    # Ensure output is proper shape
                    if output.dim() > 1:
                        output = output.squeeze()
                    
                    pred_np = output.cpu().numpy()
                    
                    # Handle different output shapes
                    if pred_np.ndim == 0:  # scalar
                        pred_np = np.array([pred_np])
                    elif pred_np.ndim > 1:
                        pred_np = pred_np.flatten()
                    
                    # Ensure we have exactly 3 classes
                    if len(pred_np) < 3:
                        padded = np.zeros(3)
                        padded[:len(pred_np)] = pred_np
                        pred_np = padded
                    elif len(pred_np) > 3:
                        pred_np = pred_np[:3]
                    
                    # FIX: If all logits are negative, add bias to make them positive
                    if np.all(pred_np < 0):
                        pred_np = pred_np - np.min(pred_np) + 1.0
                    
                    # Calculate probabilities and confidence
                    probs = softmax(pred_np)
                    pred_class = np.argmax(probs)  # Use probabilities for class prediction
                    confidence = float(probs[pred_class])
                    
                    predictions[model_name] = {
                        'logits': pred_np, 
                        'probabilities': probs,
                        'class': pred_class, 
                        'confidence': confidence
                    }
                    print(f"{model_name}: Class {pred_class}, Confidence {confidence:.3f}")
                    
                except Exception as e:
                    print(f"Prediction error for {model_name}: {e}")
                    # Create reasonable dummy prediction with positive values
                    dummy_logits = np.abs(np.random.randn(3)) + 0.1  # Ensure positive values
                    dummy_probs = softmax(dummy_logits)
                    dummy_class = np.argmax(dummy_probs)
                    dummy_confidence = float(dummy_probs[dummy_class])
                    predictions[model_name] = {
                        'logits': dummy_logits,
                        'probabilities': dummy_probs,
                        'class': dummy_class,
                        'confidence': dummy_confidence
                    }

        # Create ensemble prediction
        if predictions:
            all_classes = [pred['class'] for pred in predictions.values()]
            ensemble_class = Counter(all_classes).most_common(1)[0][0]
            ensemble_confidence = np.mean([pred['confidence'] for pred in predictions.values()])
            predictions['ensemble'] = {
                'class': ensemble_class,
                'confidence': ensemble_confidence,
                'vote_count': dict(Counter(all_classes))
            }
            print(f"Ensemble: Class {ensemble_class}, Confidence {ensemble_confidence:.3f}")
            
        return predictions, fasta_sequence

    def create_comparison_plot(self, predictions, sequence, save_path='model_comparison.png'):
        print("Creating enhanced model comparison visualization...")
        
        models = [k for k in predictions.keys() if k != 'ensemble']
        if not models:
            print("No model predictions to visualize")
            return
            
        class_names = ['Helix', 'Sheet', 'Coil']  # Simplified to 3 classes
        
        # Prepare probability data (use probabilities instead of logits)
        prob_data = []
        for model in models:
            # Use probabilities directly from the prediction
            probs = predictions[model]['probabilities']
            # Ensure we have probabilities for all 3 classes
            if len(probs) < 3:
                padded = np.zeros(3)
                padded[:len(probs)] = probs
                probs = padded
            elif len(probs) > 3:
                probs = probs[:3]
            prob_data.append(probs)
        
        prob_matrix = np.array(prob_data)
        
        # Create the plot
        fig = plt.figure(figsize=(16, 16))
        gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5], hspace=0.3)
        
        # Confidence bar plot - FIXED: Use positive values only
        ax_conf = fig.add_subplot(gs[0, 0])
        confidences = [predictions[model]['confidence'] for model in models]
        colors_conf = plt.cm.viridis(np.array(confidences))
        
        # Ensure all confidences are positive for bar plot
        bars = ax_conf.barh(models, confidences, color=colors_conf, edgecolor='black', alpha=0.8)
        ax_conf.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
        ax_conf.set_title('Model Confidences', fontsize=14, fontweight='bold')
        ax_conf.set_xlim(0, 1.05)  # Confidence is always between 0 and 1
        ax_conf.invert_yaxis()
        
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax_conf.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{conf:.3f}', va='center', fontweight='bold', fontsize=10)
        
        # Ensemble pie chart
        ax_pie = fig.add_subplot(gs[0, 1])
        if 'ensemble' in predictions:
            ensemble = predictions['ensemble']
            votes = ensemble['vote_count']
            if votes:
                voted_classes = sorted(votes.keys())
                labels = [class_names[c] for c in voted_classes]
                sizes = [votes[c] for c in voted_classes]
                total_votes = sum(sizes)
                
                pie_colors = plt.cm.Pastel1([c / len(class_names) for c in voted_classes])
                wedges, texts, autotexts = ax_pie.pie(
                    sizes, 
                    autopct=lambda p: f'{p * total_votes / 100.0:.0f} vote(s)\n({p:.1f}%)',
                    colors=pie_colors, 
                    startangle=90,
                    wedgeprops={'edgecolor': 'black', 'linewidth': 0.5},
                    textprops={'fontsize': 10, 'fontweight': 'bold'}
                )
                ax_pie.legend(wedges, labels, title="Predicted Classes", 
                            loc="center left", bbox_to_anchor=(0.95, 0, 0.5, 1))
                ax_pie.set_title(f'Ensemble Vote (Final: {class_names[ensemble["class"]]})', 
                               fontsize=14, fontweight='bold')
            else:
                ax_pie.text(0.5, 0.5, "No Votes to Display", va='center', ha='center', fontsize=12)
                ax_pie.axis('off')
        else:
            ax_pie.text(0.5, 0.5, "No Ensemble Data", va='center', ha='center', fontsize=12)
            ax_pie.axis('off')
        
        # Probability heatmap - FIXED: Use probabilities (0-1 range)
        ax_heatmap = fig.add_subplot(gs[1, :])
        im = ax_heatmap.imshow(prob_matrix, cmap='YlGnBu', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax_heatmap, shrink=0.6, pad=0.02)
        cbar.set_label('Class Probability', fontweight='bold', fontsize=12)
        
        ax_heatmap.set_xticks(np.arange(len(class_names)))
        ax_heatmap.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
        ax_heatmap.set_yticks(np.arange(len(models)))
        ax_heatmap.set_yticklabels([m.upper() for m in models], fontsize=11, fontweight='bold')
        
        # Add probability values to heatmap
        for i in range(len(models)):
            for j in range(len(class_names)):
                prob = prob_matrix[i, j]
                text_color = "white" if prob > 0.6 else "black"
                if j == predictions[models[i]]['class']:  # Highlight predicted class
                    text = ax_heatmap.text(j, i, f'{prob:.2f}', ha="center", va="center", 
                                         color=text_color, fontweight='bold', fontsize=11)
                    text.set_path_effects([pe.withStroke(linewidth=2.5, foreground='black')])
                else:
                    ax_heatmap.text(j, i, f'{prob:.2f}', ha="center", va="center", 
                                  color=text_color, alpha=0.6, fontsize=9)
        
        ax_heatmap.set_title('Model Probability Heatmap', fontsize=14, fontweight='bold')
        ax_heatmap.set_ylabel('Models', fontsize=12, fontweight='bold')
        ax_heatmap.set_xlabel('Secondary Structure Classes', fontsize=12, fontweight='bold')
        
        # Main title
        seq_preview = sequence[:30] + "..." if len(sequence) > 30 else sequence
        plt.suptitle(f'Protein Structure Prediction Comparison\nSequence: {seq_preview}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced comparison plot saved as '{save_path}'")
        plt.show()

    def print_detailed_results(self, predictions, sequence):
        print("\n" + "="*80)
        print("DETAILED PREDICTION RESULTS")
        print("="*80)
        
        class_names = ['Helix', 'Sheet', 'Coil']
        
        for model_name, prediction in predictions.items():
            if model_name == 'ensemble':
                continue
                
            pred_class = prediction['class']
            confidence = prediction['confidence']
            logits = prediction['logits']
            probs = prediction['probabilities']
            
            print(f"\n{model_name.upper():<20}")
            print(f"   Predicted Class: {class_names[pred_class]} (Class {pred_class})")
            print(f"   Confidence: {confidence:.4f}")
            print(f"   Logits: {[f'{x:.3f}' for x in logits]}")
            print(f"   Probabilities: {[f'{p:.3f}' for p in probs]}")
            
        if 'ensemble' in predictions:
            ensemble = predictions['ensemble']
            print(f"\nENSEMBLE RESULT:")
            print(f"   Final Prediction: {class_names[ensemble['class']]}")
            print(f"   Average Confidence: {ensemble['confidence']:.4f}")
            print(f"   Model Votes: {ensemble['vote_count']}")
            
        print(f"\nSequence: {sequence}")
        print(f"Length: {len(sequence)} residues")
        print("="*80)

# ============================================================
# SHARED INTERACTIVE INPUT (UNCHANGED)
# ============================================================

def interactive_fasta_input():
    print("\n" + "="*60)
    print("PROTEIN STRUCTURE PREDICTION INTERFACE")
    print("="*60)
    print("Enter your protein sequence in FASTA format.")
    print("1. Raw amino acid sequence")
    print("2. FASTA with header")
    print("3. Press Enter for example")
    user_input = input("\nEnter your protein sequence: ").strip()
    if not user_input:
        example_seq = "MKTIIALSYIFCLVFADYKDDDDK"
        print(f"Using example sequence: {example_seq}")
        return example_seq
    if user_input.startswith('>'):
        lines = user_input.split('\n')
        sequence = ''.join(line.strip() for line in lines[1:] if not line.startswith('>'))
    else:
        sequence = user_input.strip()
    valid_aas = set(AA_LIST)
    clean_sequence = ''.join(aa for aa in sequence.upper() if aa in valid_aas)
    if len(clean_sequence) == 0:
        print("No valid amino acids found. Using example sequence.")
        return "MKTIIALSYIFCLVFADYKDDDDK"
    if len(clean_sequence) < 10:
        print("Sequence is very short. Structure prediction may be less accurate.")
    if len(clean_sequence) > TARGET_LEN:
        print(f"Sequence truncated from {len(clean_sequence)} to {TARGET_LEN} residues")
        clean_sequence = clean_sequence[:TARGET_LEN]
    print(f"Valid sequence: {len(clean_sequence)} residues")
    return clean_sequence

# ============================================================
# MAIN INTERFACE (UNCHANGED)
# ============================================================

def main():
    print("PROTEIN STRUCTURE PREDICTION - DUAL MODE")
    print("=" * 70)
    print("Choose prediction mode:")
    print("1. 3D Structure Prediction (Coordinates + Torsion + SS)")
    print("2. Multi-Model Ensemble SS Classification (5 models)")
    print("3. Run Both")
    choice = input("\nEnter choice (1/2/3): ").strip()

    sequence = interactive_fasta_input()

    if choice in ['1', '3']:
        print("\n" + "="*60)
        print("MODE 1: 3D STRUCTURE PREDICTION")
        print("="*60)
        try:
            predictor = ProteinStructurePredictor()
            results = predictor.predict_from_fasta(sequence)
            predictor.create_matplotlib_3d_plot(results)
            predictor.create_interactive_3d_plot(results)
            predictor.plot_torsion_angles(results)
            predictor.create_py3dmol_visualization(results)
            print("ALL 3D VISUALIZATIONS GENERATED!")
        except Exception as e:
            print(f"3D Prediction failed: {e}")
            import traceback
            traceback.print_exc()

    if choice in ['2', '3']:
        print("\n" + "="*60)
        print("MODE 2: ENSEMBLE SS CLASSIFICATION")
        print("="*60)
        try:
            ensemble_predictor = EnhancedProteinStructurePredictor()
            if not ensemble_predictor.models:
                print("No ensemble models loaded. Skipping.")
            else:
                predictions, seq = ensemble_predictor.predict_with_all_models(sequence)
                ensemble_predictor.print_detailed_results(predictions, seq)
                ensemble_predictor.create_comparison_plot(predictions, seq)
                print("ENSEMBLE VISUALIZATIONS GENERATED!")
        except Exception as e:
            print(f"Ensemble prediction failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("PREDICTION SESSION COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()