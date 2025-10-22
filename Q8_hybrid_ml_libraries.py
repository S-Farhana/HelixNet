# PyTorch version of Hybrid CNN + RNN with Skip-gram embedding and sliding window
# Make sure cb513_ready_dataset_ml_with_features.csv is uploaded

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.stats import mode

# ---------------- Hyperparameters ----------------
TARGET_LEN = 150
EMBED_DIM = 64
WINDOW_SIZE = 7
NUM_FILTERS = 128
KERNEL_SIZES = [3,5,7]
POOL_SIZE = 2
RNN_HIDDEN = 512
HIDDEN_DIM = 128
NUM_CLASSES = 8

LR = 3e-4
EPOCHS = 150
BATCH_SIZE = 32

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa:i for i, aa in enumerate(AA_LIST)}
NUM_AA = len(AA_LIST)

CSV_PATH = "cb513_ready_dataset_ml_with_features.csv"

# ---------------- Dataset ----------------
def parse_literal(s):
    try:
        return np.array(eval(s))
    except:
        return np.array([])

def sliding_window(seq_idx, window_size=WINDOW_SIZE):
    pad = window_size // 2
    padded = np.pad(seq_idx, (pad,pad), 'constant')
    windows = [padded[i:i+window_size] for i in range(len(seq_idx))]
    return np.array(windows, dtype=int)

class ProteinDataset(Dataset):
    def __init__(self, csv_path=CSV_PATH, target_len=TARGET_LEN):
        df = pd.read_csv(csv_path)
        X_list, y_list = [], []
        for _, row in df.iterrows():
            seq = row.get('sequence','')
            labels = parse_literal(row.get('numeric_labels','[]'))
            if len(seq)==0 or labels.size==0:
                continue
            seq_idx = np.array([AA_TO_IDX.get(aa,0) for aa in seq])
            X_sw = sliding_window(seq_idx)
            
            # pad/truncate
            if len(labels) > target_len:
                labels = labels[:target_len]
                X_sw = X_sw[:target_len]
            else:
                pad_len = target_len - len(labels)
                labels = np.pad(labels, (0,pad_len))
                X_sw = np.pad(X_sw, ((0,pad_len),(0,0)), 'constant')
            
            X_list.append(X_sw)
            y_list.append(labels)
        
        self.X = torch.LongTensor(np.array(X_list))
        self.y = torch.LongTensor(np.array(y_list))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------- Model ----------------
class HybridModel(nn.Module):
    def __init__(self, vocab_size=NUM_AA, embed_dim=EMBED_DIM, window_size=WINDOW_SIZE,
                 kernel_sizes=KERNEL_SIZES, num_filters=NUM_FILTERS, pool_size=POOL_SIZE,
                 rnn_hidden=RNN_HIDDEN, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.in_ch = embed_dim * window_size
        
        self.convs = nn.ModuleList([
            nn.Conv1d(self.in_ch, num_filters, k) for k in kernel_sizes
        ])
        self.pools = nn.ModuleList([
            nn.MaxPool1d(pool_size) for _ in kernel_sizes
        ])
        self.rnn_hidden = rnn_hidden
        self.rnn_in = num_filters * len(kernel_sizes)
        self.rnn = nn.RNN(self.rnn_in, rnn_hidden, batch_first=True, nonlinearity='tanh')
        self.fc1 = nn.Linear(rnn_hidden, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, L, window_size)
        B,L,W = x.shape
        emb = self.embed(x)                        # (B,L,W,embed_dim)
        emb = emb.view(B,L,-1).transpose(1,2)     # (B, embed_dim*W, L) for Conv1d
        
        conv_outs = []
        for conv,pool in zip(self.convs, self.pools):
            c = self.relu(conv(emb))
            c = pool(c)
            conv_outs.append(c)
        # align time dimension: take min length
        min_len = min([c.shape[2] for c in conv_outs])
        seqs = [c[:,:,:min_len] for c in conv_outs]
        concat = torch.cat(seqs, dim=1).transpose(1,2)  # (B, min_len, rnn_in)
        
        _, h_last = self.rnn(concat)   # h_last: (1,B,H)
        h_last = h_last.squeeze(0)
        h = self.relu(self.fc1(h_last))
        logits = self.fc2(h)
        return logits

# ---------------- Training ----------------
def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0
        for Xb,yb in train_loader:
            Xb,yb = Xb.to(device), yb.to(device)
            # reduce sequence labels to mode
            yb_mode = torch.tensor(mode(yb.cpu().numpy(), axis=1)[0].flatten()).to(device)
            
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb_mode)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # validation
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for Xv,yv in val_loader:
                Xv,yv = Xv.to(device), yv.to(device)
                yv_mode = torch.tensor(mode(yv.cpu().numpy(), axis=1)[0].flatten()).to(device)
                logits = model(Xv)
                preds = torch.argmax(logits, dim=1)
                correct += (preds==yv_mode).sum().item()
                total += len(yv_mode)
        acc = correct/total
        if acc>best_acc:
            best_acc = acc
            best_model = model.state_dict()
        print(f"Epoch {epoch:02d} | Train Loss {train_loss/len(train_loader):.4f} | Val Acc {acc:.4f}")
    print("Best Val Acc:", best_acc)
    model.load_state_dict(best_model)
    return model

# ---------------- Main ----------------
def main():
    dataset = ProteinDataset()
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    test_size = int(0.2 * n_samples)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    # split train -> train/val
    n_train = len(train_dataset)
    val_size = int(0.1*n_train)
    val_idx = np.random.choice(n_train, val_size, replace=False)
    train_idx_final = np.setdiff1d(np.arange(n_train), val_idx)
    train_final = torch.utils.data.Subset(train_dataset, train_idx_final)
    val_final = torch.utils.data.Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_final, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_final, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    model = HybridModel()
    model = train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)
    
    # final test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for Xb,yb in test_loader:
            Xb,yb = Xb.to(device), yb.to(device)
            yb_mode = torch.tensor(mode(yb.cpu().numpy(), axis=1)[0].flatten()).to(device)
            logits = model(Xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds==yb_mode).sum().item()
            total += len(yb_mode)
    print("Final Test Accuracy:", correct/total)

if __name__=="__main__":
    main()
