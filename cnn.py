import numpy as np
import pandas as pd
import ast
import os

np.random.seed(42)

# ------------------- Hyperparameters ------------------------
TARGET_LEN = 300          # sequence length for sliding windows
EMBED_DIM = 20            # embedding dimension for skip-gram style
WINDOW_SIZE = 7           # sliding window size
NUM_FILTERS = 128
KERNEL_SIZES = [3,5,7,9]
POOL_SIZE = 2
HIDDEN_DIM = 128
NUM_CLASSES = 3
LR = 5e-4
EPOCHS = 50
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa:i for i, aa in enumerate(AA_LIST)}
NUM_AA = len(AA_LIST)

# ------------------- Data preprocessing --------------------
def parse_literal(s):
    try:
        return np.array(ast.literal_eval(s))
    except:
        return np.array([])

def sliding_window_features(seq_idx, window_size=WINDOW_SIZE):
    # seq_idx: array of amino acid indices
    L = len(seq_idx)
    padded = np.pad(seq_idx, (window_size//2, window_size//2), 'constant', constant_values=-1)
    windows = []
    for i in range(L):
        w = padded[i:i+window_size]
        w[w==-1] = 0  # map padding to index 0
        windows.append(w)
    return np.array(windows, dtype=int)

def load_dataset(path='cb513_ready_dataset_ml_with_features.csv'):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    X_list, y_list = [], []
    for idx, row in df.iterrows():
        seq = row.get('sequence','')
        labels = parse_literal(row.get('numeric_labels','[]'))
        if len(labels)==0:
            continue
        majority = np.argmax(np.bincount(labels))
        seq_idx = np.array([AA_TO_IDX.get(aa,0) for aa in seq])
        X_sw = sliding_window_features(seq_idx, WINDOW_SIZE)
        if X_sw.shape[0] > TARGET_LEN:
            X_sw = X_sw[:TARGET_LEN]
        else:
            pad_len = TARGET_LEN - X_sw.shape[0]
            X_sw = np.pad(X_sw, ((0,pad_len),(0,0)), 'constant', constant_values=0)
        X_list.append(X_sw)
        y_list.append(int(majority))
    X = np.array(X_list, dtype=int)
    y = np.array(y_list, dtype=int)
    return X, y

def stratified_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for c in classes:
        idx = np.where(y==c)[0]
        np.random.shuffle(idx)
        n_test = max(1,int(len(idx)*test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx = np.array(train_idx); test_idx = np.array(test_idx)
    np.random.shuffle(train_idx); np.random.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ------------------- Embedding layer -----------------------
class Embedding:
    def __init__(self, vocab_size, dim):
        self.vocab_size = vocab_size
        self.dim = dim
        self.W = np.random.randn(vocab_size, dim) * 0.1
        self.vW = np.zeros_like(self.W)
        self.dW = np.zeros_like(self.W)   # <-- initialize dW

    def forward(self, X):
        batch, seq_len, wsize = X.shape
        self.cache = X
        out = np.zeros((batch, seq_len, wsize*self.dim))
        for i in range(batch):
            for j in range(seq_len):
                emb = self.W[X[i,j]]  # (window_size, dim)
                out[i,j] = emb.flatten()
        return out

    def backward(self, d_out):
        batch, seq_len, feat_dim = d_out.shape
        wsize = feat_dim // self.dim
        dW = np.zeros_like(self.W)
        X = self.cache
        for i in range(batch):
            for j in range(seq_len):
                idx = X[i,j]
                grad = d_out[i,j].reshape(wsize, self.dim)
                for k, aa_idx in enumerate(idx):
                    dW[aa_idx] += grad[k]
        self.dW = dW / batch    # <-- assign to self.dW
        return None

    def step(self, lr, momentum=MOMENTUM):
        self.vW = momentum*self.vW - lr*self.dW
        self.W += self.vW


# ------------------- CNN & Dense layers -------------------
class Conv1D:
    def __init__(self, in_ch, out_ch, kernel_size):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = kernel_size
        scale = np.sqrt(2.0/(in_ch*kernel_size))
        self.W = np.random.randn(out_ch, in_ch, kernel_size) * scale
        self.b = np.zeros(out_ch)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)
    def forward(self, x):
        self.last_x = x.copy()
        B,L,C = x.shape
        out_len = L - self.k + 1
        out = np.zeros((B,out_len,self.out_ch))
        for i in range(out_len):
            patch = x[:,i:i+self.k,:]
            out[:,i,:] = np.tensordot(patch, self.W, axes=([1,2],[2,1])) + self.b
        return out
    def backward(self,d_out):
        x = self.last_x
        B,L,C = x.shape
        out_len = d_out.shape[1]
        dX = np.zeros_like(x); self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b)
        for i in range(out_len):
            patch = x[:,i:i+self.k,:]
            grad = d_out[:,i,:]
            self.dW += np.einsum('bo,bkc->ock', grad, patch)
            self.db += np.sum(grad,axis=0)
            for b in range(B):
                tmp = np.tensordot(grad[b], self.W, axes=([0],[0]))
                dX[b,i:i+self.k,:] += tmp.T
        self.dW/=max(1,B); self.db/=max(1,B)
        return dX
    def step(self,lr):
        self.vW = MOMENTUM*self.vW - lr*(self.dW + WEIGHT_DECAY*self.W)
        self.vb = MOMENTUM*self.vb - lr*self.db
        self.W += self.vW; self.b += self.vb

class MaxPool1D:
    def __init__(self,pool_size):
        self.pool_size = pool_size
    def forward(self,x):
        self.cache_x = x.copy()
        B,L,C = x.shape
        out_len = L//self.pool_size
        out = np.zeros((B,out_len,C))
        self.max_idx = np.zeros((out_len,B,C),dtype=int)
        for i in range(out_len):
            start = i*self.pool_size; end=start+self.pool_size
            window = x[:,start:end,:]
            out[:,i,:] = np.max(window,axis=1)
            self.max_idx[i,:,:] = start + np.argmax(window,axis=1)
        return out
    def backward(self, d_out):
        x = self.cache_x
        B, L, C = x.shape
        out_len = d_out.shape[1]
        dX = np.zeros_like(x)
        n_windows = min(out_len, self.max_idx.shape[0])
        for i in range(n_windows):
            for b in range(B):
                for c in range(C):
                    pos = int(self.max_idx[i, b, c])
                    if pos < L:   # <-- add this check
                        dX[b, pos, c] += d_out[b, i, c]
        return dX

class Dense:
    def __init__(self,in_dim,out_dim):
        scale=np.sqrt(2.0/in_dim)
        self.W = np.random.randn(in_dim,out_dim)*scale
        self.b = np.zeros(out_dim)
        self.vW = np.zeros_like(self.W); self.vb=np.zeros_like(self.b)
    def forward(self,x):
        self.cache=x; return x.dot(self.W)+self.b
    def backward(self,d_out):
        x=self.cache; B=x.shape[0]
        self.dW = x.T.dot(d_out)/B
        self.db = np.sum(d_out,axis=0)/B
        return d_out.dot(self.W.T)
    def step(self,lr):
        self.vW = MOMENTUM*self.vW - lr*(self.dW + WEIGHT_DECAY*self.W)
        self.vb = MOMENTUM*self.vb - lr*self.db
        self.W += self.vW; self.b += self.vb

# ------------------- Model ------------------------
class CNNModel:
    def __init__(self,input_len,embed_dim,window_size,num_classes):
        self.embed = Embedding(NUM_AA, embed_dim)
        self.conv_layers = [Conv1D(in_ch=embed_dim*window_size, out_ch=NUM_FILTERS, kernel_size=k) for k in KERNEL_SIZES]
        total_len = sum([(input_len - k + 1)//POOL_SIZE*NUM_FILTERS for k in KERNEL_SIZES])
        self.pool = MaxPool1D(POOL_SIZE)
        self.dense1 = Dense(total_len,HIDDEN_DIM)
        self.dense2 = Dense(HIDDEN_DIM,num_classes)
    def forward(self,X):
        batch=X.shape[0]
        x = self.embed.forward(X)
        conv_outs=[]
        self.caches=[]
        for conv in self.conv_layers:
            c = conv.forward(x); a = np.maximum(0,c)
            p = self.pool.forward(a)
            conv_outs.append(p)
            self.caches.append((c,a,p))
        flat = np.concatenate([c.reshape(batch,-1) for c in conv_outs],axis=1)
        h1 = np.maximum(0,self.dense1.forward(flat))
        logits = self.dense2.forward(h1)
        probs = softmax(logits)
        self.cache = (x, self.caches, flat, h1, probs)
        return probs
    def backward(self,probs_grad):
        x,caches,flat,h1,probs=self.cache
        d_h1=self.dense2.backward(probs_grad)
        d_h1*= (h1>0)
        dflat=self.dense1.backward(d_h1)
        offset=0
        for i,conv in enumerate(self.conv_layers):
            p=caches[i][2]
            size = p.shape[1]*p.shape[2]
            part = dflat[:,offset:offset+size]
            offset+=size
            part = part.reshape(p.shape)
            da = self.pool.backward(part)
            relu_mask = (caches[i][0]>0)
            min_t=min(da.shape[1],relu_mask.shape[1])
            dc=da[:,:min_t,:]*relu_mask[:,:min_t,:]
            conv.backward(dc)
            conv.step(LR)
        self.dense1.step(LR)
        self.dense2.step(LR)
        self.embed.step(LR)
    def predict(self,X):
        probs = self.forward(X)
        return np.argmax(probs,axis=1),probs

# ------------------- Loss / Accuracy --------------------
def softmax(x):
    ex=np.exp(x - np.max(x,axis=1,keepdims=True))
    return ex/np.sum(ex,axis=1,keepdims=True)

def cross_entropy_loss(probs,y):
    N=probs.shape[0]
    clipped = np.clip(probs[np.arange(N),y],1e-12,1.0)
    loss=-np.sum(np.log(clipped))/N
    grad=probs.copy()
    grad[np.arange(N),y]-=1
    grad/=N
    return loss,grad

def accuracy(preds,labels):
    return np.mean(preds==labels)

def batch_iter(X,y,batch_size=BATCH_SIZE):
    idx=np.arange(len(X))
    np.random.shuffle(idx)
    for i in range(0,len(X),batch_size):
        b = idx[i:i+batch_size]
        yield X[b], y[b]

# ------------------- Training -------------------------
def train(model,X_train,y_train,X_val,y_val,epochs=EPOCHS):
    best_val=0.0
    best_state=None
    for epoch in range(1,epochs+1):
        losses=[]
        for Xb,yb in batch_iter(X_train,y_train):
            probs = model.forward(Xb)
            loss,grad = cross_entropy_loss(probs,yb)
            losses.append(loss)
            model.backward(grad)
        preds_val,_ = model.predict(X_val)
        val_acc = accuracy(preds_val,y_val)
        print(f"Epoch {epoch:3d} | Train loss {np.mean(losses):.4f} | Val acc {val_acc:.4f}")
    return model

# ------------------- Main -----------------------------
def main():
    print("Loading dataset...")
    X, y = load_dataset('cb513_ready_dataset_ml_with_features.csv')
    X_train, X_test, y_train, y_test = stratified_split(X,y,test_size=0.2)
    X_tr, X_val, y_tr, y_val = stratified_split(X_train,y_train,test_size=0.1)
    print(f"Train {len(X_tr)} | Val {len(X_val)} | Test {len(X_test)}")
    model = CNNModel(input_len=TARGET_LEN, embed_dim=EMBED_DIM, window_size=WINDOW_SIZE, num_classes=NUM_CLASSES)
    model = train(model,X_tr,y_tr,X_val,y_val,epochs=EPOCHS)
    preds,_ = model.predict(X_test)
    acc = accuracy(preds,y_test)
    print("Final Test Accuracy:",acc)

if __name__=="__main__":
    main()
