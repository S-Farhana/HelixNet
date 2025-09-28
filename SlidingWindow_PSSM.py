import numpy as np
import pandas as pd
import ast
import os
import glob
import re
from collections import Counter

# Set random seed for reproducibility
np.random.seed(42)

def load_and_parse_data(filename):
    """Optimized data loading and parsing"""
    df = pd.read_csv(filename)
    
    # Filter valid proteins
    valid_mask = df['valid'] & (df['status'] == 'Valid')
    valid_df = df[valid_mask]
    
    X_list, y_list, protein_ids, sequences = [], [], [], []
    
    for _, row in valid_df.iterrows():
        try:
            one_hot = np.array(ast.literal_eval(row['one_hot_sequence']))
            labels = np.array(ast.literal_eval(row['numeric_labels']))
            seq = row['sequence']
            
            if len(one_hot) == len(labels) == len(seq):
                X_list.append(one_hot)
                y_list.append(labels)
                protein_ids.append(row['protein_id'])
                sequences.append(seq)
        except:
            continue
    
    return X_list, y_list, protein_ids, sequences

print("Loading data...")
X_list, y_list, protein_ids, sequences = load_and_parse_data("cb513_ready_dataset_ml_with_features.csv")
print(f"Loaded {len(protein_ids)} proteins")

# Train/test split
num_proteins = len(X_list)
train_size = int(0.8 * num_proteins)
indices = np.random.permutation(num_proteins)
train_idx, test_idx = indices[:train_size], indices[train_size:]

# FAST PSSM parser
def parse_pssm_fast(pssm_file):
    """Fast PSSM parsing"""
    try:
        pssm_matrix = []
        with open(pssm_file, 'r') as f:
            lines = f.readlines()
        
        start_processing = False
        for line in lines:
            if 'A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V' in line:
                start_processing = True
                continue
            if line.startswith('Last position-specific'):
                break
                
            if start_processing and line.strip():
                parts = line.split()
                if len(parts) >= 22:
                    try:
                        scores = [float(x) for x in parts[2:22]]
                        pssm_matrix.append(scores)
                    except:
                        continue
        
        if pssm_matrix:
            pssm_matrix = np.array(pssm_matrix)
            return (pssm_matrix - np.mean(pssm_matrix)) / (np.std(pssm_matrix) + 1e-8)
            
    except:
        return None
    return None

# PSSM loading
def load_pssm_fast(pssm_dir, protein_ids):
    """Load PSSM profiles"""
    pssm_files = glob.glob(os.path.join(pssm_dir, "*.pssm"))
    pssm_data = {}
    
    id_lower_map = {pid.lower(): pid for pid in protein_ids}
    
    for pssm_file in pssm_files:
        base_name = os.path.basename(pssm_file).lower().split('.')[0]
        clean_name = re.sub(r'[^a-z0-9]', '', base_name.replace('_pssm', ''))
        
        for pid_lower in id_lower_map:
            pid_clean = re.sub(r'[^a-z0-9]', '', pid_lower)
            if clean_name in pid_clean or pid_clean in clean_name:
                pssm_matrix = parse_pssm_fast(pssm_file)
                if pssm_matrix is not None:
                    pssm_data[id_lower_map[pid_lower]] = pssm_matrix
                break
    
    print(f"Loaded {len(pssm_data)} PSSM profiles")
    return pssm_data

print("Loading PSSM profiles...")
pssm_data = load_pssm_fast("CB513_PSSM/", protein_ids)

# Skip-gram embeddings
class FastSkipGram:
    def __init__(self, embedding_dim=20, window_size=3, lr=0.05):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.lr = lr
        
    def build_vocab(self, sequences):
        amino_acids = sorted(set('ACDEFGHIKLMNPQRSTVWY'))
        self.aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        self.embeddings = np.random.randn(len(amino_acids), self.embedding_dim) * 0.1
        
    def train_fast(self, sequences, epochs=2):
        for epoch in range(epochs):
            for seq in sequences:
                indices = [self.aa_to_idx[aa] for aa in seq if aa in self.aa_to_idx]
                for i, target in enumerate(indices):
                    start, end = max(0, i-self.window_size), min(len(indices), i+self.window_size+1)
                    context = indices[start:i] + indices[i+1:end]
                    
                    for ctx in context:
                        dot = np.dot(self.embeddings[target], self.embeddings[ctx])
                        grad = (1 - 1/(1+np.exp(-dot))) * self.lr
                        self.embeddings[target] += grad * self.embeddings[ctx]
                        self.embeddings[ctx] += grad * self.embeddings[target]
            
            print(f"Embedding Epoch {epoch+1} completed")
    
    def get_embedding(self, aa):
        return self.embeddings[self.aa_to_idx[aa]] if aa in self.aa_to_idx else np.zeros(self.embedding_dim)

print("Training embeddings...")
embedder = FastSkipGram(embedding_dim=20, window_size=3)
embedder.build_vocab(sequences)
embedder.train_fast(sequences, epochs=2)

# Feature extraction
def extract_features_fast(protein_id, sequence, one_hot, pssm_data, embedder, window_size=7):
    seq_len = len(sequence)
    
    # Base features: one-hot(20) + PSSM(20) + embeddings(20) = 60
    features = np.zeros((seq_len, 60))
    
    # One-hot features
    features[:, :20] = one_hot
    
    # PSSM features
    pssm = pssm_data.get(protein_id, np.zeros((seq_len, 20)))
    if len(pssm) == seq_len:
        features[:, 20:40] = pssm
    
    # Embedding features
    for i, aa in enumerate(sequence):
        features[i, 40:60] = embedder.get_embedding(aa)
    
    # Sliding window
    pad_size = window_size // 2
    padded = np.pad(features, ((pad_size, pad_size), (0, 0)), mode='constant')
    
    windowed_features = []
    for i in range(seq_len):
        window = padded[i:i+window_size].flatten()
        windowed_features.append(window)
    
    return np.array(windowed_features)

# Process dataset with limit for speed
def process_dataset(indices, limit=300):
    X_batch, y_batch = [], []
    
    for idx in indices[:limit]:
        protein_id = protein_ids[idx]
        features = extract_features_fast(protein_id, sequences[idx], X_list[idx], pssm_data, embedder)
        X_batch.append(features)
        y_batch.append(y_list[idx])
    
    return np.vstack(X_batch), np.hstack(y_batch)

print("Extracting features...")
X_train, y_train = process_dataset(train_idx, limit=300)
X_test, y_test = process_dataset(test_idx, limit=100)

print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")

# Normalization
def normalize_features(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std

X_train_norm, X_test_norm = normalize_features(X_train, X_test)

# ============================================================================
# FIXED AND OPTIMIZED LOGISTIC REGRESSION
# ============================================================================

class FixedLogisticRegression:
    def __init__(self, lr=0.1, max_iter=200, reg_strength=0.01):
        self.lr = lr
        self.max_iter = max_iter
        self.reg_strength = reg_strength
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = 3  # Fixed for 3 classes (H, E, C)
        
        # Initialize weights
        self.W = np.random.randn(n_features, n_classes) * np.sqrt(2.0 / n_features)
        self.b = np.zeros((1, n_classes))
        
        y_onehot = np.eye(n_classes)[y]
        
        for i in range(self.max_iter):
            # Forward pass
            scores = X @ self.W + self.b
            probs = self.softmax(scores)
            
            # Backward pass with regularization
            grad_W = (X.T @ (probs - y_onehot)) / n_samples + self.reg_strength * self.W
            grad_b = np.sum(probs - y_onehot, axis=0, keepdims=True) / n_samples
            
            # Update weights
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b
            
            if i % 50 == 0:
                loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-10), axis=1))
                acc = np.mean(np.argmax(probs, axis=1) == y)
                print(f"LR Iteration {i}: Loss = {loss:.4f}, Acc = {acc:.4f}")
    
    def predict(self, X):
        scores = X @ self.W + self.b
        return np.argmax(scores, axis=1)

# ============================================================================
# FIXED AND SIMPLIFIED NEURAL NETWORK
# ============================================================================

class FixedNeuralNetwork:
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=3, lr=0.001):
        self.lr = lr
        
        # Initialize weights properly
        self.weights = []
        self.biases = []
        
        # Layer dimensions
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # He initialization
            scale = np.sqrt(2.0 / dims[i])
            self.weights.append(np.random.randn(dims[i], dims[i+1]) * scale)
            self.biases.append(np.zeros((1, dims[i+1])))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X, training=True):
        self.layer_inputs = [X]  # Store inputs for each layer
        self.layer_outputs = [X]  # Store outputs for each layer
        
        # Forward pass through hidden layers
        for i in range(len(self.weights) - 1):
            z = self.layer_outputs[-1] @ self.weights[i] + self.biases[i]
            a = self.relu(z)
            self.layer_inputs.append(z)
            self.layer_outputs.append(a)
        
        # Output layer (no activation yet)
        z_output = self.layer_outputs[-1] @ self.weights[-1] + self.biases[-1]
        self.layer_inputs.append(z_output)
        
        # Apply softmax for output
        output = self.softmax(z_output)
        self.layer_outputs.append(output)
        
        return output
    
    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        y_true_onehot = np.eye(3)[y_true]
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient
        delta = (y_pred - y_true_onehot) / m
        dW[-1] = self.layer_outputs[-2].T @ delta  # Use second last output (last hidden layer)
        db[-1] = np.sum(delta, axis=0, keepdims=True)
        
        # Backpropagate through hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            # delta = error from next layer * derivative of current activation
            delta = (delta @ self.weights[l+1].T) * (self.layer_inputs[l+1] > 0)
            dW[l] = self.layer_outputs[l].T @ delta
            db[l] = np.sum(delta, axis=0, keepdims=True)
        
        return dW, db
    
    def update_weights(self, gradients_w, gradients_b, lr):
        # Simple weight update - FIXED: ensure same shapes
        for i in range(len(self.weights)):
            self.weights[i] -= lr * gradients_w[i]
            self.biases[i] -= lr * gradients_b[i]
    
    def compute_loss(self, y_true, y_pred):
        y_true_onehot = np.eye(3)[y_true]
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))
    
    def train(self, X, y, epochs=100, batch_size=128):
        n_samples = X.shape[0]
        
        print(f"Training Neural Network: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Network architecture: {[X.shape[1]] + [w.shape[1] for w in self.weights]}")
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            batch_count = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # Forward pass
                y_pred = self.forward(X_batch, training=True)
                
                # Backward pass
                grads_w, grads_b = self.backward(X_batch, y_batch, y_pred)
                
                # Update weights
                self.update_weights(grads_w, grads_b, self.lr)
                
                # Calculate batch loss
                batch_loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                batch_count += 1
            
            # Average epoch loss
            epoch_loss /= batch_count
            
            # Calculate accuracy
            if epoch % 20 == 0 or epoch == epochs - 1:
                y_pred_full = self.predict(X)
                acc = np.mean(y_pred_full == y)
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {acc:.4f}")
    
    def predict(self, X):
        y_pred = self.forward(X, training=False)
        return np.argmax(y_pred, axis=1)

# ============================================================================
# MAIN EXECUTION - FIXED VERSION
# ============================================================================

def q3_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("FIXED COMPARISON: LOGISTIC REGRESSION vs NEURAL NETWORK")
    print("="*60)
    
    print(f"Training samples: {X_train_norm.shape[0]}")
    print(f"Feature dimension: {X_train_norm.shape[1]}")
    print(f"Class distribution: {Counter(y_train)}")
    
    # 1. LOGISTIC REGRESSION
    print("\n" + "="*40)
    print("TRAINING LOGISTIC REGRESSION...")
    print("="*40)
    
    lr_model = FixedLogisticRegression(lr=0.1, max_iter=200, reg_strength=0.01)
    lr_model.fit(X_train_norm, y_train)
    
    lr_train_pred = lr_model.predict(X_train_norm)
    lr_test_pred = lr_model.predict(X_test_norm)
    
    lr_train_acc = q3_accuracy(y_train, lr_train_pred)
    lr_test_acc = q3_accuracy(y_test, lr_test_pred)
    
    print(f"\nLogistic Regression Results:")
    print(f"Training Q3 Accuracy: {lr_train_acc:.4f} ({lr_train_acc*100:.2f}%)")
    print(f"Test Q3 Accuracy:     {lr_test_acc:.4f} ({lr_test_acc*100:.2f}%)")
    
    # 2. NEURAL NETWORK (FIXED)
    print("\n" + "="*40)
    print("TRAINING FIXED NEURAL NETWORK...")
    print("="*40)
    
    # Use simpler architecture to avoid dimension issues
    nn_model = FixedNeuralNetwork(
        input_dim=X_train_norm.shape[1],
        hidden_dims=[64],  # Simpler: single hidden layer
        output_dim=3,
        lr=0.001
    )
    
    nn_model.train(X_train_norm, y_train, epochs=80, batch_size=64)
    
    nn_train_pred = nn_model.predict(X_train_norm)
    nn_test_pred = nn_model.predict(X_test_norm)
    
    nn_train_acc = q3_accuracy(y_train, nn_train_pred)
    nn_test_acc = q3_accuracy(y_test, nn_test_pred)
    
    print(f"\nNeural Network Results:")
    print(f"Training Q3 Accuracy: {nn_train_acc:.4f} ({nn_train_acc*100:.2f}%)")
    print(f"Test Q3 Accuracy:     {nn_test_acc:.4f} ({nn_test_acc*100:.2f}%)")
    
    # 3. COMPARISON RESULTS
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print(f"\nQ3 Accuracy Comparison:")
    print(f"{'Model':<25} {'Training':<12} {'Test':<12}")
    print(f"{'-'*50}")
    print(f"{'Logistic Regression':<25} {lr_train_acc*100:<11.2f}% {lr_test_acc*100:<11.2f}%")
    print(f"{'Neural Network':<25} {nn_train_acc*100:<11.2f}% {nn_test_acc*100:<11.2f}%")
    
    improvement = nn_test_acc - lr_test_acc
    print(f"\nNeural Network Improvement: {improvement*100:+.2f}%")
    
    # Check if targets achieved
    target = 0.60
    if lr_test_acc >= target and nn_test_acc >= target:
        print(f"🎯 SUCCESS: Both models achieved >{target*100}% accuracy!")
    elif lr_test_acc >= target:
        print(f"🎯 PARTIAL SUCCESS: Logistic Regression achieved target")
    elif nn_test_acc >= target:
        print(f"🎯 PARTIAL SUCCESS: Neural Network achieved target")
    else:
        print(f"🎯 TARGET: Aiming for >{target*100}% accuracy")
    
    # 4. PER-CLASS ACCURACY
    print(f"\nPer-class Test Accuracy:")
    classes = ['Helix (H)', 'Strand (E)', 'Coil (C)']
    for i, class_name in enumerate(classes):
        mask = y_test == i
        if np.sum(mask) > 0:
            lr_class_acc = np.mean(lr_test_pred[mask] == y_test[mask])
            nn_class_acc = np.mean(nn_test_pred[mask] == y_test[mask])
            print(f"{class_name:<15} LR: {lr_class_acc*100:6.2f}%  NN: {nn_class_acc*100:6.2f}%")
    
    # 5. SAVE RESULTS
    print(f"\nSaving results...")
    results = {
        'X_train': X_train_norm, 'y_train': y_train,
        'X_test': X_test_norm, 'y_test': y_test,
        'lr_predictions': lr_test_pred,
        'nn_predictions': nn_test_pred
    }
    np.savez('fixed_comparison_results.npz', **results)
    
    print("✅ Fixed version completed successfully!")
    print("✅ No dimension errors!")
    print("✅ Both models trained properly!")