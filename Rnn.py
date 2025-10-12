import numpy as np
import pandas as pd
import ast
import time

# ============================================================
# 🧠 Skip-Gram Embedding Layer
# ============================================================
class SkipGramEmbedding:
    def __init__(self, vocab_size=20, embedding_dim=16):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        # Small random initialization
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1

    def encode(self, one_hot_sequence):
        """Convert one-hot residues to dense embeddings"""
        indices = np.argmax(one_hot_sequence, axis=1)
        return self.embeddings[indices]

embedding_layer = SkipGramEmbedding(vocab_size=20, embedding_dim=32)
# ============================================================
# ⚙️ Sliding Window Context Function
# ============================================================
def apply_sliding_window(features, window_size=7):
    pad = window_size // 2
    padded = np.pad(features, ((pad, pad), (0, 0)), mode='constant')
    output = []
    for i in range(pad, len(padded) - pad):
        window = padded[i - pad:i + pad + 1].flatten()
        output.append(window)
    return np.array(output)


# ============================================================
# 🔥 Improved Protein RNN Model (same as before)
# ============================================================
class ImprovedProteinRNN:
    def __init__(self, input_size=95, hidden_size=512, num_classes=3, sequence_length=150, dropout_rate=0.25):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate

        # Improved weight initialization
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.bh = np.zeros((hidden_size, 1))

        self.Why = np.random.randn(num_classes, hidden_size) * np.sqrt(2.0 / (hidden_size + num_classes))
        self.by = np.zeros((num_classes, 1))

        # Adam optimizer states
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

        self.vWxh = np.zeros_like(self.Wxh)
        self.vWhh = np.zeros_like(self.Whh)
        self.vWhy = np.zeros_like(self.Why)
        self.vbh = np.zeros_like(self.bh)
        self.vby = np.zeros_like(self.by)

        self.t = 0

    def forward(self, X, training=True):
        h = np.zeros((self.hidden_size, 1))
        self.h_states = [h.copy()]
        self.inputs = []

        for t in range(self.sequence_length):
            x_t = X[t].reshape(-1, 1)
            self.inputs.append(x_t.copy())
            h_prev = h.copy()
            h = np.tanh(np.dot(self.Wxh, x_t) + np.dot(self.Whh, h_prev) + self.bh)

            # Inverted dropout
            if training:
                dropout_mask = (np.random.rand(*h.shape) > self.dropout_rate).astype(float)
                h *= dropout_mask / (1.0 - self.dropout_rate)

            self.h_states.append(h.copy())

        y = np.dot(self.Why, np.tanh(h)) + self.by
        return self.stable_softmax(y.flatten()), h

    def stable_softmax(self, x):
        x -= np.max(x)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x) + 1e-10)

    def backward(self, X, y_true, y_pred, l2_lambda=0.0005):
        dy = y_pred.copy()
        dy[y_true] -= 1

        dWhy = np.outer(dy, self.h_states[-1].flatten()) + l2_lambda * self.Why
        dby = dy.reshape(-1, 1)
        dh_next = np.dot(self.Why.T, dy.reshape(-1, 1))

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        for t in reversed(range(self.sequence_length)):
            h_curr = self.h_states[t + 1]
            h_prev = self.h_states[t]
            x_t = self.inputs[t]
            dh = dh_next * (1 - h_curr ** 2)
            dh = np.clip(dh, -3.0, 3.0)
            dWxh += np.dot(dh, x_t.T)
            dWhh += np.dot(dh, h_prev.T)
            dbh += dh
            dh_next = np.dot(self.Whh.T, dh)
            dh_next = np.clip(dh_next, -3.0, 3.0)

        grads = [dWxh, dWhh, dWhy, dbh, dby]
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        max_norm = 5.0
        if total_norm > max_norm:
            for g in grads:
                g *= max_norm / total_norm

        return grads

    def train(self, data, labels, epochs=130, lr=0.0002, beta1=0.9, beta2=0.999, l2_lambda=0.003):
        print(f"🚀 Training IMPROVED RNN for {epochs} epochs")
        best_accuracy, patience, patience_counter = 0, 30, 0

        for epoch in range(epochs):
            start_time = time.time()
            total_loss, correct = 0.0, 0

            current_lr = lr / (1 + 0.005 * epoch)
            batch_size = 16
            indices = np.random.permutation(len(data))
            n_batches = len(data) // batch_size

            for batch in range(n_batches):
                batch_indices = indices[batch * batch_size:(batch + 1) * batch_size]
                batch_grads = [np.zeros_like(g) for g in [self.Wxh, self.Whh, self.Why, self.bh, self.by]]
                batch_loss, batch_correct = 0.0, 0

                for idx in batch_indices:
                    X_seq = data[idx]
                    y_true = labels[idx]
                    y_pred, _ = self.forward(X_seq, training=True)
                    loss = -np.log(y_pred[y_true] + 1e-10)
                    batch_loss += loss
                    if np.argmax(y_pred) == y_true:
                        batch_correct += 1
                    grads = self.backward(X_seq, y_true, y_pred, l2_lambda)
                    for i in range(len(grads)):
                        batch_grads[i] += grads[i]

                batch_size_actual = len(batch_indices)
                self.t += 1
                for i, (param, m, v) in enumerate(zip(
                    [self.Wxh, self.Whh, self.Why, self.bh, self.by],
                    [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby],
                    [self.vWxh, self.vWhh, self.vWhy, self.vbh, self.vby]
                )):
                    grad = batch_grads[i] / batch_size_actual
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad ** 2)
                    m_hat = m / (1 - beta1 ** self.t)
                    v_hat = v / (1 - beta2 ** self.t)
                    param -= current_lr * m_hat / (np.sqrt(v_hat) + 1e-8)

                total_loss += batch_loss
                correct += batch_correct

            acc = correct / len(data)
            avg_loss = total_loss / len(data)
            if acc > best_accuracy:
                best_accuracy, patience_counter = acc, 0
            else:
                patience_counter += 1

            if epoch % 8 == 0:
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | LR: {current_lr:.6f}")

            if patience_counter >= patience and epoch > 60:
                print(f"🛑 Early stopping at epoch {epoch}")
                break


# ============================================================
# 🧩 Enhanced Feature Engineering with Skip-Gram + Sliding Window
# ============================================================
def create_enhanced_features(one_hot, hydrophobicity, charge, polarity, sequence_length=150):
    global embedding_layer
    seq_len = len(one_hot) // 20
    if seq_len > sequence_length:
        start = (seq_len - sequence_length) // 2
        one_hot = one_hot[start * 20:(start + sequence_length) * 20]
        hydrophobicity = hydrophobicity[start:start + sequence_length]
        charge = charge[start:start + sequence_length]
        polarity = polarity[start:start + sequence_length]
    else:
        pad_len = sequence_length - seq_len
        one_hot = np.pad(one_hot, (0, pad_len * 20))
        hydrophobicity = np.pad(hydrophobicity, (0, pad_len))
        charge = np.pad(charge, (0, pad_len))
        polarity = np.pad(polarity, (0, pad_len))

    one_hot_reshaped = one_hot.reshape(sequence_length, 20)
    global embedding_layer  # use the global shared one
    embedded_seq = embedding_layer.encode(one_hot_reshaped)

    base_features = np.concatenate([
        embedded_seq,
        hydrophobicity.reshape(-1, 1),
        charge.reshape(-1, 1),
        polarity.reshape(-1, 1)
    ], axis=1)

    window_features = apply_sliding_window(base_features, window_size=7)

    helix_propensity = np.array([1.45, 0.97, 0.74, 1.17, 0.53, 1.07, 0.87, 1.45, 0.54, 1.13,
                                 1.64, 1.01, 1.34, 0.73, 1.14, 0.57, 1.11, 0.57, 1.30, 0.97])
    sheet_propensity = np.array([0.97, 1.30, 1.19, 0.75, 1.37, 0.87, 1.15, 0.75, 1.70, 0.75,
                                 0.90, 1.34, 1.20, 1.45, 0.75, 1.56, 0.75, 1.56, 0.75, 1.19])
    turn_propensity = np.array([0.77, 1.24, 1.23, 1.32, 1.45, 0.59, 1.04, 0.39, 1.00, 0.69,
                                0.60, 1.30, 0.79, 1.56, 1.43, 1.23, 1.10, 1.23, 0.98, 1.02])

    helix_scores = np.dot(one_hot_reshaped, helix_propensity).reshape(-1, 1)
    sheet_scores = np.dot(one_hot_reshaped, sheet_propensity).reshape(-1, 1)
    turn_scores = np.dot(one_hot_reshaped, turn_propensity).reshape(-1, 1)

    interaction_term = (hydrophobicity * polarity).reshape(-1, 1)
    combined_features = np.concatenate([window_features, helix_scores, sheet_scores, turn_scores, interaction_term], axis=1)

    return combined_features


# ============================================================
# 🌟 Normalization, Loading, and Main
# ============================================================
def enhanced_normalize_sequences(X):
    Xn = X.copy()
    for f in range(X.shape[2]):
        d = X[:, :, f]
        med, q75, q25 = np.median(d), np.percentile(d, 75), np.percentile(d, 25)
        iqr = q75 - q25
        if iqr > 0:
            Xn[:, :, f] = (d - med) / iqr
        else:
            std = np.std(d)
            if std > 0:
                Xn[:, :, f] = (d - med) / std
        Xn[:, :, f] = np.clip(Xn[:, :, f], -4, 4)
    return Xn


def load_and_preprocess_enhanced_data():
    df = pd.read_csv("cb513_ready_dataset_ml_with_features.csv")
    X, y, seq_len = [], [], 150
    for _, row in df.iterrows():
        try:
            one_hot = np.array(ast.literal_eval(row['one_hot_sequence'])).flatten()
            hydrophobicity = np.array(ast.literal_eval(row['hydrophobicity']))
            charge = np.array(ast.literal_eval(row['charge']))
            polarity = np.array(ast.literal_eval(row['polarity']))
            feats = create_enhanced_features(one_hot, hydrophobicity, charge, polarity, seq_len)
            X.append(feats)
            numeric_labels = ast.literal_eval(row['numeric_labels'])
            unique, counts = np.unique(numeric_labels, return_counts=True)
            y.append(unique[np.argmax(counts)])
        except Exception:
            continue
    X = enhanced_normalize_sequences(np.array(X, dtype=np.float32))
    return X, np.array(y)


def manual_stratified_split(X, y, test_size=0.15):
    np.random.seed(42)
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        n_test = int(len(idx) * test_size)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    print("🧬 IMPROVED PROTEIN RNN (Skip-Gram + Sliding Window)")
    X, y = load_and_preprocess_enhanced_data()
    X_train, X_test, y_train, y_test = manual_stratified_split(X, y)
    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

    model = ImprovedProteinRNN(input_size=X.shape[2], hidden_size=320, sequence_length=X.shape[1])
    model.train(X_train, y_train, epochs=130, lr=0.0015)

    print("🧪 Evaluating...")
    correct, cm = 0, np.zeros((3, 3), int)
    for i in range(len(X_test)):
        y_pred, _ = model.forward(X_test[i], training=False)
        yp, yt = np.argmax(y_pred), y_test[i]
        if yp == yt:
            correct += 1
        cm[yt, yp] += 1
    acc = correct / len(X_test)
    print(f"🎯 Final Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    main()
