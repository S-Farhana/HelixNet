import numpy as np
import pandas as pd
import ast
from collections import Counter

# Load real dataset
df = pd.read_csv("cb513_ready_dataset_ml_with_features.csv")

# Parse one_hot_sequence and numeric_labels, filter valid
X_list = []
y_list = []
for row in df.itertuples():
    if getattr(row, 'valid') and getattr(row, 'status') == 'Valid':
        one_hot_str = getattr(row, 'one_hot_sequence')
        labels_str = getattr(row, 'numeric_labels')
        one_hot = ast.literal_eval(one_hot_str)
        labels = ast.literal_eval(labels_str)
        X_list.append(np.array(one_hot))
        y_list.append(np.array(labels))

# Protein-level train/test split (80/20)
num_proteins = len(X_list)
train_size = int(0.8 * num_proteins)
perm = np.random.permutation(num_proteins)
train_idx, test_idx = perm[:train_size], perm[train_size:]

X_train = np.vstack([X_list[i] for i in train_idx]).astype(float)
y_train = np.hstack([y_list[i] for i in train_idx]).astype(int)
X_test = np.vstack([X_list[i] for i in test_idx]).astype(float)
y_test = np.hstack([y_list[i] for i in test_idx]).astype(int)

# Compute Q3 accuracy from scratch
def q3_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = y_true.size
    return correct / total

# Compute cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    y_one_hot = np.zeros((y_true.size, 3))
    y_one_hot[np.arange(y_true.size), y_true] = 1
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Avoid log(0)
    return -np.mean(np.sum(y_one_hot * np.log(y_pred), axis=-1))

# Majority-class baseline
def majority_class_baseline(y_train, y_test):
    majority_class = Counter(y_train).most_common(1)[0][0]
    y_pred = np.full_like(y_test, majority_class)
    return q3_accuracy(y_test, y_pred)

# Logistic Regression (from scratch)
class LogisticRegression:
    def __init__(self, input_dim, output_dim, lr=0.01):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.lr = lr
        self.losses = []
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def forward(self, X):
        z = np.dot(X, self.W) + self.b
        return self.softmax(z)
    
    def train(self, X, y, epochs=200, batch_size=1024):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            perm = np.random.permutation(n_samples)
            X_shuf = X[perm]
            y_shuf = y[perm]
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuf[i:i+batch_size]
                y_batch = y_shuf[i:i+batch_size]
                y_pred = self.forward(X_batch)
                y_one_hot = np.zeros((y_batch.size, 3))
                y_one_hot[np.arange(y_batch.size), y_batch] = 1
                grad_W = np.dot(X_batch.T, (y_pred - y_one_hot)) / X_batch.shape[0]
                grad_b = np.sum(y_pred - y_one_hot, axis=0, keepdims=True) / X_batch.shape[0]
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b
            # Compute and store loss
            y_pred_full = self.forward(X)
            loss = cross_entropy_loss(y, y_pred_full)
            self.losses.append(loss)
            if epoch % 50 == 0:
                print(f"Logistic Regression Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=-1)

# Feedforward Neural Network (2 hidden layers, from scratch)
class FeedforwardNN:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, lr=0.01):
        self.W1 = np.random.randn(input_dim, hidden_dim1) * 0.01
        self.b1 = np.zeros((1, hidden_dim1))
        self.W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.01
        self.b2 = np.zeros((1, hidden_dim2))
        self.W3 = np.random.randn(hidden_dim2, output_dim) * 0.01
        self.b3 = np.zeros((1, output_dim))
        self.lr = lr
        self.losses = []
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_deriv(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.softmax(self.z3)
    
    def train(self, X, y, epochs=200, batch_size=1024):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            perm = np.random.permutation(n_samples)
            X_shuf = X[perm]
            y_shuf = y[perm]
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuf[i:i+batch_size]
                y_batch = y_shuf[i:i+batch_size]
                y_pred = self.forward(X_batch)
                y_one_hot = np.zeros((y_batch.size, 3))
                y_one_hot[np.arange(y_batch.size), y_batch] = 1
                delta3 = y_pred - y_one_hot
                grad_W3 = np.dot(self.a2.T, delta3) / X_batch.shape[0]
                grad_b3 = np.sum(delta3, axis=0, keepdims=True) / X_batch.shape[0]
                delta2 = np.dot(delta3, self.W3.T) * self.relu_deriv(self.z2)
                grad_W2 = np.dot(self.a1.T, delta2) / X_batch.shape[0]
                grad_b2 = np.sum(delta2, axis=0, keepdims=True) / X_batch.shape[0]
                delta1 = np.dot(delta2, self.W2.T) * self.relu_deriv(self.z1)
                grad_W1 = np.dot(X_batch.T, delta1) / X_batch.shape[0]
                grad_b1 = np.sum(delta1, axis=0, keepdims=True) / X_batch.shape[0]
                self.W1 -= self.lr * grad_W1
                self.W2 -= self.lr * grad_W2
                self.W3 -= self.lr * grad_W3
                self.b1 -= self.lr * grad_b1
                self.b2 -= self.lr * grad_b2
                self.b3 -= self.lr * grad_b3
            # Compute and store loss
            y_pred_full = self.forward(X)
            loss = cross_entropy_loss(y, y_pred_full)
            self.losses.append(loss)
            if epoch % 50 == 0:
                print(f"Feedforward NN Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=-1)

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Majority-class baseline
    baseline_acc = majority_class_baseline(y_train, y_test)
    print(f"Majority Class Baseline Q3 Accuracy: {baseline_acc:.4f}")
    
    # Logistic Regression
    lr_model = LogisticRegression(input_dim=20, output_dim=3, lr=0.01)
    lr_model.train(X_train, y_train, epochs=200)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = q3_accuracy(y_test, lr_pred)
    print(f"Logistic Regression Q3 Accuracy: {lr_accuracy:.4f}")
    
    # Feedforward Neural Network (2 hidden layers)
    nn_model = FeedforwardNN(input_dim=20, hidden_dim1=64, hidden_dim2=32, output_dim=3, lr=0.01)
    nn_model.train(X_train, y_train, epochs=200)
    nn_pred = nn_model.predict(X_test)
    nn_accuracy = q3_accuracy(y_test, nn_pred)
    print(f"Feedforward NN (2 hidden layers) Q3 Accuracy: {nn_accuracy:.4f}")