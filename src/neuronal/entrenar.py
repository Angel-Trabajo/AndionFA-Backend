import numpy as np
import pandas as pd
import json

# ----------------------------------------------------------
# ACTIVACIONES
# ----------------------------------------------------------
def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ----------------------------------------------------------
# INICIALIZACIÓN HE
# ----------------------------------------------------------
def he_init(n_in, n_out):
    std = np.sqrt(2.0 / n_in)
    return np.random.randn(n_in, n_out).astype(np.float32) * std

# ----------------------------------------------------------
# RED NEURONAL PROFUNDA (3 capas ocultas)
# ----------------------------------------------------------
class BinaryNN:
    def __init__(self, input_dim, lr=0.01, target_loss=0.10):
        self.lr = lr
        self.target_loss = target_loss

        # ------- Pesos -------
        self.W1 = he_init(input_dim, 128)
        self.b1 = np.zeros((1, 128), dtype=np.float32)

        self.W2 = he_init(128, 64)
        self.b2 = np.zeros((1, 64), dtype=np.float32)

        self.W3 = he_init(64, 32)
        self.b3 = np.zeros((1, 32), dtype=np.float32)

        self.W4 = he_init(32, 1)
        self.b4 = np.zeros((1, 1), dtype=np.float32)

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = relu(self.z3)

        self.z4 = self.a3 @ self.W4 + self.b4
        self.a4 = sigmoid(self.z4)

        return self.a4

    # ------------------------------------------------------
    # Loss BCE
    # ------------------------------------------------------
    def loss(self, y_true, y_pred):
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # ------------------------------------------------------
    # Backpropagation profundo
    # ------------------------------------------------------
    def backward(self, X, y):
        m = len(y)

        # capa salida
        dz4 = (self.a4 - y) / m
        dW4 = self.a3.T @ dz4
        db4 = dz4.sum(axis=0, keepdims=True)

        # capa 3
        da3 = dz4 @ self.W4.T
        dz3 = da3 * drelu(self.z3)
        dW3 = self.a2.T @ dz3
        db3 = dz3.sum(axis=0, keepdims=True)

        # capa 2
        da2 = dz3 @ self.W3.T
        dz2 = da2 * drelu(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        # capa 1
        da1 = dz2 @ self.W2.T
        dz1 = da1 * drelu(self.z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3, dW4, db4

    # ------------------------------------------------------
    # Entrenamiento con early stopping
    # ------------------------------------------------------
    def fit(self, X, y, epochs=20000, batch_size=32):
        n = len(X)

        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            X = X[idx]
            y = y[idx]

            for i in range(0, n, batch_size):
                Xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]

                pred = self.forward(Xb)
                grads = self.backward(Xb, yb)

                dW1, db1, dW2, db2, dW3, db3, dW4, db4 = grads

                # SGD update
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2

                self.W3 -= self.lr * dW3
                self.b3 -= self.lr * db3

                self.W4 -= self.lr * dW4
                self.b4 -= self.lr * db4

            # cada 200 epochs mostramos
            if epoch % 5 == 0:
                pred_all = self.forward(X)
                loss_val = self.loss(y, pred_all)
                print(f"Epoch {epoch} | Loss {loss_val:.5f}")

            # ------------ EARLY STOPPING ------------
            pred_all = self.forward(X)
            loss_val = self.loss(y, pred_all)
            if loss_val <= self.target_loss:
                print(f"\n✔ Early stopping en epoch {epoch}, loss={loss_val:.5f}")
                break

    # ------------------------------------------------------
    # Predicción
    # ------------------------------------------------------
    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)



def load_trained_model(json_file, input_dim, lr=0.01):
    with open(json_file, "r") as f:
        data = json.load(f)

    # crear red con misma estructura
    nn = BinaryNN(input_dim=input_dim, lr=lr)

    nn.W1 = np.array(data['W1'], dtype=np.float32)
    nn.b1 = np.array(data['b1'], dtype=np.float32)

    nn.W2 = np.array(data['W2'], dtype=np.float32)
    nn.b2 = np.array(data['b2'], dtype=np.float32)

    nn.W3 = np.array(data['W3'], dtype=np.float32)
    nn.b3 = np.array(data['b3'], dtype=np.float32)

    nn.W4 = np.array(data['W4'], dtype=np.float32)
    nn.b4 = np.array(data['b4'], dtype=np.float32)

    return nn


# ----------------------------------------------------------
# # PREDICCIÓN DESDE INPUTS
# ----------------------------------------------------------
def predict_from_inputs(nn, input1, input2):
    a = np.array(input1, dtype=np.float32)
    b = np.array(input2, dtype=np.float32)

    X = np.concatenate([a, b]).reshape(1, -1)
    pred = nn.forward(X)[0][0]

    return int(pred >= 0.5), pred

# ----------------------------------------------------------
# CARGA DE DATOS
# ----------------------------------------------------------
def load_data(csv_file):
    df = pd.read_csv(csv_file)

    X_list, Y_list = [], []

    for _, row in df.iterrows():
        a = np.fromstring(row.iloc[0].strip("()"), sep=',')
        b = np.fromstring(row.iloc[1].strip("()"), sep=',')
        y = float(row.iloc[2])

        X_list.append(np.concatenate([a, b]))
        Y_list.append(y)

    X = np.vstack(X_list).astype(np.float32)
    Y = np.array(Y_list).reshape(-1,1).astype(np.float32)

    return X, Y

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    X, Y = load_data("src/neuronal/data/data_cleaned_UP_EURUSD.csv")
    print("Datos cargados:", X.shape)

    nn = BinaryNN(input_dim=X.shape[1], lr=0.01, target_loss=0.01)
    nn.fit(X, Y, epochs=20000, batch_size=32)

    preds = nn.predict(X)
    acc = (preds.reshape(-1) == Y.reshape(-1)).mean() * 100
    print(f"\nAccuracy final: {acc:.2f}%")

    # ------------ GUARDAR EL MODELO ------------
    model = {
        'W1': nn.W1.tolist(), 'b1': nn.b1.tolist(),
        'W2': nn.W2.tolist(), 'b2': nn.b2.tolist(),
        'W3': nn.W3.tolist(), 'b3': nn.b3.tolist(),
        'W4': nn.W4.tolist(), 'b4': nn.b4.tolist(),
    }

    with open("src/neuronal/data/nn_binary_best.json", "w") as f:
        json.dump(model, f, indent=2)

    print("\n✔ Modelo guardado en nn_binary_best.json")
    
    
    
    # nn = load_trained_model("src/neuronal/data/nn_binary_best.json", input_dim=18)

    # # Ejemplo de datos nuevos
    # input1 = (0, 0, 0, 0, 0, 0, 0, 0, 1)
    # input2 = (0, 0, 0, 0, 0, 0, 0, 0, 1)

    # # Predicción
    # clase, valor = predict_from_inputs(nn, input1, input2)

    # print("Predicción binaria:", clase)
    # print("Valor de salida (sigmoid):", valor)
