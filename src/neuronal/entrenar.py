import os
import sys
import numpy as np
import pandas as pd
import json
import base64
import io
import joblib
from xgboost import XGBRegressor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.common_functions import crear_carpeta_si_no_existe




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
        self.input_dim = input_dim
        self.model = None

        h1, h2, h3 = 32, 16, 8

        # ------- Pesos -------
        self.W1 = he_init(input_dim, h1)
        self.b1 = np.zeros((1, h1), dtype=np.float32)

        self.W2 = he_init(h1, h2)
        self.b2 = np.zeros((1, h2), dtype=np.float32)

        self.W3 = he_init(h2, h3)
        self.b3 = np.zeros((1, h3), dtype=np.float32)

        self.W4 = he_init(h3, 1)
        self.b4 = np.zeros((1, 1), dtype=np.float32)

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(self, X):
        if self.model is not None:
            # Con regresión: predict() devuelve valor continuo [-1, 1]
            pred = self.model.predict(X)
            return pred.reshape(-1, 1).astype(np.float32)

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

    # Entrenamiento con early stopping - REGRESIÓN
    # Cambio 1: Se usa XGBRegressor en lugar de Classifier
    # La salida es continua [-1, 1] representando magnitud de profit esperado
    def fit(self, X, y, epochs=20000, batch_size=32, sample_weight=None):
        y_flat = y.ravel().astype(np.float32)
        
        # Para regresión, simplemente usar el valor continuo directamente
        # No hay "class balance" en regresión, pero podemos informar estadísticas
        y_mean = float(np.mean(y_flat))
        y_std = float(np.std(y_flat))
        y_min = float(np.min(y_flat))
        y_max = float(np.max(y_flat))
        print(f"Target distribution | mean={y_mean:.4f} std={y_std:.4f} min={y_min:.4f} max={y_max:.4f}")

        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror',  # MSE para regresión continua
            n_jobs=4,
        )
        
        # Cambio 4: Pasar sample_weight si está disponible
        if sample_weight is not None:
            self.model.fit(X, y_flat, sample_weight=sample_weight)
        else:
            self.model.fit(X, y_flat)

        payload = _serialize_xgb_to_base64(self.model)
        self.W1 = np.array([f"__XGB__{payload}"], dtype=object)
        self.b1 = np.array([], dtype=np.float32)
        self.W2 = np.array([], dtype=np.float32)
        self.b2 = np.array([], dtype=np.float32)
        self.W3 = np.array([], dtype=np.float32)
        self.b3 = np.array([], dtype=np.float32)
        self.W4 = np.array([], dtype=np.float32)
        self.b4 = np.array([], dtype=np.float32)

    # ------------------------------------------------------
    # Predicción
    # ------------------------------------------------------
    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)


def _serialize_xgb_to_base64(model):
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _deserialize_xgb_from_base64(text):
    raw = base64.b64decode(text.encode('utf-8'))
    buffer = io.BytesIO(raw)
    return joblib.load(buffer)



def load_trained_model(json_file, input_dim, lr=0.01):
    if not os.path.exists(json_file):
        raise ValueError(f"Modelo no existe: {json_file}")
    if os.path.getsize(json_file) == 0:
        raise ValueError(f"Modelo vacío: {json_file}")

    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Modelo corrupto (JSON inválido): {json_file} | {e}")

    marker = None
    w1 = data.get('W1', [])
    if isinstance(w1, list) and len(w1) > 0 and isinstance(w1[0], str) and w1[0].startswith('__XGB__'):
        marker = w1[0]

    if marker is not None:
        payload = marker.replace('__XGB__', '', 1)
        model = _deserialize_xgb_from_base64(payload)
        expected_dim = int(getattr(model, 'n_features_in_', input_dim))
        nn = BinaryNN(input_dim=expected_dim, lr=lr)
        nn.model = model
        nn.input_dim = expected_dim
        nn.W1 = np.array([marker], dtype=object)
        nn.b1 = np.array([], dtype=np.float32)
        nn.W2 = np.array([], dtype=np.float32)
        nn.b2 = np.array([], dtype=np.float32)
        nn.W3 = np.array([], dtype=np.float32)
        nn.b3 = np.array([], dtype=np.float32)
        nn.W4 = np.array([], dtype=np.float32)
        nn.b4 = np.array([], dtype=np.float32)
        return nn

    stored_input_dim = int(len(data['W1'])) if isinstance(data.get('W1'), list) else int(input_dim)

    # crear red con la dimensión realmente almacenada en el modelo
    nn = BinaryNN(input_dim=stored_input_dim, lr=lr)

    nn.W1 = np.array(data['W1'], dtype=np.float32)
    nn.b1 = np.array(data['b1'], dtype=np.float32)

    nn.W2 = np.array(data['W2'], dtype=np.float32)
    nn.b2 = np.array(data['b2'], dtype=np.float32)

    nn.W3 = np.array(data['W3'], dtype=np.float32)
    nn.b3 = np.array(data['b3'], dtype=np.float32)

    nn.W4 = np.array(data['W4'], dtype=np.float32)
    nn.b4 = np.array(data['b4'], dtype=np.float32)
    nn.input_dim = stored_input_dim

    return nn


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return float(default)
        v = float(text)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _normalize_context_features(atr=0.0, adx=0.0, rsi=0.0, hour=0.0, spread=0.0, stoch=0.0):
    atr_v = max(0.0, _safe_float(atr, 0.0))
    adx_v = max(0.0, _safe_float(adx, 0.0))
    rsi_v = _safe_float(rsi, 50.0)
    hour_v = _safe_float(hour, 0.0)
    spread_v = max(0.0, _safe_float(spread, 0.0))
    stoch_v = _safe_float(stoch, 50.0)

    atr_n = np.log1p(atr_v)
    adx_n = np.clip(adx_v / 100.0, 0.0, 2.0)
    rsi_n = np.clip(rsi_v / 100.0, 0.0, 1.0)
    hour_n = np.clip(hour_v / 23.0, 0.0, 1.0)
    spread_n = np.log1p(spread_v) / np.log(101.0)
    stoch_n = np.clip(stoch_v / 100.0, 0.0, 1.0)

    return np.array([atr_n, adx_n, rsi_n, hour_n, spread_n, stoch_n], dtype=np.float32)


# ----------------------------------------------------------
# # PREDICCIÓN DESDE INPUTS
# ----------------------------------------------------------
def predict_from_inputs(nn, input1, input2, atr=0.0, adx=0.0, rsi=0.0, hour=0.0, spread=0.0,
                        atr_open=0.0, adx_open=0.0, rsi_open=0.0, stoch=0.0, stoch_open=0.0,
                        returns_1=0.0, volatility=0.0, trend=0.0, return_raw=False):

    # Asegurar que sean strings binarios de 8 bits (consistente con mapping)
    input1_raw = ''.join(ch for ch in str(input1) if ch in '01')
    input2_raw = ''.join(ch for ch in str(input2) if ch in '01')
    input1 = input1_raw[-8:].zfill(8)
    input2 = input2_raw[-8:].zfill(8)

    # Convertir cada carácter en bit
    a = np.array([int(bit) for bit in input1], dtype=np.float32)
    b = np.array([int(bit) for bit in input2], dtype=np.float32)

    base = np.concatenate((a, b))  # 16
    ctx = _normalize_context_features(
        atr=atr,
        adx=adx,
        rsi=rsi,
        hour=hour,
        spread=spread,
        stoch=stoch,
    )  # 6
    # Contexto en apertura (atr, adx, rsi, stoch al momento de abrir la operación)
    ctx_open = _normalize_context_features(
        atr=atr_open, adx=adx_open, rsi=rsi_open, stoch=stoch_open,
    )[:4]
    
    # Cambio 3: Añadir market features crudas
    # returns_1, volatility, trend normalizados
    market_features = np.array([
        float(np.clip(returns_1, -1.0, 1.0)),  # Clip returns a [-1, 1]
        float(np.clip(volatility, 0.0, 2.0)),  # Clip volatility a [0, 2]
        float(trend),  # Trend es [-1 o 1]
    ], dtype=np.float32)

    full = np.concatenate((base, ctx, ctx_open, market_features))  # 26 + 3 = 29

    expected_dim = int(getattr(nn, "input_dim", nn.W1.shape[0]))
    # Backward compatibility: si el modelo es viejo (24 o 26), ajustamos automáticamente
    if len(full) != expected_dim and expected_dim in [24, 26] and len(full) == 29:
        full = full[:expected_dim]  # Silencio: truncar features nuevas para modelos viejos
    
    if expected_dim == len(full):
        X = full.reshape(1, -1)
    elif expected_dim == len(base):
        X = base.reshape(1, -1)
    elif expected_dim > len(full):
        pad = np.zeros(expected_dim - len(full), dtype=np.float32)
        X = np.concatenate((full, pad)).reshape(1, -1)
    elif expected_dim > len(base):
        X = full[:expected_dim].reshape(1, -1)
    else:
        raise ValueError(
            "Dimensión de entrada inválida: "
            f"expected={expected_dim}, base={len(base)}, full={len(full)}. "
            f"input1_raw='{input1_raw}' (len={len(input1_raw)}), "
            f"input2_raw='{input2_raw}' (len={len(input2_raw)}), "
            f"input1='{input1}' (len={len(input1)}), "
            f"input2='{input2}' (len={len(input2)})"
        )

    pred = float(nn.forward(X)[0][0])

    # Cambio 1: Con regresión continua [-1, 1]
    # - Si pred > 0: "close" (clase=1)  
    # - Si pred <= 0: "hold" (clase=0)
    # - Usar abs(pred) como "confidence"
    clase = 1 if pred > 0.0 else 0
    prob = abs(pred)  # Confidence es la magnitud del valor continuo

    if return_raw:
        return clase, prob, pred
    return clase, prob

# ----------------------------------------------------------
# CARGA DE DATOS
# ----------------------------------------------------------
# NOTA: Este loader es BACKWARD COMPATIBLE con STOCH
# - Si CSV tiene columnas [stoch, stoch_open]: genera 26 features (16 base + 6 ctx + 4 ctx_open con STOCH)
# - Si CSV NO tiene STOCH: genera 24 features (16 base + 5 ctx + 3 ctx_open sin STOCH)
# - Modelos antiguos (24 features) seguirán funcionando, ignorando STOCH automáticamente
# - Para usar STOCH en predicción, reentrenar con CSV que incluya columnas stoch/stoch_open
def load_data(csv_file):
    if not os.path.exists(csv_file):
        raise ValueError(f"Dataset no existe: {csv_file}")

    if os.path.getsize(csv_file) == 0:
        raise ValueError(f"Dataset vacío: {csv_file}")

    # Forzar lectura como string
    try:
        df = pd.read_csv(csv_file, dtype=str)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset corrupto o sin columnas: {csv_file}")

    required_cols = ["input1", "input2", "output"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset inválido, faltan columnas {missing_cols}: {csv_file}"
        )

    if df.empty:
        raise ValueError(f"Dataset sin filas: {csv_file}")

    X_list = []
    Y_list = []

    has_context = all(col in df.columns for col in ["atr", "adx", "rsi", "hour", "spread"])
    has_stoch = "stoch" in df.columns
    has_open_context = all(col in df.columns for col in ["atr_open", "adx_open", "rsi_open"])
    has_stoch_open = "stoch_open" in df.columns
    # EXPECTED_DIM: 16 (base) + 6 (context with stoch) + 4 (ctx_open with stoch) = 26
    # Or 16 + 5 (context without stoch) + 3 (ctx_open without stoch) = 24 (backwards compatible)
    EXPECTED_DIM = 26 if has_stoch else 24

    for _, row in df.iterrows():
        input1_raw = ''.join(ch for ch in str(row["input1"]).strip() if ch in '01')
        input2_raw = ''.join(ch for ch in str(row["input2"]).strip() if ch in '01')
        input1 = input1_raw[-8:].zfill(8)
        input2 = input2_raw[-8:].zfill(8)
        y = _safe_float(row["output"], 0.0)

        # Convertir string binario a vector de bits
        a = np.array([int(bit) for bit in input1], dtype=np.float32)
        b = np.array([int(bit) for bit in input2], dtype=np.float32)
        base = np.concatenate([a, b])

        if has_context:
            stoch_val = _safe_float(row.get("stoch", 50.0), 50.0) if has_stoch else 0.0
            context = _normalize_context_features(
                atr=row["atr"],
                adx=row["adx"],
                rsi=row["rsi"],
                hour=row["hour"],
                spread=row["spread"],
                stoch=stoch_val,
            )
        else:
            context = np.zeros(6 if has_stoch else 5, dtype=np.float32)

        if has_open_context:
            stoch_open_val = _safe_float(row.get("stoch_open", 50.0), 50.0) if has_stoch_open else 0.0
            ctx_open = _normalize_context_features(
                atr=row["atr_open"],
                adx=row["adx_open"],
                rsi=row["rsi_open"],
                stoch=stoch_open_val,
            )[:4 if has_stoch_open else 3]
        else:
            ctx_open = np.zeros(4 if has_stoch_open else 3, dtype=np.float32)

        full = np.concatenate([base, context, ctx_open])

        if len(full) < EXPECTED_DIM:
            pad = np.zeros(EXPECTED_DIM - len(full), dtype=np.float32)
            full = np.concatenate([full, pad])
        elif len(full) > EXPECTED_DIM:
            full = full[:EXPECTED_DIM]

        X_list.append(full)
        Y_list.append(y)

    X = np.vstack(X_list)
    Y = np.array(Y_list).reshape(-1, 1).astype(np.float32)
    
    # Cambio 4: Calcular sample_weight basado en "executed" / "reason"
    sample_weight = None
    if "executed" in df.columns:
        # Si hay columna "executed", usar 1.0 para ejecutadas, 0.3 para no ejecutadas
        weights = []
        for _, row in df.iterrows():
            executed = int(_safe_float(row.get("executed", 0), 0)) if "executed" in df.columns else 1
            weight = 1.0 if executed else 0.3  # Penalizar decisiones no ejecutadas
            weights.append(weight)
        sample_weight = np.array(weights, dtype=np.float32)
    elif "reason" in df.columns:
        # Alternativa: basarse en "reason"
        # MODEL=1.0, SL/TP/TIMEOUT=0.8, BOOTSTRAP/otros=0.3
        weights = []
        for _, row in df.iterrows():
            reason = str(row.get("reason", "UNKNOWN")).strip().upper()
            if reason == "MODEL":
                weight = 1.0
            elif reason in ["SL", "TP", "TIMEOUT"]:
                weight = 0.8  # Menos confianza en cierres por reglas fijas
            else:
                weight = 0.3  # Bootstrap o desconocido = bajo peso
            weights.append(weight)
        sample_weight = np.array(weights, dtype=np.float32)

    return X, Y, sample_weight
  

def execute_entrenar(principal_symbol, mercados, list_algorithms = None):
    
    crear_carpeta_si_no_existe(f'output/{principal_symbol}/data_for_neuronal/model_trainer')
    list_algorithms = list_algorithms if list_algorithms else ['UP', 'DOWN']
    
    for algorithm in list_algorithms:
        for mercado in mercados:
            path = f'output/{principal_symbol}/data_for_neuronal/data/data_{mercado}_{algorithm}.csv' 
            if not os.path.exists(path):
                print(f"⚠️ Dataset no encontrado ({mercado}/{algorithm}), se omite entrenamiento inicial.")
                continue
            X, Y, sample_weight = load_data(path)
            if len(X) == 0:
                print(f"⚠️ Dataset vacío ({mercado}/{algorithm}), se omite entrenamiento inicial.")
                continue
            print("Datos cargados:")
            print("X shape:", X.shape)
            print("Y shape:", Y.shape)
            if sample_weight is not None:
                print(f"Sample weights loaded: {len(sample_weight)} samples")
            nn = BinaryNN(input_dim=X.shape[1], lr=0.01, target_loss=0.10)
            # Cambio 4: Pasar sample_weight al entrenamiento
            nn.fit(X, Y, epochs=20000, batch_size=32, sample_weight=sample_weight)
            # Guardar modelo entrenado
            model_data = {
                'W1': nn.W1.tolist(),
                'b1': nn.b1.tolist(),
                'W2': nn.W2.tolist(),
                'b2': nn.b2.tolist(),
                'W3': nn.W3.tolist(),
                'b3': nn.b3.tolist(),
                'W4': nn.W4.tolist(),
                'b4': nn.b4.tolist()
            }
            with open(f'output/{principal_symbol}/data_for_neuronal/model_trainer/model_{mercado}_{algorithm}.json', 'w') as f:
                json.dump(model_data, f, indent=4)
            print(f"Modelo entrenado guardado en 'output/{principal_symbol}/data_for_neuronal/model_trainer/model_{mercado}_{algorithm}.json'")
            
    
if __name__ == "__main__":
    execute_entrenar('AUDCAD', ['Asia'], list_algorithms = None)
    
    
    
    