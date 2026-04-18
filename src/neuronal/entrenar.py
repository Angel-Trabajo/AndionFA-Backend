import os
import sys
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.common_functions import crear_carpeta_si_no_existe


EXTRA_FEATURE_COLUMNS = [
    "ret_1",
    "range_1",
    "trend",
    "vol_10",
    "zscore_20",
    "momentum_ratio",
]

OPEN_NODE_BITS = 8
CLOSE_NODE_BITS = 8
HOUR_BITS = 5
MIN_HOUR_VOCAB = 24
MIN_TRAINING_SAMPLES = 30




def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_binary_id(value, width):
    raw = str(value).strip()
    padded = raw.zfill(width)
    return int(padded, 2)


def _extract_extra_features(row):
    return np.array([
        float(row[column]) if column in row else 0.0
        for column in EXTRA_FEATURE_COLUMNS
    ], dtype=np.float32)


def _binary_id_to_bit_array(value, width):
    bit_positions = np.arange(width - 1, -1, -1, dtype=np.int64)
    return ((int(value) >> bit_positions) & 1).astype(np.float32)


def _augment_extra_features(input1_id, input2_id, extra_features):
    extra = np.asarray(extra_features, dtype=np.float32).reshape(-1)
    return np.concatenate([
        extra,
        _binary_id_to_bit_array(input1_id, OPEN_NODE_BITS),
        _binary_id_to_bit_array(input2_id, CLOSE_NODE_BITS),
    ]).astype(np.float32)


def get_embedding_vocab_sizes(input1_ids, input2_ids, hour_ids):
    num_open = max(2 ** OPEN_NODE_BITS, int(np.max(input1_ids)) + 1)
    num_close = max(2 ** CLOSE_NODE_BITS, int(np.max(input2_ids)) + 1)
    num_hours = max(MIN_HOUR_VOCAB, int(np.max(hour_ids)) + 1)
    return num_open, num_close, num_hours


def validate_embedding_vocab(nn_model, input1_ids=None, input2_ids=None, hour_ids=None):
    if isinstance(nn_model, dict) and nn_model.get("model_type") == "ensemble":
        for submodel in nn_model.get("models", []):
            validate_embedding_vocab(submodel, input1_ids, input2_ids, hour_ids)
        return

    if isinstance(nn_model, LegacyBinaryNN):
        return

    if input1_ids is not None and len(input1_ids) > 0:
        max_open = int(np.max(input1_ids))
        if max_open >= nn_model.num_open:
            raise ValueError(
                f"input1_id fuera de rango: max={max_open}, num_open={nn_model.num_open}"
            )

    if input2_ids is not None and len(input2_ids) > 0:
        max_close = int(np.max(input2_ids))
        if max_close >= nn_model.num_close:
            raise ValueError(
                f"input2_id fuera de rango: max={max_close}, num_close={nn_model.num_close}"
            )

    if hour_ids is not None and len(hour_ids) > 0:
        max_hour = int(np.max(hour_ids))
        if max_hour >= nn_model.num_hours:
            raise ValueError(
                f"hour_id fuera de rango: max={max_hour}, num_hours={nn_model.num_hours}"
            )


class LegacyBinaryNN(nn.Module):
    def __init__(self, input_dim, lr=0.001, target_loss=0.10):
        super().__init__()
        self.lr = lr
        self.target_loss = target_loss
        self.input_dim = input_dim
        self.feature_mean = None
        self.feature_std = None
        self.device = _get_device()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.to(self.device)

    def forward(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        elif not torch.is_tensor(X):
            X = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device, dtype=torch.float32)
        return self.model(X)


class BinaryNN(nn.Module):
    def __init__(
        self,
        input_dim_extra,
        num_open=256,
        num_close=256,
        num_hours=32,
        emb_open_dim=4,
        emb_close_dim=4,
        emb_hour_dim=3,
        dropout_rate=0.20,
        weight_decay=1e-4,
        false_positive_cost_ratio=2.0,
        hidden_dim_1=48,
        hidden_dim_2=24,
        lr=0.001,
        target_loss=0.10,
    ):
        super().__init__()
        if emb_open_dim != emb_close_dim:
            raise ValueError("emb_open_dim y emb_close_dim deben coincidir para usar interacción e1*e2")
        self.lr = lr
        self.target_loss = target_loss
        self.input_dim_extra = input_dim_extra
        self.num_open = num_open
        self.num_close = num_close
        self.num_hours = num_hours
        self.emb_open_dim = emb_open_dim
        self.emb_close_dim = emb_close_dim
        self.emb_hour_dim = emb_hour_dim
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.false_positive_cost_ratio = false_positive_cost_ratio
        self.hidden_dim_1 = int(hidden_dim_1)
        self.hidden_dim_2 = int(hidden_dim_2)
        self.feature_mean = None
        self.feature_std = None
        self.device = _get_device()
        self.emb_open = nn.Embedding(num_open, emb_open_dim)
        self.emb_close = nn.Embedding(num_close, emb_close_dim)
        self.emb_hour = nn.Embedding(num_hours, emb_hour_dim)
        self.dropout = nn.Dropout(dropout_rate)
        total_input = emb_open_dim + emb_close_dim + emb_open_dim + emb_hour_dim + input_dim_extra
        self.logits_model = nn.Sequential(
            nn.Linear(total_input, self.hidden_dim_1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_2, 1),
        )
        self.to(self.device)

    def forward_logits(self, input1, input2, hour, extra_features):
        if not torch.is_tensor(input1):
            input1 = torch.tensor(input1, dtype=torch.long, device=self.device)
        else:
            input1 = input1.to(self.device, dtype=torch.long)

        if not torch.is_tensor(input2):
            input2 = torch.tensor(input2, dtype=torch.long, device=self.device)
        else:
            input2 = input2.to(self.device, dtype=torch.long)

        if not torch.is_tensor(hour):
            hour = torch.tensor(hour, dtype=torch.long, device=self.device)
        else:
            hour = hour.to(self.device, dtype=torch.long)

        if not torch.is_tensor(extra_features):
            extra_features = torch.tensor(extra_features, dtype=torch.float32, device=self.device)
        else:
            extra_features = extra_features.to(self.device, dtype=torch.float32)

        if input1.dim() == 0:
            input1 = input1.unsqueeze(0)
        if input2.dim() == 0:
            input2 = input2.unsqueeze(0)
        if hour.dim() == 0:
            hour = hour.unsqueeze(0)
        if extra_features.dim() == 1:
            extra_features = extra_features.unsqueeze(0)

        e1 = self.emb_open(input1)
        e2 = self.emb_close(input2)
        e3 = self.emb_hour(hour)
        interaction = e1 * e2
        X = torch.cat([e1, e2, interaction, e3, extra_features], dim=1)
        X = self.dropout(X)
        return self.logits_model(X)

    def forward(self, input1, input2, hour, extra_features):
        logits = self.forward_logits(input1, input2, hour, extra_features)
        return torch.sigmoid(logits)

    def custom_loss(self, logits, target):
        base_loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
        probs = torch.sigmoid(logits)
        false_positive_penalty = probs * (1.0 - target) * (self.false_positive_cost_ratio - 1.0)
        return (base_loss + false_positive_penalty).mean()

    def validation_objective(self, probs, target, min_precision=0.55, min_trades=20):
        preds = (probs >= 0.5).float()
        predicted_positive = int(preds.sum().item())
        if predicted_positive == 0:
            return -1.0, 0.0, 0

        true_positive = float(((preds == 1) & (target == 1)).sum().item())
        precision = true_positive / max(predicted_positive, 1)
        if predicted_positive < min_trades:
            trade_factor = predicted_positive / max(min_trades, 1)
        else:
            trade_factor = 1.0

        precision_factor = precision / max(min_precision, 1e-6) if precision < min_precision else 1.0 + (precision - min_precision)
        objective = precision_factor * trade_factor
        return float(objective), float(precision), predicted_positive

    def fit(self, input1, input2, hour, extra_features, y, epochs=200, batch_size=32):
        validate_embedding_vocab(self, input1, input2, hour)
        input1_tensor = torch.tensor(input1, dtype=torch.long)
        input2_tensor = torch.tensor(input2, dtype=torch.long)
        hour_tensor = torch.tensor(hour, dtype=torch.long)
        extra_tensor = torch.tensor(extra_features, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(
            input1_tensor,
            input2_tensor,
            hour_tensor,
            extra_tensor,
            y_tensor,
        )
        val_size = max(1, int(len(dataset) * 0.15)) if len(dataset) >= 20 else 0
        train_size = len(dataset) - val_size
        if val_size > 0 and train_size > 0:
            split_generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=split_generator)
        else:
            train_dataset = dataset
            val_dataset = None

        loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_state_dict = None
        best_val_objective = float('-inf')
        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):
            self.train()
            for input1_b, input2_b, hour_b, extra_b, yb in loader:
                input1_b = input1_b.to(self.device)
                input2_b = input2_b.to(self.device)
                hour_b = hour_b.to(self.device)
                extra_b = extra_b.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                logits = self.forward_logits(input1_b, input2_b, hour_b, extra_b)
                loss = self.custom_loss(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

            if epoch % 10 == 0:
                self.eval()
                with torch.no_grad():
                    if val_dataset is not None:
                        val_input1, val_input2, val_hour, val_extra, val_y = val_dataset[:]
                        val_logits = self.forward_logits(
                            val_input1.to(self.device),
                            val_input2.to(self.device),
                            val_hour.to(self.device),
                            val_extra.to(self.device),
                        )
                        val_probs = torch.sigmoid(val_logits)
                        loss_val = self.custom_loss(val_logits, val_y.to(self.device)).item()
                        val_objective, val_precision, val_trades = self.validation_objective(val_probs, val_y.to(self.device))
                    else:
                        all_logits = self.forward_logits(
                            input1_tensor.to(self.device),
                            input2_tensor.to(self.device),
                            hour_tensor.to(self.device),
                            extra_tensor.to(self.device),
                        )
                        all_probs = torch.sigmoid(all_logits)
                        loss_val = self.custom_loss(all_logits, y_tensor.to(self.device)).item()
                        val_objective, val_precision, val_trades = self.validation_objective(all_probs, y_tensor.to(self.device))

                print(
                    f"Epoch {epoch} | Loss {loss_val:.5f} | "
                    f"ValObj {val_objective:.4f} | Precision {val_precision:.4f} | Trades {val_trades}"
                )

                if val_objective > best_val_objective:
                    best_val_objective = val_objective
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if loss_val <= self.target_loss:
                    print(f"\nEarly stopping en epoch {epoch}, loss={loss_val:.5f}")
                    break

                if no_improve_epochs >= 6:
                    print(f"\nEarly stopping por validacion en epoch {epoch}, objective={best_val_objective:.4f}")
                    break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

    def predict(self, input1, input2, hour, extra_features):
        self.eval()
        with torch.no_grad():
            pred = self.forward(input1, input2, hour, extra_features)
        return float(pred.reshape(-1)[0].item())



def save_trained_model(model, model_file):
    feature_mean = None
    feature_std = None
    if model.feature_mean is not None:
        feature_mean = np.asarray(model.feature_mean, dtype=np.float32).tolist()
    if model.feature_std is not None:
        feature_std = np.asarray(model.feature_std, dtype=np.float32).tolist()

    torch.save(
        {
            "model_type": "embedding",
            "input_dim_extra": model.input_dim_extra,
            "num_open": model.num_open,
            "num_close": model.num_close,
            "num_hours": model.num_hours,
            "emb_open_dim": model.emb_open_dim,
            "emb_close_dim": model.emb_close_dim,
            "emb_hour_dim": model.emb_hour_dim,
            "hidden_dim_1": int(getattr(model, "hidden_dim_1", 48)),
            "hidden_dim_2": int(getattr(model, "hidden_dim_2", 24)),
            "dropout_rate": model.dropout_rate,
            "weight_decay": model.weight_decay,
            "false_positive_cost_ratio": model.false_positive_cost_ratio,
            "state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
        model_file,
    )


def _build_embedding_model_from_checkpoint(checkpoint, input_dim_extra=None, lr=0.001):
    resolved_input_dim_extra = checkpoint.get("input_dim_extra", input_dim_extra)
    if resolved_input_dim_extra is None:
        feature_mean = checkpoint.get("feature_mean")
        resolved_input_dim_extra = len(feature_mean) if feature_mean is not None else len(EXTRA_FEATURE_COLUMNS)

    state_dict = checkpoint.get("state_dict", checkpoint)

    def _get_hidden_dims_from_state(sd):
        # Compatibilidad hacia atras: inferir arquitectura desde pesos guardados.
        h1 = checkpoint.get("hidden_dim_1")
        h2 = checkpoint.get("hidden_dim_2")
        if h1 is None:
            if "logits_model.0.weight" in sd:
                h1 = int(sd["logits_model.0.weight"].shape[0])
            elif "model.0.weight" in sd:
                h1 = int(sd["model.0.weight"].shape[0])
            else:
                h1 = 48
        if h2 is None:
            if "logits_model.2.weight" in sd:
                h2 = int(sd["logits_model.2.weight"].shape[0])
            elif "model.2.weight" in sd:
                h2 = int(sd["model.2.weight"].shape[0])
            else:
                h2 = 24
        return int(h1), int(h2)

    hidden_dim_1, hidden_dim_2 = _get_hidden_dims_from_state(state_dict)

    nn_model = BinaryNN(
        input_dim_extra=resolved_input_dim_extra,
        num_open=checkpoint.get("num_open", 256),
        num_close=checkpoint.get("num_close", 256),
        num_hours=checkpoint.get("num_hours", 32),
        emb_open_dim=checkpoint.get("emb_open_dim", 8),
        emb_close_dim=checkpoint.get("emb_close_dim", 8),
        emb_hour_dim=checkpoint.get("emb_hour_dim", 4),
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        dropout_rate=checkpoint.get("dropout_rate", 0.20),
        weight_decay=checkpoint.get("weight_decay", 1e-4),
        false_positive_cost_ratio=checkpoint.get("false_positive_cost_ratio", 2.0),
        lr=lr,
    )

    if any(key.startswith("model.") for key in state_dict.keys()):
        remapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                remapped_state_dict[key.replace("model.", "logits_model.", 1)] = value
            else:
                remapped_state_dict[key] = value
        state_dict = remapped_state_dict

    nn_model.load_state_dict(state_dict)
    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")
    nn_model.feature_mean = None if feature_mean is None else np.asarray(feature_mean, dtype=np.float32)
    nn_model.feature_std = None if feature_std is None else np.asarray(feature_std, dtype=np.float32)
    nn_model.eval()
    return nn_model


def load_trained_model(model_file, input_dim=None, input_dim_extra=None, lr=0.001):
    if model_file.endswith('.json'):
        resolved_input_dim = input_dim if input_dim is not None else 33
        nn_model = LegacyBinaryNN(input_dim=resolved_input_dim, lr=lr)
        with open(model_file, "r") as f:
            data = json.load(f)

        legacy_state = {
            'model.0.weight': torch.tensor(np.array(data['W1'], dtype=np.float32).T),
            'model.0.bias': torch.tensor(np.array(data['b1'], dtype=np.float32).reshape(-1)),
            'model.2.weight': torch.tensor(np.array(data['W2'], dtype=np.float32).T),
            'model.2.bias': torch.tensor(np.array(data['b2'], dtype=np.float32).reshape(-1)),
            'model.4.weight': torch.tensor(np.array(data['W3'], dtype=np.float32).T),
            'model.4.bias': torch.tensor(np.array(data['b3'], dtype=np.float32).reshape(-1)),
            'model.6.weight': torch.tensor(np.array(data['W4'], dtype=np.float32).T),
            'model.6.bias': torch.tensor(np.array(data['b4'], dtype=np.float32).reshape(-1)),
        }
        nn_model.load_state_dict(legacy_state)
    else:
        checkpoint = torch.load(
            model_file,
            map_location=_get_device(),
            weights_only=False,
        )
        if checkpoint.get("model_type") == "ensemble":
            models = [
                _build_embedding_model_from_checkpoint(model_checkpoint, input_dim_extra=input_dim_extra, lr=lr)
                for model_checkpoint in checkpoint.get("models", [])
            ]
            return {
                "model_type": "ensemble",
                "models": models,
                "weights": checkpoint.get("weights", [1.0] * len(models)),
            }

        state_dict = checkpoint.get("state_dict", checkpoint)
        is_embedding_checkpoint = (
            checkpoint.get("model_type") == "embedding"
            or "emb_open.weight" in state_dict
        )

        if is_embedding_checkpoint:
            nn_model = _build_embedding_model_from_checkpoint(checkpoint, input_dim_extra=input_dim_extra, lr=lr)
        else:
            resolved_input_dim = checkpoint.get("input_dim", input_dim)
            if resolved_input_dim is None:
                raise ValueError("No se pudo resolver input_dim para checkpoint legacy")
            nn_model = LegacyBinaryNN(input_dim=resolved_input_dim, lr=lr)

        if not is_embedding_checkpoint:
            nn_model.load_state_dict(state_dict)
            feature_mean = checkpoint.get("feature_mean")
            feature_std = checkpoint.get("feature_std")
            nn_model.feature_mean = None if feature_mean is None else np.asarray(feature_mean, dtype=np.float32)
            nn_model.feature_std = None if feature_std is None else np.asarray(feature_std, dtype=np.float32)

    nn_model.eval()
    return nn_model


# ----------------------------------------------------------
# # PREDICCIÓN DESDE INPUTS
# ----------------------------------------------------------
def predict_from_inputs(nn, input1, input2, hour, extra_features=None):
    if isinstance(nn, dict) and nn.get("model_type") == "ensemble":
        models = nn.get("models", [])
        weights = nn.get("weights", [])
        if not models:
            return 0.5

        if not weights or len(weights) != len(models):
            weights = [1.0] * len(models)

        total_weight = float(sum(weights))
        if total_weight <= 0:
            weights = [1.0] * len(models)
            total_weight = float(len(models))

        weighted_sum = 0.0
        for model, weight in zip(models, weights):
            weighted_sum += predict_from_inputs(model, input1, input2, hour, extra_features) * float(weight)
        return float(weighted_sum / total_weight)

    if isinstance(nn, LegacyBinaryNN):
        input1_raw = str(input1)
        input2_raw = str(input2)
        hour_raw = str(hour)
        input1 = input1_raw.zfill(8)
        input2 = input2_raw.zfill(8)
        hour = hour_raw.zfill(5)

        a = np.array([int(bit) for bit in input1], dtype=np.float32)
        b = np.array([int(bit) for bit in input2], dtype=np.float32)
        c = np.array([int(bit) for bit in hour], dtype=np.float32)

        features = [a, b, c]
        if extra_features is not None:
            features.append(np.asarray(extra_features, dtype=np.float32).reshape(-1))

        X = np.concatenate(features).reshape(1, -1)

        if nn.feature_mean is not None and nn.feature_std is not None:
            feature_mean = np.asarray(nn.feature_mean, dtype=np.float32).reshape(1, -1)
            feature_std = np.asarray(nn.feature_std, dtype=np.float32).reshape(1, -1)
            X = (X - feature_mean) / (feature_std + 1e-8)

        expected_dim = nn.input_dim
        if X.shape[1] != expected_dim:
            if extra_features is None and X.shape[1] < expected_dim:
                padding = np.zeros(expected_dim - X.shape[1], dtype=np.float32)
                X = np.concatenate((X.reshape(-1), padding)).reshape(1, -1)
            else:
                raise ValueError(
                    "Dimensión de entrada inválida para checkpoint legacy: "
                    f"X={X.shape[1]}, expected={expected_dim}."
                )

        nn.eval()
        with torch.no_grad():
            pred = float(nn.forward(X).reshape(-1)[0].item())
        return pred

    if nn.feature_mean is not None and nn.feature_std is not None:
        feature_mean = np.asarray(nn.feature_mean, dtype=np.float32)
        feature_std = np.asarray(nn.feature_std, dtype=np.float32)
    else:
        feature_mean = None
        feature_std = None

    input1_id = _parse_binary_id(input1, OPEN_NODE_BITS)
    input2_id = _parse_binary_id(input2, CLOSE_NODE_BITS)
    hour_id = _parse_binary_id(hour, HOUR_BITS)

    validate_embedding_vocab(
        nn,
        np.asarray([input1_id], dtype=np.int64),
        np.asarray([input2_id], dtype=np.int64),
        np.asarray([hour_id], dtype=np.int64),
    )

    if extra_features is None:
        extra = np.zeros(len(EXTRA_FEATURE_COLUMNS), dtype=np.float32)
    else:
        extra = np.asarray(extra_features, dtype=np.float32).reshape(-1)

    if nn.input_dim_extra >= (len(EXTRA_FEATURE_COLUMNS) + OPEN_NODE_BITS + CLOSE_NODE_BITS):
        extra = _augment_extra_features(input1_id, input2_id, extra)

    if extra.shape[0] != nn.input_dim_extra:
        if extra.shape[0] > nn.input_dim_extra:
            extra = extra[:nn.input_dim_extra]
        else:
            padding = np.zeros(nn.input_dim_extra - extra.shape[0], dtype=np.float32)
            extra = np.concatenate([extra, padding])

    if feature_mean is not None and feature_std is not None:
        extra = (extra - feature_mean) / (feature_std + 1e-8)

    input1_tensor = torch.tensor([input1_id], dtype=torch.long)
    input2_tensor = torch.tensor([input2_id], dtype=torch.long)
    hour_tensor = torch.tensor([hour_id], dtype=torch.long)
    extra_tensor = torch.tensor(extra.reshape(1, -1), dtype=torch.float32)

    nn.eval()
    with torch.no_grad():
        pred = float(nn.forward(input1_tensor, input2_tensor, hour_tensor, extra_tensor).reshape(-1)[0].item())
    
    return pred

# ----------------------------------------------------------
# CARGA DE DATOS
# ----------------------------------------------------------
def load_data(csv_file, return_stats=False):
    df = pd.read_csv(csv_file, dtype=str)

    input1_list = []
    input2_list = []
    hour_list = []
    extra_list = []
    Y_list = []

    for _, row in df.iterrows():
        input1_list.append(_parse_binary_id(row["input1"], OPEN_NODE_BITS))
        input2_list.append(_parse_binary_id(row["input2"], CLOSE_NODE_BITS))
        hour_list.append(_parse_binary_id(row["hour"], HOUR_BITS))
        extra_list.append(_extract_extra_features(row))
        Y_list.append(float(row["output"]))

    input1_arr = np.asarray(input1_list, dtype=np.int64)
    input2_arr = np.asarray(input2_list, dtype=np.int64)
    hour_arr = np.asarray(hour_list, dtype=np.int64)
    X_extra = np.vstack([
        _augment_extra_features(input1_id, input2_id, extra)
        for input1_id, input2_id, extra in zip(input1_arr, input2_arr, extra_list)
    ]).astype(np.float32)
    Y = np.array(Y_list).reshape(-1, 1).astype(np.float32)

    mean = X_extra.mean(axis=0).astype(np.float32)
    std = (X_extra.std(axis=0) + 1e-8).astype(np.float32)
    X_extra = (X_extra - mean) / std

    if return_stats:
        return input1_arr, input2_arr, hour_arr, X_extra, Y, {"mean": mean, "std": std}

    return input1_arr, input2_arr, hour_arr, X_extra, Y


def has_minimum_training_data(csv_file, min_samples=MIN_TRAINING_SAMPLES, min_class_count=2):
    if not os.path.exists(csv_file):
        return False, {"reason": "missing_file", "samples": 0, "class_count": 0}

    df = pd.read_csv(csv_file)
    if df.empty:
        return False, {"reason": "empty_file", "samples": 0, "class_count": 0}

    samples = int(len(df))
    class_count = int(df["output"].nunique()) if "output" in df.columns else 0

    if samples < min_samples:
        return False, {"reason": "low_samples", "samples": samples, "class_count": class_count}
    if class_count < min_class_count:
        return False, {"reason": "single_class", "samples": samples, "class_count": class_count}

    return True, {"reason": "ok", "samples": samples, "class_count": class_count}
  

def execute_entrenar(principal_symbol, mercados, list_algorithms = None):
    
    crear_carpeta_si_no_existe(f'output/{principal_symbol}/data_for_neuronal/model_trainer')
    list_algorithms = list_algorithms if list_algorithms else ['UP', 'DOWN']
    
    for algorithm in list_algorithms:
        for mercado in mercados:
            path = f'output/{principal_symbol}/data_for_neuronal/data/data_{mercado}_{algorithm}.csv' 
            is_valid, dataset_info = has_minimum_training_data(path)
            if not is_valid:
                print(
                    f"Saltando entrenamiento {principal_symbol}-{mercado}-{algorithm}: "
                    f"{dataset_info['reason']} | samples={dataset_info['samples']} | classes={dataset_info['class_count']}"
                )
                continue
            input1_ids, input2_ids, hour_ids, X_extra, Y, norm_stats = load_data(path, return_stats=True)
            print("Datos cargados:")
            print("X_extra shape:", X_extra.shape)
            print("Y shape:", Y.shape)
            num_open, num_close, num_hours = get_embedding_vocab_sizes(input1_ids, input2_ids, hour_ids)
            nn = BinaryNN(
                input_dim_extra=X_extra.shape[1],
                num_open=num_open,
                num_close=num_close,
                num_hours=num_hours,
                lr=0.001,
                target_loss=0.10,
            )
            nn.feature_mean = norm_stats["mean"]
            nn.feature_std = norm_stats["std"]
            nn.fit(input1_ids, input2_ids, hour_ids, X_extra, Y, epochs=80, batch_size=64)
            model_path = f'output/{principal_symbol}/data_for_neuronal/model_trainer/model_{mercado}_{algorithm}.pt'
            save_trained_model(nn, model_path)
            print(f"Modelo entrenado guardado en '{model_path}'")
            
    
if __name__ == "__main__":
    execute_entrenar('AUDCHF', ['Asia'], list_algorithms = None)