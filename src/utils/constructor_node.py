import numpy as np
import random
import pandas as pd
import time


class NodeGenerator:

    def __init__(self, df, min_cond=5, max_cond=8, pct_min=10, pct_max=90):

        self.MIN_COND = min_cond
        self.MAX_COND = max_cond

        self.features = [c for c in df.columns if c not in ["time", "label"]]

        # Convertir a numpy (MUY RAPIDO)
        self.data = df[self.features].to_numpy(dtype=np.float64)

        # Labels codificados
        labels_raw = df["label"].values
        self.unique_labels, self.labels = np.unique(labels_raw, return_inverse=True)

        self.n_rows, self.n_features = self.data.shape

        # Rangos
        self.mins = np.min(self.data, axis=0)
        self.maxs = np.max(self.data, axis=0)

        # Percentiles precomputados por feature
        self.pct_min = pct_min
        self.pct_max = pct_max
        self.pct_values = np.percentile(
            self.data,
            np.arange(self.pct_min, self.pct_max + 1),
            axis=0
        )

    # ---------- generar condicion ----------
    def _generar_condicion(self):

        idx = random.randrange(self.n_features)
        if self.pct_min is not None and self.pct_max is not None:
            pct_idx = random.randrange(0, self.pct_values.shape[0])
            valor = float(self.pct_values[pct_idx, idx])
        else:
            valor = random.uniform(self.mins[idx], self.maxs[idx])
        op = random.getrandbits(1)

        return (idx, op, valor)

    # ---------- generar nodo ----------
    def _generar_nodo(self):

        num_cond = random.randint(self.MIN_COND, self.MAX_COND)

        usadas = set()
        condiciones = []

        while len(condiciones) < num_cond:
            c = self._generar_condicion()
            if c[0] not in usadas:
                usadas.add(c[0])
                condiciones.append(c)

        mask = np.ones(self.n_rows, dtype=bool)

        for idx, op, v in condiciones:
            col = self.data[:, idx]
            if op == 0:
                mask &= col < v
            else:
                mask &= col >= v

        subset = self.labels[mask]

        if subset.size == 0:
            return None

        counts = np.bincount(subset)
        label_idx = np.argmax(counts)

        return {
            "label": self.unique_labels[label_idx],
            "conditions": [
                (self.features[i], "<" if op == 0 else ">=", v)
                for i, op, v in condiciones
            ]
        }

    # ---------- función pública ----------
    def generar_nodos(self, n):

        nodos = []

        while len(nodos) < n:
            nodo = self._generar_nodo()
            if nodo:
                nodos.append(nodo)

        return nodos



# import numpy as np
# import pandas as pd
# import random


# class NodeGenerator:

#     def __init__(self, df, min_cond=3, max_cond=7, pct_min=10, pct_max=90, exclude_columns=None, random_state=None):

#         if random_state is not None:
#             random.seed(random_state)
#             np.random.seed(random_state)

#         self.MIN_COND = min_cond
#         self.MAX_COND = max_cond

#         exclude = set(["time", "label"]) if exclude_columns is None else set(exclude_columns) | {"label"}
#         numeric_df = df.copy()
#         for col in numeric_df.columns:
#             if col not in exclude:
#                 numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

#         self.features = [c for c in numeric_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(numeric_df[c])]

#         # Convertir a numpy (MUY RAPIDO)
#         self.data = numeric_df[self.features].to_numpy(dtype=np.float64)

#         self.n_rows, self.n_features = self.data.shape

#         # Rangos
#         self.mins = np.min(self.data, axis=0)
#         self.maxs = np.max(self.data, axis=0)

#         # Config de muestreo de thresholds
#         self.pct_min = pct_min
#         self.pct_max = pct_max

#     def _pick_threshold_value(self, idx):
#         col = self.data[:, idx]
#         valid = col[np.isfinite(col)]
#         if valid.size == 0:
#             return None
#         if self.pct_min is not None and self.pct_max is not None:
#             return float(valid[random.randrange(valid.size)])
#         return float(random.uniform(np.min(valid), np.max(valid)))

#     # ---------- generar condicion ----------
#     def _generar_condicion(self):

#         idx = random.randrange(self.n_features)
#         valor = self._pick_threshold_value(idx)
#         if valor is None:
#             return None
#         op = random.getrandbits(1)

#         return (idx, op, valor)

#     def _build_mask(self, condiciones):
#         mask = np.ones(self.n_rows, dtype=bool)

#         for idx, op, v in condiciones:
#             col = self.data[:, idx]
#             valid = np.isfinite(col)
#             if op == 0:
#                 mask &= valid & (col < v)
#             else:
#                 mask &= valid & (col >= v)

#         return mask

#     def generar_feature(self, return_definition=False):
#         num_cond = random.randint(self.MIN_COND, self.MAX_COND)

#         usadas = set()
#         condiciones = []

#         attempts = 0
#         max_attempts = max(self.n_features * 4, 20)
#         while len(condiciones) < num_cond and attempts < max_attempts:
#             attempts += 1
#             c = self._generar_condicion()
#             if c is None or c[0] in usadas:
#                 continue
#             usadas.add(c[0])
#             condiciones.append(c)

#         if not condiciones:
#             raise ValueError("No se pudo generar una feature de nodo válida")

#         mask = self._build_mask(condiciones)
#         definition = {
#             "conditions": [
#                 (self.features[i], "<" if op == 0 else ">=", v)
#                 for i, op, v in condiciones
#             ]
#         }

#         if return_definition:
#             return mask.astype(np.int8), definition
#         return mask.astype(np.int8)

#     def generar_features(self, n, prefix="node_", return_definitions=False):
#         data = {}
#         definitions = []

#         for idx in range(n):
#             feature_values, definition = self.generar_feature(return_definition=True)
#             data[f"{prefix}{idx}"] = feature_values
#             definitions.append(definition)

#         df_features = pd.DataFrame(data)
#         if return_definitions:
#             return df_features, definitions
#         return df_features

#     def apply_feature(self, node_definition, df):
#         mask = np.ones(len(df), dtype=bool)
#         for feature_name, op_text, value in node_definition.get("conditions", []):
#             if feature_name not in df.columns:
#                 mask &= False
#                 continue
#             col = pd.to_numeric(df[feature_name], errors="coerce").to_numpy(dtype=np.float64)
#             valid = np.isfinite(col)
#             if op_text == "<":
#                 mask &= valid & (col < value)
#             else:
#                 mask &= valid & (col >= value)
#         return mask.astype(np.int8)

#     def apply_features(self, node_definitions, prefix="node_", df=None):
#         target_df = self._to_frame(df)
#         data = {}
#         for idx, definition in enumerate(node_definitions):
#             data[f"{prefix}{idx}"] = self.apply_feature(definition, target_df)
#         return pd.DataFrame(data, index=target_df.index)

#     def _to_frame(self, df=None):
#         if df is None:
#             return pd.DataFrame(self.data, columns=self.features)
#         return df.copy()

#     # ---------- generar nodo ----------
#     def _generar_nodo(self):

#         mask, definition = self.generar_feature(return_definition=True)

#         if not mask.any():
#             return None

#         return definition

#     # ---------- función pública ----------
#     def generar_nodos(self, n):

#         nodos = []

#         while len(nodos) < n:
#             nodo = self._generar_nodo()
#             if nodo:
#                 nodos.append(nodo)

#         return nodos


