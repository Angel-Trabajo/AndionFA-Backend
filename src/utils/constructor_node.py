import numpy as np
import random


class NodeGenerator:

    def __init__(self, df, min_cond=3, max_cond=6, pct_min=10, pct_max=90):

        self.MIN_COND = min_cond
        self.MAX_COND = max_cond

        self.features = [c for c in df.columns if c not in ["time", "label"]]

        # Convertir a numpy (MUY RAPIDO)
        self.data = df[self.features].to_numpy(dtype=np.float64)

        # Labels codificados
        labels_raw = df["label"].values
        self.unique_labels, self.labels = np.unique(labels_raw, return_inverse=True)

        self.n_rows, self.n_features = self.data.shape
        self.binary_feature_idx = set()

        for idx in range(self.n_features):
            unique_values = np.unique(self.data[:, idx])
            finite_values = unique_values[np.isfinite(unique_values)]
            if finite_values.size > 0 and np.all(np.isin(finite_values, [0.0, 1.0])):
                self.binary_feature_idx.add(idx)

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

        # Peso por feature: prioriza variables con mayor varianza efectiva.
        feature_std = np.nanstd(self.data, axis=0)
        finite_mask = np.isfinite(feature_std)
        safe_std = np.where(finite_mask, feature_std, 0.0)
        safe_std = np.maximum(safe_std, 1e-6)
        self.feature_weights = (safe_std / safe_std.sum()).astype(np.float64)

    def _sample_feature_idx(self):
        if self.n_features == 0:
            raise ValueError("No hay features disponibles para generar nodos")
        # Si por algún motivo los pesos no son válidos, fallback uniforme.
        if not np.isfinite(self.feature_weights).all() or self.feature_weights.sum() <= 0:
            return random.randrange(self.n_features)
        return int(np.random.choice(self.n_features, p=self.feature_weights))

    # ---------- generar condicion ----------
    def _generar_condicion(self):

        idx = self._sample_feature_idx()
        if idx in self.binary_feature_idx:
            return (idx, 2, 1.0)

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
            elif op == 2:
                mask &= col == v
            else:
                mask &= col >= v

        subset = self.labels[mask]

        if subset.size == 0:
            return None

        counts = np.bincount(subset)
        label_idx = np.argmax(counts)

        return {
            "label": self.unique_labels[label_idx],
            "num_conditions": int(num_cond),
            "conditions": [
                (self.features[i], "<" if op == 0 else ("==" if op == 2 else ">="), v)
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




