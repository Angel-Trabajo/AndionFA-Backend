import numpy as np
import random
import pandas as pd
import time


class NodeGenerator:

    def __init__(self, df, min_cond=7, max_cond=12, pct_min=10, pct_max=90):

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


