# Reentrenamiento de Modelos con STOCH

## Estado Actual
- **Código actualizado:** Sistema completo preparado para usar STOCH
- **Modelos antiguos:** Compatibles (24 features) - seguirán funcionando sin STOCH
- **Nuevos modelos:** Si reentrenás, tendrán 26 features e incluirán STOCH automáticamente

## ¿Qué cambió?

### Entrada a la red neuronal
**Antes (24 features):**
- 16 bits (nodos binarios)
- 5 contexto: ATR, ADX, RSI, hour, spread
- 3 contexto apertura: ATR, ADX, RSI

**Ahora (26 features):**
- 16 bits (nodos binarios)  
- **6 contexto:** ATR, ADX, RSI, **STOCH**, hour, spread
- **4 contexto apertura:** ATR, ADX, RSI, **STOCH**

### Flujo de datos STOCH
```
config/extractor/Ext-*.csv (definen STOCH;...)
    ↓
indicadores_for_crossing.py (calcula con talib)
    ↓
output/{symbol}/extrac/*.csv (con columna STOCH_14_3_SMA_3_SMA_pos0)
    ↓
backtester.py (_get_market_context extrae STOCH)
    ↓
predict_from_inputs(..., stoch=75.0)
    ↓
Modelo neuronal (26 features con STOCH normalizado)
```

## Pasos para Reentrenar

### Opción A: Desde principal_script.py (recomendado)

```python
from src.scripts.principal_script import principal_main

# Ejecuta el pipeline completo:
# 1. Extrae indicadores (incluye STOCH)
# 2. Ejecuta backtesting (recolecta trades con STOCH)
# 3. Genera dataset CSV con stoch/stoch_open
# 4. Entrena modelos nuevos (26 features con STOCH)
# 5. Guarda modelos en output/{symbol}/data_for_neuronal/model_trainer/

principal_main()
```

### Opción B: Paso a paso con Backtester

```python
from src.neuronal.backtester import Backtester

# 1. Ejecuta backtesting que genera trades con STOCH
bt = Backtester(
    principal_symbol="AUDCAD",
    mercado="Asia",
    algorithm="UP",
    date_end="2026-03-18"
)
bt.run()

# Esto genera:
# - output/AUDCAD/data_for_neuronal/trade_dataset/*.json (con stoch/stoch_open)
# - output/AUDCAD/data_for_neuronal/data/*.csv (con columnas stoch/stoch_open)
```

### Opción C: Entrenar directamente

```python
from src.neuronal.entrenar import train_binary_nn
import pandas as pd

# Si ya tienes CSV con stoch/stoch_open:
train_binary_nn(
    principal_symbol="AUDCAD",
    mercado="Asia",
    algorithm="UP"
)

# Lee automáticamente:
# output/AUDCAD/data_for_neuronal/data/data_Asia_UP.csv
# Detecta que tiene stoch → entrena con 26 features
# Guarda modelo con 26 features
```

## Verificación

Después de reentrenar, los **nuevos warnings deberían desaparecer**:

**Antes:**
```
⚠️ mismatch: input=26 model=24
⚠️ mismatch: input=26 model=24
```

**Después (sin warnings):**
```
CLOSE [MODEL] 2014-07-04 09:00:00 | pips=5.50 | dur=6
```

## Compatibilidad

| Escenario | Resultado |
|-----------|-----------|
| Modelo 24 features + STOCH en código | ✅ Funciona (trunca STOCH a 24) |
| Modelo 26 features + STOCH en código | ✅ Funciona (usa todos 26) |
| Modelo 24 features + sin STOCH | ✅ Funciona (original) |
| Modelo 26 features + sin STOCH | ❌ Error (requiere STOCH) |

## Notas

- **STOCH elegido:** STOCH(14, 3, 3) - parámetros estándar
- **Normalización STOCH:** `stoch_n = clip(stoch / 100, 0, 1)` - rango 0-100 → 0-1
- **Default si STOCH no existe:** 50.0 (neutral)
- **Generación automática:** Al reentrenar, si indicadores tienen STOCH, CSV incluirá stoch/stoch_open automáticamente

## Próximos pasos

1. Ejecuta el reentrenamiento cuando esté listo
2. Los nuevos modelos se guardarán en:
   - `output/{symbol}/data_for_neuronal/model_trainer/model_{mercado}_{algorithm}.json`
3. Al usar esos modelos, automáticamente usarán STOCH sin mensajes de incompatibilidad
