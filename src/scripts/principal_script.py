import os
import sys
import json
import operator
import ast
import time as time_module
import pandas as pd 
import numpy as np
import MetaTrader5 as mt5

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.routes.peticiones import get_historical_data, get_timeframes, initialize_mt5
from src.utils.indicadores_for_principal_script import generate_files
from src.db.query import get_nodes_by_label
from src.neuronal.entrenar import EXTRA_FEATURE_COLUMNS, load_trained_model, predict_from_inputs, load_data, validate_embedding_vocab
from src.signals.event_generator import add_event_features, has_entry_event
from src.utils.common_functions import hora_en_mercado, crear_carpeta_si_no_existe, should_backtest_strategy


# =============================================================================
# TRADING EN TIEMPO REAL
# =============================================================================

CANDLES_BUFFER = 200  # velas históricas para calcular indicadores (periodo max=80, x2 warmup + margen)


class TradingEngine:
    """
    Versión en tiempo real de Backtest para un símbolo + mercado + algoritmo.
    La lógica de nodos/indicadores/modelo es idéntica a test_iteration.
    MT5 gestiona SL/TP de forma nativa.
    """

    def __init__(self, principal_symbol, mercado, algorithm, general_config):
        self.principal_symbol = principal_symbol
        self.mercado = mercado
        self.algorithm = algorithm
        self.other_algorithm = 'DOWN' if algorithm == 'UP' else 'UP'
        self.general_config = general_config

        with open(f'config/divisas/{principal_symbol}/config_{principal_symbol}.json', 'r', encoding='utf-8') as f:
            config_symbol = json.load(f)

        self.list_symbols = config_symbol['list_symbol'].copy()
        self.list_symbols.insert(0, principal_symbol)

        self.timeframe = get_timeframes()['timeframes'][general_config['timeframe']]
        count_proces = int(general_config.get('use_proces', 40))//2
        self.list_files = general_config['indicators_files']
        if len(self.list_files) > count_proces:
            self.list_files = self.list_files[:count_proces]
            
        self.pip_size = 0.01 if 'JPY' in principal_symbol.upper() else 0.0001
        info = mt5.symbol_info(principal_symbol)
        self.point_size = info.point if info else self.pip_size / 10

        # Parámetros de riesgo / cierre (igual que Backtest)
        self.stop_loss = int(general_config.get('stop_loss', 20))
        self.take_profit = int(general_config.get('take_profit', 150))
        self.max_holding = 120
        self.min_model_holding = 15
        self.close_confirmation_bars = 2
        self.close_threshold_floor = 0.60
        self.min_open_symbol_confirmations = int(general_config.get('MinOpenSymbolConfirmations', 4))
        self.lot_size = float(general_config.get('lot_size', 0.01))
        self.magic = abs(hash(f"{principal_symbol}_{mercado}_{algorithm}")) % 99999 + 1

        self.horas_mercado = [hora_en_mercado(h, mercado) for h in range(24)]
        self.operadores = {
            "<": operator.lt, "<=": operator.le, ">": operator.gt,
            ">=": operator.ge, "==": operator.eq, "!=": operator.ne,
        }

        # Estado de la operación
        self.is_open = False
        self.ticket = None
        self.open_price_open = 0.0
        self.entry_red_open = []
        self.cierre = 0
        self.model_close_streak = 0
        self.time_comienzo = None
        self.current_stop_loss = self.stop_loss
        self.current_take_profit = self.take_profit

        # Datos actuales (se actualizan en fetch_indicators)
        self.indicators = {}
        self.current_row = None       # última vela cerrada  ("row" en backtest)
        self.prev_row = None          # penúltima vela cerrada ("prev_row" en backtest)
        self.current_forming_time_np = None  # tiempo de la vela en formación
        self.current_forming_hour = 0
        self.current_open_price = 0.0

        self._load()

    # ------------------------------------------------------------------
    # Carga inicial
    # ------------------------------------------------------------------

    def _load(self):
        # Mappings
        open_path = f'output/{self.principal_symbol}/data_for_neuronal/maping/maping_open_{self.mercado}_{self.algorithm}.json'
        close_path = f'output/{self.principal_symbol}/data_for_neuronal/maping/maping_close_{self.mercado}_{self.algorithm}.json'
        with open(open_path) as f:
            self.maping_open = json.load(f)
        with open(close_path) as f:
            self.maping_close = json.load(f)

        # Score / threshold
        score_path = f'output/{self.principal_symbol}/data_for_neuronal/best_score/score_{self.mercado}_{self.algorithm}.json'
        self.close_threshold = self.close_threshold_floor
        ensemble_model_path = None
        if os.path.exists(score_path):
            with open(score_path) as f:
                score_data = json.load(f)
            metrics = score_data.get('metrics', {})
            ensemble_info = metrics.get('ensemble', {})
            ensemble_model_path = ensemble_info.get('model_path')
            self.close_threshold = max(
                ensemble_info.get('threshold', metrics.get('best_threshold', 0.5)),
                self.close_threshold_floor,
            )

        # Modelo
        model_path = ensemble_model_path
        if not model_path or not os.path.exists(model_path):
            model_path = f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.pt'
        data_path = f'output/{self.principal_symbol}/data_for_neuronal/data/data_{self.mercado}_{self.algorithm}.csv'

        input1_ids, input2_ids, hour_ids, X_extra, Y = load_data(data_path)
        self.nn = load_trained_model(model_path, input_dim_extra=X_extra.shape[1])
        validate_embedding_vocab(self.nn, input1_ids, input2_ids, hour_ids)

        # Nodos
        self.dict_nodos = {}
        for symbol in self.list_symbols:
            self.dict_nodos[symbol] = self._parsear_nodos(
                get_nodes_by_label(self.principal_symbol, symbol, self.mercado, self.algorithm) or []
            )
        self.nodos_close = self._parsear_nodos(
            get_nodes_by_label(self.principal_symbol, self.principal_symbol, mercado=None, label=self.other_algorithm) or []
        )

    # ------------------------------------------------------------------
    # Helpers (idénticos a Backtest)
    # ------------------------------------------------------------------

    def _parsear_nodos(self, nodos):
        if not nodos:
            return []
        return [{'key': n[0], 'conditions': ast.literal_eval(n[0]), 'file': n[1]} for n in nodos]

    def _build_combined(self, df):
        return {
            'df': df,
            'values': df.values,
            'col_map': {col: i for i, col in enumerate(df.columns)},
            'index_values': df.time.values,
        }

    def _enrich(self, df):
        df = df.copy()
        df['ret_1'] = df['close'] - df['open']
        df['ret_3'] = df['close'] - df['close'].shift(3)
        df['ret_10'] = df['close'] - df['close'].shift(10)
        df['range_1'] = df['high'] - df['low']
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_10'] = df['close'].rolling(10).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['trend'] = df['close'] - df['ma_5']
        df['trend_10'] = df['close'] - df['ma_10']
        df['trend_20'] = df['close'] - df['ma_20']
        df['vol'] = df['close'].rolling(5).std()
        df['vol_10'] = df['close'].rolling(10).std()
        df['vol_20'] = df['close'].rolling(20).std()
        df['zscore_20'] = (df['close'] - df['ma_20']) / (df['vol_20'] + 1e-8)
        df['momentum_ratio'] = df['ret_3'] / (df['vol_10'] + 1e-8)
        df = add_event_features(df)
        return df

    def _get_market_features(self, row):
        return np.array([float(getattr(row, col, 0.0)) for col in EXTRA_FEATURE_COLUMNS], dtype=np.float32)

    def _cumple_condiciones_fast(self, df_struct, row_idx, condiciones):
        row = df_struct['values'][row_idx]
        col_map = df_struct['col_map']
        for col, op, valor in condiciones:
            if col not in col_map:
                return False
            v = row[col_map[col]]
            if v is None or not self.operadores[op](v, valor):
                return False
        return True

    # ------------------------------------------------------------------
    # Obtención de datos frescos
    # ------------------------------------------------------------------

    def apply_shared_indicators(self, shared_indicators):
        """
        Recibe la cache global {symbol: {file_key: df_struct}} calculada por TradingServer
        y construye self.indicators reutilizando los datos ya calculados.
        También actualiza current_row, prev_row y current_forming_time_np a partir
        de los datos enriquecidos del principal_symbol que vienen en la cache.
        """
        self.indicators = {}
        for symbol in self.list_symbols:
            symbol_cache = shared_indicators.get(symbol)
            if not symbol_cache:
                return False
            for file in self.list_files:
                file_key = file.split('.')[0]
                df_struct = symbol_cache.get(file_key)
                if df_struct is None:
                    return False
                key = f'{self.principal_symbol}_{symbol}_{file_key}'
                self.indicators[key] = df_struct

        # Datos enriquecidos del principal_symbol para current_row / prev_row
        main_enriched = shared_indicators.get(self.principal_symbol, {}).get('__enriched__')
        if main_enriched is None or len(main_enriched) < 3:
            return False

        forming = main_enriched.iloc[-1]
        self.current_row = main_enriched.iloc[-2]
        self.prev_row = main_enriched.iloc[-3]
        self.current_forming_time_np = np.datetime64(forming['time'])
        self.current_forming_hour = pd.Timestamp(forming['time']).hour
        tick = mt5.symbol_info_tick(self.principal_symbol)
        self.current_open_price = tick.ask if self.algorithm == 'UP' else tick.bid
        return True

    # ------------------------------------------------------------------
    # Lógica de apertura
    # ------------------------------------------------------------------

    def _resolve_entry_open_nodes(self):
        matched_symbols = 0
        open_nodes = []
        time_np = self.current_forming_time_np

        for symbol in self.list_symbols:
            for nodo in self.dict_nodos.get(symbol, []):
                key = f'{self.principal_symbol}_{symbol}_{nodo["file"].split("_")[0]}'
                if key not in self.indicators:
                    continue
                df_struct = self.indicators[key]
                pos = df_struct['index_values'].searchsorted(time_np)
                if pos == 0:
                    continue
                if self._cumple_condiciones_fast(df_struct, pos - 1, nodo['conditions']):
                    matched_symbols += 1
                    if symbol == self.principal_symbol:
                        nodo_open = self.maping_open.get(nodo['key'])
                        if nodo_open is not None:
                            open_nodes.append(nodo_open)
                    break

        if matched_symbols < self.min_open_symbol_confirmations or not open_nodes:
            return []
        return list(dict.fromkeys(open_nodes))

    def _get_risk_limits(self):
        if self.prev_row is None:
            return self.stop_loss, self.take_profit
        vol_10 = float(getattr(self.prev_row, 'vol_10', 0.0))
        adaptive_stop = max(self.stop_loss, 1.5 * vol_10 / self.pip_size)
        adaptive_take = max(self.take_profit, 2.0 * vol_10 / self.pip_size)
        return adaptive_stop, adaptive_take

    # ------------------------------------------------------------------
    # Órdenes MT5
    # ------------------------------------------------------------------

    def _get_filling_mode(self):
        info = mt5.symbol_info(self.principal_symbol)
        if info is None:
            return mt5.ORDER_FILLING_IOC
        fm = info.filling_mode
        if fm & 2:  # IOC
            return mt5.ORDER_FILLING_IOC
        if fm & 1:  # FOK
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def _get_position(self):
        if self.ticket is None:
            return None
        positions = mt5.positions_get(ticket=self.ticket)
        return positions[0] if positions else None

    def _check_closed_by_mt5(self):
        """Devuelve (True, pips) si MT5 cerró la posición por SL/TP, sino (False, 0)."""
        if not self.is_open or self.ticket is None:
            return False, 0.0
        if self._get_position() is not None:
            return False, 0.0
        # La posición ya no existe → buscar en historial
        pips = 0.0
        deals = mt5.history_deals_get(position=self.ticket)
        if deals:
            entry_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_IN]
            exit_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
            if entry_deals and exit_deals:
                ep = entry_deals[0].price
                xp = exit_deals[0].price
                pips = (xp - ep) / self.pip_size if self.algorithm == 'UP' else (ep - xp) / self.pip_size
        return True, pips

    def _open_trade(self):
        tick = mt5.symbol_info_tick(self.principal_symbol)
        if tick is None:
            return None
        if self.algorithm == 'UP':
            price = tick.ask
            sl = round(price - self.current_stop_loss * self.pip_size, 5)
            tp = round(price + self.current_take_profit * self.pip_size, 5)
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = round(price + self.current_stop_loss * self.pip_size, 5)
            tp = round(price - self.current_take_profit * self.pip_size, 5)
            order_type = mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.principal_symbol,
            "volume": self.lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic,
            "comment": f"{self.algorithm}_{self.mercado}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(),
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  ✅ OPEN {self.algorithm} {self.principal_symbol}/{self.mercado} @ {price} SL={sl} TP={tp} ticket={result.order}")
            return result.order
        print(f"  ❌ Error apertura {self.principal_symbol}/{self.mercado}/{self.algorithm}: {result.retcode} - {result.comment}")
        return None

    def _close_trade(self):
        position = self._get_position()
        if position is None:
            return False
        tick = mt5.symbol_info_tick(self.principal_symbol)
        if tick is None:
            return False
        if self.algorithm == 'UP':
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.principal_symbol,
            "volume": position.volume,
            "type": order_type,
            "position": self.ticket,
            "price": price,
            "deviation": 20,
            "magic": self.magic,
            "comment": f"close_{self.algorithm}_{self.mercado}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(),
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  ✅ CLOSE {self.algorithm} {self.principal_symbol}/{self.mercado} @ {price}")
            return True
        print(f"  ❌ Error cierre {self.principal_symbol}/{self.mercado}/{self.algorithm}: {result.retcode} - {result.comment}")
        return False

    def _reset_state(self):
        self.is_open = False
        self.ticket = None
        self.open_price_open = 0.0
        self.entry_red_open = []
        self.cierre = 0
        self.model_close_streak = 0
        self.time_comienzo = None

    # ------------------------------------------------------------------
    # Procesamiento de nueva vela (lógica idéntica a test_iteration)
    # ------------------------------------------------------------------

    def process(self, shared_indicators):
        if not self.apply_shared_indicators(shared_indicators):
            print(f"  ⚠️ [{self.principal_symbol}/{self.mercado}/{self.algorithm}] Error aplicando indicadores.")
            return

        hour = format(self.current_forming_hour, "05b")
        time_np = self.current_forming_time_np
        market_features = (
            self._get_market_features(self.prev_row)
            if self.prev_row is not None
            else np.zeros(len(EXTRA_FEATURE_COLUMNS), dtype=np.float32)
        )

        # ── POSICIÓN ABIERTA ──────────────────────────────────────────
        if self.is_open:
            self.cierre += 1

            # 1. ¿La cerró MT5 por SL/TP?
            closed_by_mt5, pips = self._check_closed_by_mt5()
            if closed_by_mt5:
                print(f"  [{self.principal_symbol}/{self.mercado}/{self.algorithm}] Cerrado por MT5: {pips:.1f} pips tras {self.cierre} barras")
                self._reset_state()
                return

            # 2. Max holding
            if self.cierre >= self.max_holding:
                if self._close_trade():
                    print(f"  [{self.principal_symbol}/{self.mercado}/{self.algorithm}] Cerrado por max_holding")
                    self._reset_state()
                return

            # 3. Cierre por modelo neuronal (con confirmación de barras)
            if self.cierre >= self.min_model_holding:
                model_signal = False
                for nodo in self.nodos_close:
                    key = f'{self.principal_symbol}_{self.principal_symbol}_{nodo["file"].split("_")[0]}'
                    if key not in self.indicators:
                        continue
                    df_struct = self.indicators[key]
                    pos = df_struct['index_values'].searchsorted(time_np)
                    if pos == 0:
                        continue
                    if self._cumple_condiciones_fast(df_struct, pos - 1, nodo['conditions']):
                        nodo_close = self.maping_close.get(nodo['key'])
                        if nodo_close is None:
                            continue
                        for entry in self.entry_red_open:
                            prob = predict_from_inputs(self.nn, entry, nodo_close, hour, market_features)
                            if prob > self.close_threshold:
                                model_signal = True
                                break
                        if model_signal:
                            break

                self.model_close_streak = self.model_close_streak + 1 if model_signal else 0

                if self.model_close_streak >= self.close_confirmation_bars:
                    if self._close_trade():
                        print(f"  [{self.principal_symbol}/{self.mercado}/{self.algorithm}] Cerrado por modelo tras {self.cierre} barras")
                        self._reset_state()

        # ── SIN POSICIÓN ──────────────────────────────────────────────
        else:
            self.cierre = 0
            self.model_close_streak = 0

            # Fuera de horario de mercado
            if not self.horas_mercado[self.current_forming_hour]:
                return

            # Evento de entrada (sobre última vela cerrada = current_row)
            if self.current_row is None or not has_entry_event(self.current_row, self.algorithm):
                return

            open_nodes = self._resolve_entry_open_nodes()
            if not open_nodes:
                return

            self.current_stop_loss, self.current_take_profit = self._get_risk_limits()
            ticket = self._open_trade()
            if ticket is not None:
                self.is_open = True
                self.ticket = ticket
                self.open_price_open = self.current_open_price
                self.entry_red_open = open_nodes
                self.cierre = 0
                self.model_close_streak = 0
                self.time_comienzo = pd.Timestamp(self.current_forming_time_np.item())


class TradingServer:
    """
    Carga todos los algoritmos que pasan el filtro para cada símbolo
    y los ejecuta en tiempo real esperando nuevas velas.
    """

    def __init__(self):
        initialize_mt5()
        with open('config/general_config.json', 'r', encoding='utf-8') as f:
            self.general_config = json.load(f)

        self.timeframe = get_timeframes()['timeframes'][self.general_config['timeframe']]
        self.engines: list = []
        self._last_candle_time = None
        self._running = False
        self._build_engines()

    def _build_engines(self):
        print("Cargando estrategias activas...\n")
        list_principal_symbols = self.general_config['list_principal_symbols']
        list_mercados = ['Asia', 'Europa', 'America']
        list_algorithms = ['UP', 'DOWN']

        for principal_symbol in list_principal_symbols:
            for mercado in list_mercados:
                for algorithm in list_algorithms:
                    score_path = (
                        f'output/{principal_symbol}/data_for_neuronal/'
                        f'best_score/score_{mercado}_{algorithm}.json'
                    )
                    if not os.path.exists(score_path):
                        continue
                    with open(score_path) as f:
                        score_data = json.load(f)
                    metrics = score_data.get('metrics', {})
                    if not should_backtest_strategy(metrics):
                        print(f"  ⏭  {principal_symbol}/{mercado}/{algorithm} no pasa el filtro")
                        continue
                    try:
                        engine = TradingEngine(principal_symbol, mercado, algorithm, self.general_config)
                        self.engines.append(engine)
                        print(f"  ✅ {principal_symbol}/{mercado}/{algorithm} cargado")
                    except Exception as e:
                        print(f"  ❌ {principal_symbol}/{mercado}/{algorithm} error al cargar: {e}")

        print(f"\n{len(self.engines)} estrategias activas.\n")

    def _fetch_shared_indicators(self):
        """
        Calcula indicadores UNA SOLA VEZ por símbolo único presente en los engines.
        Devuelve shared_indicators = {symbol: {file_key: df_struct, '__enriched__': df}}.
        """
        _t0 = time_module.perf_counter()

        # Recopilar todos los símbolos únicos que necesitan los engines
        all_symbols: set = set()
        list_files: list = []
        for engine in self.engines:
            all_symbols.update(engine.list_symbols)
            if not list_files:
                list_files = engine.list_files  # todos usan los mismos archivos

        shared_indicators: dict = {}
        for symbol in all_symbols:
            rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, CANDLES_BUFFER)
            if rates is None or len(rates) < 30:
                print(f"  ⚠️ Sin datos para {symbol}")
                continue
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            symbol_cache: dict = {}
            for file in list_files:
                file_key = file.split('.')[0]
                try:
                    indicator = generate_files(file, df)
                    symbol_cache[file_key] = self.engines[0]._build_combined(indicator)
                except Exception as e:
                    print(f"  ⚠️ {symbol}/{file}: {e}")

            # Guardar también el DataFrame enriquecido para current_row/prev_row
            # Solo lo necesitan los principal_symbols, pero calcularlo siempre es barato
            enriched = self.engines[0]._enrich(df)
            symbol_cache['__enriched__'] = enriched

            shared_indicators[symbol] = symbol_cache

        elapsed = time_module.perf_counter() - _t0
        print(f"  ⏱  Indicadores calculados ({len(all_symbols)} símbolos únicos) en {elapsed:.2f}s")
        return shared_indicators

    def stop(self):
        self._running = False

    def _wait_for_new_candle(self):
        """Bloquea hasta que se forma una nueva vela (timeframe compartido)."""
        if not self.engines:
            time_module.sleep(60)
            return

        ref_symbol = self.engines[0].principal_symbol
        # Inicializar referencia si es la primera vez
        if self._last_candle_time is None:
            rates = mt5.copy_rates_from_pos(ref_symbol, self.timeframe, 0, 2)
            if rates is not None:
                self._last_candle_time = rates[-2]['time']  # última vela cerrada

        while self._running:
            time_module.sleep(5)
            rates = mt5.copy_rates_from_pos(ref_symbol, self.timeframe, 0, 2)
            if rates is None:
                continue
            new_time = rates[-2]['time']  # última vela cerrada
            if new_time != self._last_candle_time:
                self._last_candle_time = new_time
                return

    def run(self):
        from datetime import datetime as _dt
        self._running = True
        print("TradingServer activo. Esperando nueva vela...\n")
        while self._running:
            self._wait_for_new_candle()
            t_start = time_module.perf_counter()
            print(f"[{_dt.now().strftime('%Y-%m-%d %H:%M:%S')}] Nueva vela. Procesando {len(self.engines)} engines...")
            shared_indicators = self._fetch_shared_indicators()
            for engine in self.engines:
                try:
                    engine.process(shared_indicators)
                except Exception as e:
                    print(f"  ❌ Error en {engine.principal_symbol}/{engine.mercado}/{engine.algorithm}: {e}")
            total = time_module.perf_counter() - t_start
            print(f"  ⏱  Total procesado en {total:.2f}s")


if __name__ == "__main__":
    server = TradingServer()
    server.run()
