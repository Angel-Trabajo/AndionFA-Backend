from pydantic import BaseModel
from typing import List

# Modelo que representa los datos que envía el frontend
class IndicatorsRequest(BaseModel):
    symbol: str
    timeframe: str
    start: str
    end: str
    indicators_files: List[str]
    
class ConfigRequest(BaseModel):
    dateStart: str
    dateEnd: str
    maxDepth: str
    SimilarityMax:float
    NTotal: int
    MinOperationsIS: int
    MinOperationsOS: int
    NumMaxOperations: int
    MinSuccessRate: float
    MaxSuccessRate: float
    ProgressiveVariation: float
    
class NodeIndRequest(BaseModel):
    symbol: str
    timeframe: str
    
class TestRequest(BaseModel):
    symbol: str
    list_id: List[int]
    
class Symbol(BaseModel):
    value: str
    label: str

class ConfigCrossingRequest(BaseModel):
    max_sri: float
    min_sri: float
    n_totales: int
    min_operaciones: float
    max_depth: str
    maximo_weka_tree: int
    intentos: int
    aumento_arboles: int
    aumento_profundidad: int
    list_symbols: List[Symbol]
    principal_symbol: str
    timeframe: str
    por_direccion: bool
    list_symbols_inversos: List[Symbol]
    
class TestRequestCros(BaseModel):
    prin_symbol: str
    symbol: str
    list_id: List[int]
    future: str