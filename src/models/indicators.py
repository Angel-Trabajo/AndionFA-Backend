from pydantic import BaseModel
from typing import List


class ConfigRequest(BaseModel):
    symbols: List[str]
    timeFrame: str
    dateStart: str
    dateEnd: str
    SimilarityMax:float
    NTotal: int
    MinOperations: int
    MinOperationsIS: int
    MinOperationsOS: int
    NumMaxOperations: int
    MinSuccessRate: float
    MaxSuccessRate: float
    ProgressiveVariation: float
    MinOpenSymbolConfirmations: int = 4
    robust_trade_penalty_center: int = 25
    stop_loss: int = 20
    take_profit: int = 150
    use_proces: int = 12


class ExecuteRequest(BaseModel):
    reset_db: bool = True

