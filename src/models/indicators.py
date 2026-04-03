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
    
