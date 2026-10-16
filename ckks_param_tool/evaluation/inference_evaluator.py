from abc import ABC, abstractmethod
from ckks_param_tool.params import CKKSParams
from .metrics import InferenceEvaluationResult

'''
This file contains the abstract inference evaluation class.
'''

class InferenceEvaluator(ABC):

    @abstractmethod
    def evaluate(self, params: CKKSParams) -> InferenceEvaluationResult:
        pass

