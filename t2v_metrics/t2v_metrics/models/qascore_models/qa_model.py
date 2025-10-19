from abc import abstractmethod
from typing import List

import torch

from ..model import ScoreModel


class QAScoreModel(ScoreModel):

    @abstractmethod
    def forward(
        self, texts: List[str], question_template: str, answer_template: str
    ) -> torch.Tensor:
        """Forward pass of the model to return n scores for n texts (in PyTorch Tensor)
        question_template: a string with optional {} to be replaced with the 'text'
        answer_template: a string with optional {} to be replaced with the 'text'
        """
        pass
