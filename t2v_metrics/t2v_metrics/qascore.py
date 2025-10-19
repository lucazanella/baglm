from typing import List

from .constants import HF_CACHE_DIR
from .models.qascore_models import get_qascore_model, list_all_qascore_models
from .score import Score


class QAScore(Score):
    def prepare_scoremodel(
        self, model="llama-3.3", device="cuda", cache_dir=HF_CACHE_DIR, **kwargs
    ):
        return get_qascore_model(model, device=device, cache_dir=cache_dir, **kwargs)

    def list_all_models(self) -> List[str]:
        return list_all_qascore_models()
