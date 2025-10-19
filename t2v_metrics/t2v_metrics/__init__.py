import os
import sys

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models/clipscore_models/InternVideo2/multi_modality/",
    )
)

from .constants import HF_CACHE_DIR
from .qascore import QAScore, list_all_qascore_models
from .vqascore import VQAScore, list_all_vqascore_models


def list_all_models():
    return list_all_vqascore_models()


def get_score_model(model="internvl2.5-8b", device="cuda", cache_dir=HF_CACHE_DIR, **kwargs):
    if model in list_all_vqascore_models():
        return VQAScore(model, device=device, cache_dir=cache_dir, **kwargs)
    elif model in list_all_qascore_models():
        return QAScore(model, device=device, cache_dir=cache_dir, **kwargs)
    else:
        raise NotImplementedError()
