from ...constants import HF_CACHE_DIR
from .internvl_model import INTERNVL2_MODELS, InternVL2Model
from .llavaov_model import LLAVA_OV_MODELS, LLaVAOneVisionModel
from .qwen2vl_model import QWEN2_VL_MODELS, Qwen2VLModel

ALL_VQA_MODELS = [
    LLAVA_OV_MODELS,
    INTERNVL2_MODELS,
    QWEN2_VL_MODELS,
]


def list_all_vqascore_models():
    return [model for models in ALL_VQA_MODELS for model in models]


def get_vqascore_model(model_name, device="cuda", cache_dir=HF_CACHE_DIR, **kwargs):
    assert model_name in list_all_vqascore_models()
    if model_name in LLAVA_OV_MODELS:
        return LLaVAOneVisionModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in INTERNVL2_MODELS:
        return InternVL2Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in QWEN2_VL_MODELS:
        return Qwen2VLModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    else:
        raise NotImplementedError()
