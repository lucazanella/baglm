from ...constants import HF_CACHE_DIR
from .gpt4_model import GPT4_MODELS, GPT4Model
from .llama33_model import LLAMA_33_MODELS, LLaMA33Model

ALL_QA_MODELS = [
    GPT4_MODELS,
    LLAMA_33_MODELS,
]


def list_all_qascore_models():
    return [model for models in ALL_QA_MODELS for model in models]


def get_qascore_model(model_name, device="cuda", cache_dir=HF_CACHE_DIR, **kwargs):
    assert model_name in list_all_qascore_models()
    if model_name in GPT4_MODELS:
        return GPT4Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in LLAMA_33_MODELS:
        return LLaMA33Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    else:
        raise NotImplementedError()
