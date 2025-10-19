from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import HF_TOKEN

LLAMA_33_MODELS = {
    "llama-3.3-70b-instruct": {
        "model": {
            "path": "meta-llama/Llama-3.3-70B-Instruct",
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        },
    }
}


class LLaMA33Model:
    def __init__(self, model_name="llama-3.3-70b-instruct", device="cuda", cache_dir=None):
        assert model_name in LLAMA_33_MODELS, f"Model {model_name} not found in LLAMA_33_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = LLAMA_33_MODELS[model_name]
        self.load_model()

    def load_model(self):
        model_config = self.model_info["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["path"],
            padding_side="left",
            token=HF_TOKEN,
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["path"],
            torch_dtype=model_config["torch_dtype"],
            device_map=model_config["device_map"],
            token=HF_TOKEN,
        )

    def load_images(self, image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device."""
        raise NotImplementedError

    def forward(
        self,
        texts: List[str],
        system_prompt: str = None,
        question_template: str = 'Does this image show "{}"? Answer the question with Yes or No',
        answer_template: str = "Yes",
    ) -> torch.Tensor:
        questions = [question_template.format(**fields) for fields in texts]
        answers = [answer_template.format(text) for text in texts]

        messages = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
            for question in questions
        ]
        texts = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(texts, padding="longest", return_tensors="pt")
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=1, output_scores=True, return_dict_in_generate=True
            )

        scores = outputs.scores[0]
        probs = torch.nn.functional.softmax(scores, dim=-1)
        yes_token_ids = torch.tensor(
            [self.tokenizer.encode(answer)[1] for answer in answers], device=probs.device
        )
        yes_token_ids = yes_token_ids.unsqueeze(1)
        lm_probs = probs.gather(dim=1, index=yes_token_ids)

        return torch.tensor(lm_probs)

    def generate(
        self, system_prompt: str, texts: List[str], max_new_tokens: int = 256
    ) -> List[str]:
        questions = texts

        generated_outputs = []
        for question in questions:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]

            with torch.no_grad():
                outputs = self.pipeline(
                    messages,
                    max_new_tokens=max_new_tokens,
                )

                text = outputs[0]["generated_text"][-1]
                generated_outputs.append(text.strip())

        return generated_outputs
