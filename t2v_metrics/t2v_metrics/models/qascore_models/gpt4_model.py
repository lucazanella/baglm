from typing import List

import tiktoken
import torch
from openai import OpenAI

from .qa_model import QAScoreModel

default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "Yes"

GPT4_MODELS = {
    "gpt-4.1-mini": {},
}


class GPT4Model(QAScoreModel):
    def __init__(
        self,
        model_name="gpt-4.1-mini",
        device="cuda",
        cache_dir=None,
        api_key=None,
        top_logprobs=2,
    ):
        assert model_name in GPT4_MODELS
        assert api_key is not None, "Please provide an OpenAI API key"
        self.api_key = api_key
        self.top_logprobs = top_logprobs
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        tokenizer_name = "gpt-4o" if self.model_name == "gpt-4.1-mini" else self.model_name
        self.tokenizer = tiktoken.encoding_for_model(tokenizer_name)
        self.client = OpenAI(api_key=self.api_key)

    def load_images(self, image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device."""
        raise NotImplementedError

    def forward_single(self, question, answer):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": question}],
                logprobs=True,
                top_logprobs=self.top_logprobs,
            )

        except Exception as e:  # noqa: F841
            try:  # Second try
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": question}],
                    logprobs=True,
                    top_logprobs=self.top_logprobs,
                )
                print(f"completion {completion}")
            except Exception as e:  # Old Error Handling
                print(f"Failed question: {question} and answer: {answer}")
                print(f"Error: {str(e)}")
                return torch.Tensor([0.0])

        is_generated = False
        for top_logprob in completion.choices[0].logprobs.content[0].top_logprobs:
            if answer.lower() == "yes":
                if top_logprob.token == "Yes" or top_logprob.token == "yes":
                    is_generated = True
                    return torch.Tensor([top_logprob.logprob]).exp()
                elif top_logprob.token == "No" or top_logprob.token == "no":
                    is_generated = True
                    return 1 - torch.Tensor([top_logprob.logprob]).exp()
            else:
                if top_logprob.token == answer:
                    is_generated = True
                    return torch.Tensor([top_logprob.logprob]).exp()
        if not is_generated:
            print(
                f"Warning: '{answer}' not included in gpt4o log probs: question: {question} and answer: {answer}"
            )
            print(completion.choices[0].logprobs.content[0].top_logprobs)
            return torch.Tensor([0.0])

    def forward(
        self,
        texts: List[str],
        system_prompt: str = None,
        question_template: str = default_question_template,
        answer_template: str = default_answer_template,
    ) -> torch.Tensor:
        questions = [question_template.format(**fields) for fields in texts]
        answers = [answer_template.format(text) for text in texts]

        for ans in answers:
            ans_tokens = self.tokenizer.encode(ans)
            assert len(ans_tokens) == 1, "Currently only support single token answers"

        lm_prob = torch.zeros(len(questions))

        for idx, (question, answer) in enumerate(zip(questions, answers)):
            lm_prob[idx] = self.forward_single(question, answer)

        return lm_prob

    def generate_single(self, question, max_new_tokens):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": question}],
                max_tokens=max_new_tokens,
            )

        except Exception as e:  # noqa: F841
            try:  # Second try
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": question}],
                    max_tokens=256,
                )
            except Exception as e:
                print(f"Failed question: {question}")
                print(f"Error: {str(e)}")
                return ""

        return completion.choices[0].message.content

    def generate(self, texts: List[str], max_new_tokens: int = 256) -> List[str]:
        questions = texts

        generated_outputs = []

        for idx, question in enumerate(questions):
            generated_text = self.generate_single(question, max_new_tokens)
            generated_outputs.append(generated_text)

        return generated_outputs
