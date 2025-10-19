import math
from abc import abstractmethod
from typing import List, TypedDict

import torch
import torch.nn as nn

from .constants import HF_CACHE_DIR


class ImageTextDict(TypedDict):
    images: List[str]
    texts: List[str]


class Score(nn.Module):

    def __init__(self, model: str, device: str = "cuda", cache_dir: str = HF_CACHE_DIR, **kwargs):
        """Initialize the ScoreModel."""
        super().__init__()
        assert model in self.list_all_models()
        self.device = device
        self.model = self.prepare_scoremodel(model, device, cache_dir, **kwargs)
        self.model_name = model

    @abstractmethod
    def prepare_scoremodel(self, model: str, device: str, cache_dir: str, **kwargs):
        """Prepare the ScoreModel."""
        pass

    @abstractmethod
    def list_all_models(self) -> List[str]:
        """List all available models."""
        pass

    def forward_vsg(self, batch, visual_batch_size: int = 1, **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s) and the text(s) If there are m
        images and n texts, return a m x n tensor."""
        media_type = "videos"

        all_visuals = batch[media_type]

        all_texts = [
            {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in text.items()}
            for text in batch["texts"]
        ]

        num_visuals, num_texts = len(all_visuals), len(all_texts) + 1  # +1 for none of the above
        scores = torch.zeros(num_visuals, num_texts, device="cpu")

        for visual_start in range(0, num_visuals, visual_batch_size):
            visual_end = min(visual_start + visual_batch_size, num_visuals)
            visual_batch = all_visuals[visual_start:visual_end]

            if len(visual_batch) == 0:
                continue

            step_texts = [text["step"] for text in all_texts]
            step_texts.append("None of the above.")
            text_batch = [{"goal": all_texts[0]["goal"], "step": step_texts}]

            expanded_visuals = [vis for vis in visual_batch for _ in range(len(text_batch))]
            expanded_texts = text_batch * len(visual_batch)

            score_batch = self.model.forward_multi_choice(
                expanded_visuals, expanded_texts, **kwargs
            )
            score_batch = score_batch.to("cpu")

            expected_shape = (visual_end - visual_start, len(step_texts))
            scores[visual_start:visual_end] = score_batch.view(expected_shape)

            del score_batch
            torch.cuda.empty_cache()

        return scores

    def forward_progress(
        self, batch, visual_batch_size: int = 1, text_batch_size: int = 64, **kwargs
    ) -> torch.Tensor:
        """Return the similarity score(s) between the image(s) and the text(s) If there are m
        images and n texts, return a m x n tensor."""
        media_type = "videos"

        all_visuals = batch[media_type]

        all_texts = [
            {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in text.items()}
            for text in batch["texts"]
        ]

        num_visuals, num_texts, num_choices = len(all_visuals), len(all_texts), 10
        scores = torch.zeros(num_visuals, num_texts, num_choices, device="cpu")

        for visual_start in range(0, num_visuals, visual_batch_size):
            visual_end = min(visual_start + visual_batch_size, num_visuals)
            visual_batch = all_visuals[visual_start:visual_end]

            if len(visual_batch) == 0:
                continue

            for text_start in range(0, num_texts, text_batch_size):
                text_end = min(text_start + text_batch_size, num_texts)
                text_batch = all_texts[text_start:text_end]

                if len(text_batch) == 0:
                    continue

                expanded_visuals = [vis for vis in visual_batch for _ in range(len(text_batch))]
                expanded_texts = text_batch * len(visual_batch)

                score_batch = self.model.forward_multi_choice(
                    expanded_visuals, expanded_texts, progress=True, **kwargs
                )
                score_batch = score_batch.to("cpu")

                expected_shape = (visual_end - visual_start, (text_end - text_start), num_choices)
                scores[visual_start:visual_end, text_start:text_end, :] = score_batch.view(
                    expected_shape
                )

                del score_batch
                torch.cuda.empty_cache()

        return scores

    def batch_forward_text(self, batch, batch_size: int = 64, **kwargs) -> torch.Tensor:

        num_texts = int(math.sqrt(len(batch)))
        assert num_texts * num_texts == len(batch)

        all_scores = []

        for batch_start in range(0, len(batch), batch_size):
            batch_end = min(batch_start + batch_size, len(batch))
            text_batch = batch[batch_start:batch_end]

            if len(text_batch) == 0:
                continue

            score_batch = self.model.forward(text_batch, **kwargs).to("cpu")
            all_scores.append(score_batch)

        scores = torch.cat(all_scores, dim=0)
        scores = scores.view(num_texts, num_texts)

        return scores
