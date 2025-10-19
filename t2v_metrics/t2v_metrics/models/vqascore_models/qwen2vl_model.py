import math
import os
import re
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

from utils.text_utils import format_questions

from .vqa_model import VQAScoreModel

QWEN2_VL_MODELS = {
    # Qwen2_VL
    "qwen2-vl-2b": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-2B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2-VL-2B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2-vl-7b": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2-vl-72b": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-72B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2-VL-72B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    # Qwen2.5_VL:
    "qwen2.5-vl-3b": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-3B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2.5-VL-3B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2.5-vl-7b": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-7B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2.5-VL-7B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2.5-vl-72b": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-72B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2.5-VL-72B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    # Winoground Finetuning
    "qwen2-vl-1": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
        },
        "model": {
            "path": "../LLaMA-Factory/saves/qwen2_vl-7b/lora/wino_lora_3epochs",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2-vl-2": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
        },
        "model": {
            "path": "../LLaMA-Factory/saves/qwen2_vl-7b/lora/wino_lora_5epochs",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2-vl-3": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
        },
        "model": {
            "path": "../LLaMA-Factory/saves/qwen2_vl-7b/lora/wino_dpo_3epochs",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2-vl-4": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
        },
        "model": {
            "path": "../LLaMA-Factory/saves/qwen2_vl-7b/lora/wino_dpo_5epochs",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    # GenAI-Bench Finetuning:
    "qwen2-vl-5": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
        },
        "model": {
            "path": "../LLaMA-Factory/saves/qwen2_vl-7b/lora/genai_lora_3epochs",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2-vl-6": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
        },
        "model": {
            "path": "../LLaMA-Factory/saves/qwen2_vl-7b/lora/genai_lora_5epochs",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2-vl-7": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
        },
        "model": {
            "path": "../LLaMA-Factory/saves/qwen2_vl-7b/lora/genai_dpo_3epochs",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2-vl-8": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
        },
        "model": {
            "path": "../LLaMA-Factory/saves/qwen2_vl-7b/lora/genai_dpo_5epochs",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    # NaturalBench Finetuning:
    "qwen2-vl-9": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
        },
        "model": {
            "path": "../LLaMA-Factory/saves/qwen2_vl-7b/lora/nb_lora_3epochs",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2-vl-10": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
        },
        "model": {
            "path": "../LLaMA-Factory/saves/qwen2_vl-7b/lora/nb_lora_5epochs",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2-vl-11": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
        },
        "model": {
            "path": "../LLaMA-Factory/saves/qwen2_vl-7b/lora/nb_dpo_3epochs",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2-vl-12": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
        },
        "model": {
            "path": "../LLaMA-Factory/saves/qwen2_vl-7b/lora/nb_dpo_5epochs",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    # Camera Motion Weights:
    "qwen2.5-vl-cam2500": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-7B-Instruct",
        },
        "model": {
            "path": "../LLaMA-Factory/saves/qwen2.5_vl-7b/lora/cam_motion_sft_2500",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2.5-vl-cam10000": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-7B-Instruct",
        },
        "model": {
            "path": "../LLaMA-Factory/models/qwen2.5-vl-cam10000",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2.5-vl-cam15000": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-7B-Instruct",
        },
        "model": {
            "path": "/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-cam15000",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2.5-vl-balanced": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-7B-Instruct",
        },
        "model": {
            "path": "/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-balanced",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
    "qwen2.5-vl-balanced2": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-7B-Instruct",
        },
        "model": {
            "path": "/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-balanced2",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    },
}

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9)))


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by
    'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


class ModelPreprocessor:
    def __init__(
        self, model, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False
    ):
        self.model = model
        self.image_factor = image_factor
        self.return_video_sample_fps = return_video_sample_fps

    def __call__(self, image, **kwargs):
        _, height, width = image.shape

        nframes = kwargs.get("nframes")

        min_pixels = VIDEO_MIN_PIXELS
        total_pixels = VIDEO_TOTAL_PIXELS
        max_pixels = max(
            min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05)
        )
        max_pixels_supposed = 360 * 420
        # if max_pixels_supposed > max_pixels:
        #     logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image = transforms.functional.resize(
            image,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        # if self.return_video_sample_fps:
        #     return video, sample_fps
        return image.unsqueeze(dim=0)


class Qwen2VLModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True

    def __init__(self, model_name="qwen2-vl-7b", device="cuda", cache_dir=None):
        assert model_name in QWEN2_VL_MODELS, f"Model {model_name} not found in QWEN2_VL_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = QWEN2_VL_MODELS[model_name]
        self.load_model()

    def get_preprocessor(self):
        return ModelPreprocessor(self)

    def load_model(self):
        model_path = self.model_info["model"]["path"]
        if "2.5" in model_path:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.model_info["model"]["torch_dtype"],
                attn_implementation=self.model_info["model"]["attn_implementation"],
                device_map="auto",
            )

        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.model_info["model"]["torch_dtype"],
                attn_implementation=self.model_info["model"]["attn_implementation"],
                device_map="auto",
            )
        self.processor = AutoProcessor.from_pretrained(self.model_info["tokenizer"]["path"])
        self.model.eval()

        self.device = next(
            self.model.parameters()
        ).device  # If there are multiple GPUs put the model on the first parameters GPU

    def load_images(
        self, paths: List[str], num_frames: int = 16
    ) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
        processed_data = []
        fps = self.model_info.get("fps", 2.0)
        for path in paths:
            if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):  # Video file path
                # video_frames = self.load_video(path, num_frames)
                processed_data.append(
                    {"type": "video", "video": path, "max_pixels": 360 * 420, "fps": fps}
                )
            elif path.lower().endswith(".npy"):  # NumPy file
                np_array = np.load(path)
                if np_array.ndim == 3:  # Single image
                    image = Image.fromarray(np_array.astype("uint8"), "RGB")
                    processed_data.append({"type": "image", "image": image})
                elif np_array.ndim == 4:  # Multiple frames
                    frames = [Image.fromarray(frame.astype("uint8"), "RGB") for frame in np_array]
                    processed_data.append({"type": "video", "video": frames})
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:  # Regular image file
                image = Image.open(path).convert("RGB")
                processed_data.append({"type": "image", "image": image})
        return processed_data

    def forward(
        self,
        videos,
        texts: List[str],
        question_template: str,  # "Does this image show \"{}\"? Answer the question with Yes or No",
        answer_template: str = "Yes",
    ) -> torch.Tensor:
        questions = [question_template.format(**fields) for fields in texts]
        answers = [answer_template.format(text) for text in texts]
        self.processor.tokenizer.padding_side = "left"

        # Combine messages for batch processing
        messages = [
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
            for question in questions
        ]
        print(f"Messages {messages}")

        # Preparation for batch inference
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = None, videos
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # generated_ids_trimmed = [
        #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs[0])
        # ]
        # output_texts = self.processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )

        scores = outputs.scores[0]
        probs = torch.nn.functional.softmax(scores, dim=-1)
        yes_token_ids = torch.tensor(
            [self.processor.tokenizer.encode(answer)[0] for answer in answers], device=probs.device
        )
        yes_token_ids = yes_token_ids.unsqueeze(1)
        lm_probs = probs.gather(dim=1, index=yes_token_ids)

        return torch.tensor(lm_probs)

    def forward_multi_choice(
        self,
        videos,
        texts: List[str],
        question_template: str,
        progress: bool = False,
    ) -> torch.Tensor:
        if progress:
            questions = [question_template.format(**fields) for fields in texts]
            answer_labels_per_example = [[str(n) for n in range(10)]] * len(questions)
        else:
            questions = format_questions(texts, self.processor.tokenizer, question_template)
            answer_labels_per_example = [
                [re.match(r"^([A-Z]+)\.", line).group(1) for line in text["choices"].splitlines()]
                for text in texts
            ]
        self.processor.tokenizer.padding_side = "left"

        # Combine messages for batch processing
        messages = [
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
            for question in questions
        ]

        # Preparation for batch inference
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = None, videos
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # generated_ids_trimmed = [
        #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs[0])
        # ]
        # output_texts = self.processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )

        scores = outputs.scores[0]
        probs = torch.nn.functional.softmax(scores, dim=-1)

        token_ids_per_example = [
            torch.tensor(
                [self.processor.tokenizer.encode(label)[0] for label in labels],
                device=probs.device,
            )
            for labels in answer_labels_per_example
        ]
        lm_probs = torch.stack(
            [
                prob.gather(dim=0, index=token_ids)  # gather from 1D prob vector
                for prob, token_ids in zip(probs, token_ids_per_example)
            ]
        )  # shape: (batch_size, num_choices)

        return lm_probs

    def generate(
        self, images: List[str], texts: List[str], num_frames: int = 16, max_new_tokens: int = 256
    ) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"

        processed_data = self.load_images(images, num_frames)

        generated_texts = []
        for data, text in zip(processed_data, texts):
            messages = [{"role": "user", "content": [data, {"type": "text", "text": text}]}]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                generated_texts.append(text)

        return generated_texts
