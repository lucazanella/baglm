import copy
import re
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from PIL import Image
from torchvision.transforms.functional import normalize as tv_normalize
from transformers.image_utils import ChannelDimension

from utils.text_utils import format_questions

from .vqa_model import VQAScoreModel

LLAVA_OV_MODELS = {
    "llava-onevision-qwen2-7b-si": {
        "tokenizer": {
            "path": "lmms-lab/llava-onevision-qwen2-7b-si",
        },
        "model": {
            "path": "lmms-lab/llava-onevision-qwen2-7b-si",
            "conversation": "qwen_1_5",
            "image_aspect_ratio": "pad",
        },
    },
    "llava-onevision-qwen2-7b-ov": {
        "tokenizer": {
            "path": "lmms-lab/llava-onevision-qwen2-7b-ov",
        },
        "model": {
            "path": "lmms-lab/llava-onevision-qwen2-7b-ov",
            "conversation": "qwen_1_5",
            "image_aspect_ratio": "pad",
        },
    },
}


class SigLipTensorProcessor:
    def __init__(
        self,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        size: Tuple[int, int] = (384, 384),
        rescale_factor: float = 1 / 255,
        data_format: ChannelDimension = ChannelDimension.FIRST,
    ):
        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.rescale_factor = rescale_factor
        self.data_format = data_format

    def __call__(self, tensor, **kwargs):
        """
        Args:
            tensor: (C, H, W) or (N, C, H, W) PyTorch tensor with dtype uint8 or float
            return_tensors: must be "pt"
        """
        if isinstance(tensor, list):
            tensor = torch.stack(tensor)

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # (1, C, H, W)

        assert tensor.dim() == 4 and tensor.shape[1] in [1, 3], "Expected shape (N, C, H, W)"

        tensor = tensor.float()
        tensor = tensor * self.rescale_factor

        # Resize to target size using bilinear interpolation
        tensor = F.interpolate(tensor, size=self.size, mode="bilinear", align_corners=False)

        # Normalize using torchvision's normalize (per channel)
        for t, m, s in zip(tensor, self.image_mean, self.image_std):
            tv_normalize(t, mean=[m], std=[s])

        return tensor


class LLaVAOneVisionModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True

    def __init__(self, model_name="llava-onevision-qwen2-7b-ov", device="cuda", cache_dir=None):
        assert model_name in LLAVA_OV_MODELS, f"Model {model_name} not found in LLAVA_OV_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = LLAVA_OV_MODELS[model_name]
        self.conversational_style = self.model_info["model"]["conversation"]
        self.load_model()

    def get_preprocessor(self):
        return SigLipTensorProcessor()

    def load_model(self):
        model_path = self.model_info["model"]["path"]
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            model_path, None, "llava_qwen", device_map="auto", attn_implementation="sdpa"
        )
        self.model.eval()

    def load_images(
        self, paths: List[str], num_frames: int = 16
    ) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
        processed_data = []
        for path in paths:
            if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):  # Video file
                video_frames = self.load_video(path, num_frames)
                frames = (
                    self.processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
                    .half()
                    .to(self.device)
                )
                processed_data.append(frames)
            elif path.lower().endswith(".npy"):  # NumPy file
                np_array = np.load(path)
                if np_array.ndim == 3:  # Single image
                    image = Image.fromarray(np_array.astype("uint8"), "RGB")
                    image_tensor = process_images([image], self.processor, self.model.config)
                    image_tensor = [
                        _image.to(dtype=torch.float16, device=self.device)
                        for _image in image_tensor
                    ]
                    processed_data.append(image_tensor[0])
                elif np_array.ndim == 4:  # Multiple frames
                    frames = [Image.fromarray(frame.astype("uint8"), "RGB") for frame in np_array]
                    frames_tensor = (
                        self.processor.preprocess(frames, return_tensors="pt")["pixel_values"]
                        .half()
                        .to(self.device)
                    )
                    processed_data.append(frames_tensor)
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:  # Regular image file
                image = Image.open(path).convert("RGB")
                image_tensor = process_images([image], self.processor, self.model.config)
                image_tensor = [
                    _image.to(dtype=torch.float16, device=self.device) for _image in image_tensor
                ]
                processed_data.append(image_tensor[0])
        return processed_data

    def load_video(self, video_path, max_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).numpy()
        return spare_frames

    def forward(
        self,
        videos,
        texts: List[str],
        question_template: str,  # "Does this image show \"{}\"? Answer the question with Yes or No",
        answer_template: str = "Yes",
    ) -> torch.Tensor:
        questions = [question_template.format(**fields) for fields in texts]
        answers = [answer_template.format(text) for text in texts]
        self.tokenizer.padding_side = "left"

        questions = [self.format_question(question) for question in questions]

        processed_data = [frames.half().to(self.device) for frames in videos]

        prompts = [qs for qs in questions]

        input_ids = [
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            for prompt in prompts
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.device)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.float16)
        image_sizes = [data.shape[2:] for data in processed_data]
        modalities = ["video" for _ in processed_data]
        outputs = self.model.generate(
            input_ids,
            images=processed_data,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=1,
            modalities=modalities,
            attention_mask=attention_mask,
            output_scores=True,
            return_dict_in_generate=True,
        )
        # text_outputs = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        scores = outputs.scores[0]
        probs = torch.nn.functional.softmax(scores, dim=-1)
        yes_token_ids = torch.tensor(
            [self.tokenizer.encode(answer)[0] for answer in answers], device=probs.device
        )
        yes_token_ids = yes_token_ids.unsqueeze(1)
        lm_probs = probs.gather(dim=1, index=yes_token_ids)

        return lm_probs

    def forward_multi_choice(
        self,
        videos,
        texts: List[str],
        question_template: str,
        progress: bool = False,
    ) -> torch.Tensor:
        self.tokenizer.padding_side = "left"

        if progress:
            questions = [question_template.format(**fields) for fields in texts]
            answer_labels_per_example = [[str(n) for n in range(10)]] * len(questions)
        else:
            questions = format_questions(texts, self.tokenizer, question_template)
            answer_labels_per_example = [
                [re.match(r"^([A-Z]+)\.", line).group(1) for line in text["choices"].splitlines()]
                for text in texts
            ]

        questions = [self.format_question(question) for question in questions]

        processed_data = [frames.half().to(self.device) for frames in videos]

        prompts = [qs for qs in questions]

        input_ids = [
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            for prompt in prompts
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.device)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.float16)
        image_sizes = [data.shape[2:] for data in processed_data]
        modalities = ["video" for _ in processed_data]
        outputs = self.model.generate(
            input_ids,
            images=processed_data,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=1,
            modalities=modalities,
            attention_mask=attention_mask,
            output_scores=True,
            return_dict_in_generate=True,
        )
        # text_outputs = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        scores = outputs.scores[0]
        probs = torch.nn.functional.softmax(scores, dim=-1)

        token_ids_per_example = [
            torch.tensor(
                [self.tokenizer.encode(label)[0] for label in labels],
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
        self,
        images: List[str],
        texts: List[str],
        num_frames: int = 12,
        max_new_tokens: int = 256,
    ) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"
        texts = [self.format_question(text) for text in texts]
        processed_data = self.load_images(images, num_frames)
        generated_texts = []
        for data, prompt in zip(processed_data, texts):
            if isinstance(data, torch.Tensor) and data.dim() == 4:  # Video
                image_sizes = [data.shape[2:] for _ in range(data.shape[0])]
                modalities = ["video"]
            else:  # Image
                image_sizes = [data.shape[1:]]
                modalities = None

            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(self.device)
            )

            outputs = self.model.generate(
                input_ids,
                images=[data],
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                modalities=modalities,
            )

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(text.strip())

        return generated_texts

    def format_question(self, question):
        conv = copy.deepcopy(conv_templates[self.conversational_style])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
