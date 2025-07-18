import contextlib

import clip
import torch
from torch import nn
from torchvision import transforms


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticEvaluatorTorch:
    def __init__(self, model_path, clip_model_path, device):
        self.model_path = model_path
        self.clip_model_path = clip_model_path
        self.device = device
        self.predictor, self.clip_model, self.preprocessor = self.load()

    def load(self):
        """Loads the aesthetic predictor model and CLIP model."""
        state_dict = torch.load(
            self.model_path, weights_only=True, map_location=self.device)

        # CLIP embedding dim is 768 for CLIP ViT L 14
        predictor = AestheticPredictor(768).half()
        predictor.load_state_dict(state_dict)
        predictor.to(self.device)
        predictor.eval()
        clip_model, _ = clip.load(self.clip_model_path, device=self.device)
        preprocessor = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                # transforms.Lambda(lambda x: x.clamp_(0, 1)),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        return predictor, clip_model, preprocessor

    def score(self, image: torch.Tensor, no_grad: bool = False) -> float:
        """Predicts the CLIP aesthetic score of an image."""
        if image.ndim != 4:
            raise ValueError(
                f"image must be 4 channels (shape: {image.shape})")

        with torch.no_grad() if no_grad else contextlib.nullcontext():
            image = self.preprocessor(image)
            image_features = self.clip_model.encode_image(image)
            # l2 normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            score_tensor = self.predictor(image_features)

        return score_tensor / 10.0  # scale to [0, 1]
