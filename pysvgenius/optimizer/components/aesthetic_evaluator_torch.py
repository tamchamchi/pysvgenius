import contextlib

import clip
import torch
from torch import nn
from torchvision import transforms

import os
from pathlib import Path
from typing import Optional


from pysvgenius.common import registry


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
    # Model URLs for auto-download
    AESTHETIC_MODEL_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
    AESTHETIC_MODEL_NAME = "sac+logos+ava1-l14-linearMSE.pth"

    def __init__(
        self,
        model_path: Optional[str] = None,
        clip_model_path: Optional[str] = None,
        device: str = "cuda:0",
    ):
        """
        Initialize AestheticRanker.

        Args:
            model_path (str): Path to the trained aesthetic predictor model
            clip_model_path (str): CLIP model name (e.g., "ViT-L/14")
            device (str): Device to run the models on (cuda:0 or cpu)
        """
        # Set default paths
                # Get model directory from registry with fallback
        try:
            model_dir = Path(registry.get_path("model_dir"))
        except (KeyError, AttributeError):
            model_dir = Path("models")
            print(f"⚠ Warning: Using default model directory: {model_dir}")

        # Ensure model directory exists
        if not model_dir.exists():
            print(f"Creating model directory: {model_dir}")
            model_dir.mkdir(parents=True, exist_ok=True)

        # Set default paths
        self.model_path = model_path or str(
            model_dir / self.AESTHETIC_MODEL_NAME)
        self.clip_model_path = clip_model_path or "ViT-L/14"  # CLIP model name
        self.device = device

        self.predictor, self.clip_model, self.preprocessor = self._load()

    def _load(self):
        """
        Loads the aesthetic predictor and the CLIP model (ViT-L/14).

        Returns:
            tuple: (predictor, clip_model, preprocessor)
                - predictor (nn.Module): Trained aesthetic predictor model
                - clip_model (CLIPModel): Pretrained CLIP model
                - preprocessor (Callable): Image preprocessor function

        Raises:
            FileNotFoundError: If the aesthetic model file is missing
            RuntimeError: If loading models fails
        """
        # 1. Ensure aesthetic model exists
        if not os.path.exists(self.model_path):
            error_msg = (
                f"✗ Aesthetic model not found: {self.model_path}\n"
                f"💡 Download it manually with:\n"
                f"    wget {self.AESTHETIC_MODEL_URL} -O {self.model_path}"
            )
            raise FileNotFoundError(error_msg)

        try:
            # 2. Load the trained aesthetic predictor
            state_dict = torch.load(self.model_path, weights_only=True, map_location=self.device)
            predictor = AestheticPredictor(768).half()
            predictor.load_state_dict(state_dict)
            predictor.to(self.device)
            predictor.eval()

            # 3. Load CLIP model
            clip_download_root = None
            if hasattr(registry, 'get_path'):
                try:
                    clip_download_root = str(Path(registry.get_path("model_dir")) / "clip")
                    print(f"✓ CLIP download root: {clip_download_root}")
                except Exception as e:
                    print(f"⚠ Could not determine CLIP download root: {e}")

            clip_model, _ = clip.load(
                self.clip_model_path,
                device=self.device,
                download_root=clip_download_root
            )

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

        except Exception as e:
            raise RuntimeError(f"✗ Failed to load models: {e}") from e

    # def load(self):
    #     """Loads the aesthetic predictor model and CLIP model."""
    #     state_dict = torch.load(
    #         self.model_path, weights_only=True, map_location=self.device)

    #     # CLIP embedding dim is 768 for CLIP ViT L 14
    #     predictor = AestheticPredictor(768).half()
    #     predictor.load_state_dict(state_dict)
    #     predictor.to(self.device)
    #     predictor.eval()
    #     clip_model, _ = clip.load(self.clip_model_path, device=self.device)
    #     preprocessor = transforms.Compose(
    #         [
    #             transforms.Resize(
    #                 224, interpolation=transforms.InterpolationMode.BICUBIC),
    #             transforms.CenterCrop(224),
    #             # transforms.Lambda(lambda x: x.clamp_(0, 1)),
    #             transforms.Normalize(
    #                 (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    #         ]
    #     )

    #     return predictor, clip_model, preprocessor

    def score(self, image: torch.Tensor, no_grad: bool = False) -> float:
        """Predicts the CLIP aesthetic score of an image."""
        if image.ndim != 4:
            raise ValueError(
                f"image must be 4 channels (shape: {image.shape})")

        with torch.no_grad() if no_grad else contextlib.nullcontext():
            image = self.preprocessor(image)
            image_features = self.clip_model.encode_image(image)
            # l2 normalize
            image_features = image_features / \
                image_features.norm(dim=-1, keepdim=True)

            score_tensor = self.predictor(image_features)

        return score_tensor / 10.0  # scale to [0, 1]

    @classmethod
    def from_config(cls, cfg):
        model_path = cfg.get("model_path")
        clip_model_path = cfg.get("clip_model_path", "ViT-L/14")
        device = cfg.get("device")
        return cls(model_path, clip_model_path, device)
