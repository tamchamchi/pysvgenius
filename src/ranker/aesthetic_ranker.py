from base import ISVGRanker
import os

import clip
import torch
import torch.nn as nn
from PIL import Image

from config.config_dir import MODEL_DIR
from src.utils.image_utils import ImageProcessor
from src.utils.svg_utils import svg_to_png


class AestheticPredictor(nn.Module):
    """
    A feed-forward neural network for predicting the aesthetic score
    based on CLIP image embeddings.
    """

    def __init__(self, input_size: int):
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


class AestheticRanker(ISVGRanker):
    """
    Ranks a list of SVG images by converting them to PNG and scoring
    their visual quality using CLIP embeddings and a trained aesthetic predictor.
    """

    def __init__(self):
        self.model_path = os.path.join(
            MODEL_DIR, "sac+logos+ava1-l14-linearMSE.pth")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.predictor, self.clip_model, self.preprocessor = self.load()

    def load(self):
        """
        Loads the aesthetic predictor and the CLIP model (ViT-L/14).

        Returns:
            predictor (nn.Module): Trained aesthetic predictor model.
            clip_model (CLIPModel): Pretrained CLIP model.
            preprocessor (Callable): Image preprocessor.
        """
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)

            predictor = AestheticPredictor(768)  # CLIP ViT-L/14 → 768 dims
            predictor.load_state_dict(state_dict)
            predictor.to(self.device)
            predictor.eval()

            clip_model, preprocessor = clip.load(
                "ViT-L/14", device=self.device)

            return predictor, clip_model, preprocessor
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}") from e

    def score(self, image: Image.Image) -> float:
        """
        Computes an aesthetic score for a given image.

        Args:
            image (Image.Image): A PIL image.

        Returns:
            float: Aesthetic score in [0, 1].
        """
        image = self.preprocessor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().detach().numpy()

            score = self.predictor(torch.from_numpy(
                image_features).to(self.device).float())

        return score.item() / 10.0  # Normalize to [0, 1]

    def process(self, svg_list: list[str] = [], prompt: str = None) -> dict[str, float]:
        """
        Ranks a list of SVGs based on predicted aesthetic score.

        Args:
            prompt (str): Unused in this ranker but included for interface consistency.
            svg_list (list[str]): List of SVG strings.

        Returns:
            dict[str, float]: A mapping from SVG string to its aesthetic score.
        """
        _ = prompt

        results = []

        for svg in svg_list:
            image_processor = ImageProcessor(
                image=svg_to_png(svg), seed=42).apply()
            image = image_processor.image.copy()
            aesthetic_score = self.score(image)
            results.append({
                "svg": svg,
                "score": aesthetic_score
            })

        return results


if __name__ == "__main__":
    import time
    svg = ["""<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""]
    aesthetic_ranker = AestheticRanker()
    start = time.time()
    print(aesthetic_ranker.process(svg))
    end = time.time()
    print(end - start)
