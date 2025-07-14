from .base import ISVGRanker
import os
import logging
from typing import Optional

import clip
import torch
import torch.nn as nn
from PIL import Image
from src import MODEL_DIR
from src.utils.image_utils import process_svg_to_image
from src.utils.logger import get_library_logger


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

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize AestheticRanker.

        Parameters
        ----------
        logger : Optional[logging.Logger]
            Logger instance to use. If None, a default library logger will be created.
        """
        if logger is not None:
            self.logger = logger
        else:
            self.logger = get_library_logger(
                f"{__name__}.{self.__class__.__name__}")

        self.logger.info("Initializing AestheticRanker")

        self.model_path = os.path.join(
            MODEL_DIR, "sac+logos+ava1-l14-linearMSE.pth")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"Using device: {self.device}")
        self.logger.debug(f"Model path: {self.model_path}")

        self.predictor, self.clip_model, self.preprocessor = self.load()
        self.logger.success("AestheticRanker initialization complete")

    def load(self):
        """
        Loads the aesthetic predictor and the CLIP model (ViT-L/14).

        Returns:
            predictor (nn.Module): Trained aesthetic predictor model.
            clip_model (CLIPModel): Pretrained CLIP model.
            preprocessor (Callable): Image preprocessor.
        """
        self.logger.info("Loading aesthetic predictor and CLIP model")
        try:
            self.logger.debug(f"Loading state dict from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)

            predictor = AestheticPredictor(768)  # CLIP ViT-L/14 → 768 dims
            predictor.load_state_dict(state_dict)
            predictor.to(self.device)
            predictor.eval()
            self.logger.debug(
                "Aesthetic predictor loaded and set to eval mode")

            self.logger.debug("Loading CLIP ViT-L/14 model")
            clip_model, preprocessor = clip.load(
                "ViT-L/14", device=self.device)
            self.logger.success("Models loaded successfully")

            return predictor, clip_model, preprocessor
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            raise RuntimeError(f"Failed to load models: {str(e)}") from e

    def score(self, image: Image.Image) -> float:
        """
        Computes an aesthetic score for a given image.

        Args:
            image (Image.Image): A PIL image.

        Returns:
            float: Aesthetic score in [0, 1].
        """
        self.logger.debug(
            f"Computing aesthetic score for image size: {image.size}")

        image = self.preprocessor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().detach().numpy()

            score = self.predictor(
                torch.from_numpy(image_features).to(self.device).float()
            )

        normalized_score = score.item() / 10.0  # Normalize to [0, 1]
        self.logger.debug(f"Computed aesthetic score: {normalized_score:.4f}")

        return normalized_score

    def process(self, svg_list: list[str] = [], prompt: str = None) -> list[int]:
        """
        Ranks a list of SVGs based on predicted aesthetic score.

        Args:
            prompt (str): Unused in this ranker but included for interface consistency.
            svg_list (list[str]): List of SVG strings.

        Returns:
            list[int]: Indices sorted by score in descending order
        """
        _ = prompt

        self.logger.info(
            f"Processing {len(svg_list)} SVG images for aesthetic ranking")

        results = []

        for i, svg in enumerate(svg_list, 1):
            self.logger.debug(f"Processing SVG {i}/{len(svg_list)}")
            try:
                image = process_svg_to_image(svg_code=svg)
                aesthetic_score = self.score(image)
                results.append(aesthetic_score)

                # Log progress every 5 SVGs
                if i % 5 == 0 or i == len(svg_list):
                    self.logger.info(f"Processed {i}/{len(svg_list)} SVGs")

            except Exception as e:
                self.logger.error(f"Failed to process SVG {i}: {str(e)}")
                # Add with score 0 to maintain consistency
                results.append({"svg": svg, "score": 0.0})

        self.logger.info(
            f"Completed aesthetic ranking for {len(results)} SVGs")
        avg_score = sum(score for score in results) / \
            len(results) if results else 0
        self.logger.info(f"Average aesthetic score: {avg_score:.4f}")

        # Sort indices by score in descending order
        sorted_indices = sorted(results, reverse=True)
        self.logger.debug(f"Sorted indices (desc): {sorted_indices}")

        # Log top scores
        if results:
            top_scores = [
                score for score in sorted_indices[: min(3, len(results))]]
            self.logger.info(f"Top scores: {top_scores}")

        # Log SUCCESS message when completed
        success_msg = f"Aesthetic ranking complete - Processed {len(results)} SVGs, avg score: {avg_score:.4f}"
        self.logger.success(success_msg)

        # Return only sorted indices array
        return sorted_indices


if __name__ == "__main__":
    import time
    from src.utils.logger import create_console_logger

    # Create console logger to see output
    logger = create_console_logger("AestheticRanker", logging.DEBUG)

    start = time.time()
    with open("/home/anhndt/pysvgenius/data/test/svg_1.svg", "r") as f:
        svg = f.read()
    with open("/home/anhndt/pysvgenius/data/test/test_svg.svg", "r") as f:
        raw_svg = f.read()

    aesthetic_ranker = AestheticRanker(logger=logger)
    sorted_indices = aesthetic_ranker.process([svg, raw_svg])

    print("Sorted indices (descending):", sorted_indices)
    print("Best SVG index:", sorted_indices[0] if sorted_indices else None)

    end = time.time()
    print(f"Total processing time: {end - start:.2f} seconds")
