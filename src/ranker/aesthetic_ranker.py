from .base import ISVGRanker
import os
import logging
from typing import Optional

import clip
import torch
import torch.nn as nn
from PIL import Image
from src import MODEL_DIR
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

    def score(self, images) -> list[float]:
        """
        Computes aesthetic scores for given images (supports both single image and batch).

        Args:
            images: Can be either:
                - Single PIL Image
                - List of PIL Images
                - Batch tensor

        Returns:
            list[float]: List of aesthetic scores in [0, 1].
        """
        # Handle single image case
        if isinstance(images, Image.Image):
            images = [images]

        if not images:
            return []

        self.logger.debug(
            f"Computing aesthetic scores for batch of {len(images)} images")

        # Preprocess all images into a batch
        batch_tensors = []
        for img in images:
            preprocessed = self.preprocessor(img).unsqueeze(0)
            batch_tensors.append(preprocessed)

        # Concatenate into a single batch tensor
        batch = torch.cat(batch_tensors, dim=0).to(self.device)

        scores = []
        with torch.no_grad():
            # Process the entire batch through CLIP
            batch_features = self.clip_model.encode_image(batch)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)

            # Process through aesthetic predictor
            batch_scores = self.predictor(batch_features.float())

            # Normalize scores to [0, 1] and convert to list
            normalized_scores = (batch_scores / 10.0).squeeze().cpu().numpy()

            # Handle single item case (numpy scalar)
            if normalized_scores.ndim == 0:
                scores = [float(normalized_scores)]
            else:
                scores = normalized_scores.tolist()

        self.logger.debug(
            f"Computed {len(scores)} aesthetic scores: {[f'{s:.4f}' for s in scores]}")
        return scores

    def process(self, images: list[Image.Image], prompt: str = None, batch_size: int = 16) -> list[int]:
        """
        Ranks a list of images based on predicted aesthetic score using configurable batch processing.

        Args:
            images (list[Image.Image]): List of PIL Images to rank.
            prompt (str): Unused in this ranker but included for interface consistency.
            batch_size (int): Number of images to process in each batch. Default is 16.
                             Larger batch_size = faster processing but more GPU memory usage.
                             Smaller batch_size = slower processing but less GPU memory usage.

        Returns:
            list[int]: Indices sorted by score in descending order
        """
        _ = prompt

        if not images:
            self.logger.warning("No images provided for ranking")
            return []

        self.logger.info(
            f"Processing {len(images)} images with batch_size={batch_size}")

        all_scores = []

        # Process images in batches
        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            current_batch = images[batch_start:batch_end]

            self.logger.debug(f"Processing batch {batch_start//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size} "
                              f"(images {batch_start+1}-{batch_end})")

            try:
                # Process current batch
                batch_scores = self.score(current_batch)
                all_scores.extend(batch_scores)

                # Log progress for larger datasets
                if len(images) > batch_size:
                    self.logger.info(
                        f"Processed {batch_end}/{len(images)} images")

            except Exception as e:
                self.logger.error(
                    f"Failed to process batch {batch_start//batch_size + 1}: {str(e)}")
                # Add zeros for failed batch to maintain index consistency
                all_scores.extend([0.0] * len(current_batch))

        if len(all_scores) != len(images):
            self.logger.error(
                f"Score count mismatch: {len(all_scores)} scores for {len(images)} images")
            return []

        # Sort indices by score in descending order
        indexed_scores = list(enumerate(all_scores))
        sorted_pairs = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, score in sorted_pairs]

        self.logger.debug(
            f"Sorted indices (desc): {sorted_indices[:10]}{'...' if len(sorted_indices) > 10 else ''}")

        # Log top scores
        if all_scores:
            top_count = min(5, len(all_scores))
            top_scores = [all_scores[idx]
                          for idx in sorted_indices[:top_count]]
            self.logger.info(
                f"Top {top_count} scores: {[f'{s:.4f}' for s in top_scores]}")

        # Log SUCCESS message when completed
        success_msg = f"Aesthetic ranking complete - Processed {len(images)} images in batches of {batch_size}"
        self.logger.success(success_msg)

        return sorted_indices


if __name__ == "__main__":
    from src.utils.logger import create_console_logger
    import logging
    logger = create_console_logger("AestheticRanker", logging.INFO, use_colors=True)

    import time

    print("=== AestheticRanker Batch Processing Test ===")

    start = time.time()

    aesthetic_ranker = AestheticRanker(logger=logger)

    # Load test images
    img1 = Image.open(
        "/home/anhndt/pysvgenius/data/test/image.png").convert("RGB")
    img2 = Image.open(
        "/home/anhndt/pysvgenius/data/test/image01.png").convert("RGB")

    images = [img1, img2] * 20

    # Test 1: Default batch size
    print("\n--- Test 1: Default batch size (16) ---")
    start_test = time.time()
    scores = aesthetic_ranker.score(images)
    print(f"Individual scores: {[f'{s:.4f}' for s in scores]}")
    print(f"Score computation time: {time.time() - start_test:.2f}s")

    end = time.time()
    print(f"\nTotal test time: {end - start:.2f} seconds")
