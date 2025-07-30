import os
from pathlib import Path
from typing import Optional

import clip
import torch
import torch.nn as nn
from PIL import Image

from ..common import registry
from ..utils import prepare_image_for_ranking
from .base import IRanker


class AestheticPredictor(nn.Module):
    """
    A feed-forward neural network for predicting the aesthetic score
    based on CLIP image embeddings.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        # Define multi-layer perceptron with dropout for regularization
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),  # Output single aesthetic score
        )

    def forward(self, x):
        """Forward pass through the network"""
        return self.layers(x)


@registry.register_ranker("aesthetic")
class AestheticRanker(IRanker):
    """
    Ranks a list of SVG images by converting them to PNG and scoring
    their visual quality using CLIP embeddings and a trained aesthetic predictor.
    """

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
            state_dict = torch.load(self.model_path, map_location=self.device)
            predictor = AestheticPredictor(768)
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

            clip_model, preprocessor = clip.load(
                self.clip_model_path,
                device=self.device,
                download_root=clip_download_root
            )

            return predictor, clip_model, preprocessor

        except Exception as e:
            raise RuntimeError(f"✗ Failed to load models: {e}") from e


    def score(self, images) -> list[float]:
        """
        Computes aesthetic scores for given images (supports both single image and batch).

        Args:
            images: Can be either:
                - Single PIL Image
                - List of PIL Images
                - Batch tensor

        Returns:
            list[float]: List of aesthetic scores normalized to [0, 1] range
        """
        # Handle single image case by wrapping in list
        if isinstance(images, Image.Image):
            images = [images]
            print("🔄 Processing single image")
        else:
            print(f"🔄 Processing batch of {len(images)} images")

        # Return empty list for empty input
        if not images:
            print("⚠ No images provided for scoring")
            return []

        print(f"🔄 Preprocessing {len(images)} images for CLIP...")

        # Preprocess all images and create batch tensors
        batch_tensors = []
        failed_count = 0

        for i, img in enumerate(images):
            try:
                # Apply CLIP preprocessing and add batch dimension
                preprocessed = self.preprocessor(img).unsqueeze(0)
                batch_tensors.append(preprocessed)
            except Exception as e:
                print(f"⚠ Failed to preprocess image {i}: {e}")
                failed_count += 1

        if not batch_tensors:
            print("✗ No images could be preprocessed")
            return []

        if failed_count > 0:
            print(f"⚠ {failed_count}/{len(images)} images failed preprocessing")

        # Concatenate into a single batch tensor and move to device
        print(f"🔄 Creating batch tensor and moving to {self.device}...")
        batch = torch.cat(batch_tensors, dim=0).to(self.device)
        print(f"✓ Batch tensor shape: {batch.shape}")

        scores = []
        with torch.no_grad():  # Disable gradient computation for inference
            print("🔄 Extracting CLIP features...")
            # Extract image features using CLIP encoder
            batch_features = self.clip_model.encode_image(batch)
            print(f"✓ CLIP features shape: {batch_features.shape}")

            # Normalize features to unit vectors
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            print("✓ Features normalized")

            print("🔄 Predicting aesthetic scores...")
            # Predict aesthetic scores using the trained predictor
            batch_scores = self.predictor(batch_features.float())
            print(f"✓ Raw predictor output shape: {batch_scores.shape}")

            # Normalize scores from predictor output to [0, 1] range
            normalized_scores = (batch_scores / 10.0).squeeze().cpu().numpy()
            print(
                f"✓ Normalized scores shape: {normalized_scores.shape if hasattr(normalized_scores, 'shape') else 'scalar'}")

            # Handle single item case (numpy returns scalar for single element)
            if normalized_scores.ndim == 0:
                scores = [float(normalized_scores)]
                print(f"✓ Single score: {scores[0]:.4f}")
            else:
                scores = normalized_scores.tolist()
                print(
                    f"✓ Batch scores: {[f'{s:.4f}' for s in scores[:5]]}{'...' if len(scores) > 5 else ''}")

        print(f"✅ Aesthetic scoring complete. Generated {len(scores)} scores")
        return scores

    def process(
        self, svgs: list[str], prompt: str = None, batch_size: int = 16, top_k: int = 2
    ) -> tuple[list[int], list[float]]:
        """
        Ranks a list of SVG strings based on predicted aesthetic score using batch processing.

        Args:
            svgs (list[str]): List of SVG strings to rank
            prompt (str): Unused in this ranker but included for interface consistency
            batch_size (int): Number of images to process in each batch. Default is 16
                             Larger batch_size = faster processing but more GPU memory usage
                             Smaller batch_size = slower processing but less GPU memory usage
            top_k (int): Number of top ranked indices to return

        Returns:
            tuple[list[int], list[float]]: (sorted_indices, all_scores)
                - sorted_indices: Indices sorted by aesthetic score in descending order (best first)
                - all_scores: All computed scores for debugging/analysis
        """
        # Ignore prompt parameter (not used in aesthetic ranking)
        if prompt:
            print(
                f"⚠ Prompt provided but ignored in aesthetic ranking: '{prompt}'")

        # Return empty results if no input provided
        if not svgs:
            print("⚠ No SVGs provided, returning empty results")
            return [], []

        print("🔄 Converting SVGs to PIL Images...")
        # Convert SVG strings to PIL Images for processing
        images, valid_indices = prepare_image_for_ranking(svgs)

        if not images:
            print("✗ No valid images after SVG conversion")
            return [], []

        if len(images) < len(svgs):
            failed_count = len(svgs) - len(images)
            print(f"⚠ {failed_count} SVGs failed conversion")

        all_scores = []

        # Process images in batches to manage memory usage
        for batch_idx, batch_start in enumerate(range(0, len(images), batch_size)):
            batch_end = min(batch_start + batch_size, len(images))
            current_batch = images[batch_start:batch_end]

            try:
                # Compute aesthetic scores for current batch
                batch_scores = self.score(current_batch)
                all_scores.extend(batch_scores)

            except Exception as e:
                print(f"✗ Failed to process batch {batch_idx + 1}: {e}")
                # Add zeros for failed batch to maintain index consistency
                zeros_added = len(current_batch)
                all_scores.extend([0.0] * zeros_added)
                print(f"⚠ Added {zeros_added} zero scores for failed batch")

        # Verify that we have scores for all images
        if len(all_scores) != len(images):
            print(
                f"✗ Score count mismatch: {len(all_scores)} scores != {len(images)} images")
            return [], []

        # Create (index, score) pairs and sort by score in descending order
        indexed_scores = list(enumerate(all_scores))
        sorted_pairs = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, score in sorted_pairs]

        # Return top k indices and all scores
        result_indices = sorted_indices[:top_k]

        return result_indices

    @classmethod
    def from_config(cls, cfg):
        """Create AestheticRanker from configuration dictionary."""
        print(f"🔄 Creating AestheticRanker from config: {cfg}")

        ranker = cls(
            model_path=cfg.get("model_path"),
            clip_model_path=cfg.get("clip_model_path"),
            device=cfg.get("device", "cuda:0"),
            auto_download=cfg.get("auto_download", True)
        )

        print("✅ AestheticRanker created from config")
        return ranker
