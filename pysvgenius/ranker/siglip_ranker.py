
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from ..common import registry
from ..utils import prepare_image_for_ranking
from .base import IRanker


@registry.register_ranker("siglip")
class SigLipRanker(IRanker):
    def __init__(self, model_path: str = "google/siglip-so400m-patch14-384", device: str = "cuda:0"):
        """
        Initialize SigLipRanker.

        Args:
            model_path (str): HuggingFace model path for SigLIP model
            device (str): Device to run the model on (cuda:0 or cpu)
        """
        self.device = device
        self.model_path = model_path

        try:
            # Load processor and model
            print(f"Loading SigLIP model: {self.model_path}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, use_fast=True)
            self.model = AutoModel.from_pretrained(
                self.model_path).to(self.device)

            print(f"✓ Using device: {self.device}")
            print(f"✓ Model path: {self.model_path}")
            print("✓ SigLIP Model and processor loaded successfully.")

        except Exception as e:
            print(
                f"✗ Failed to load model or processor: {type(e).__name__}: {e}")
            raise

    def score(self, images: list[Image.Image], prompt: str) -> list[float]:
        """
        Compute similarity scores between images and text prompt using SigLIP.

        Args:
            images (list[Image.Image]): List of PIL Images to score.
            prompt (str): Text prompt to compare against.

        Returns:
            list[float]: List of similarity scores for each image.
        """
        if not images:
            return []

        texts = [prompt]

        print(
            f"Computing SigLIP scores for {len(images)} images with prompt: '{prompt}'")

        try:
            # Prepare inputs for the model
            inputs = self.processor(
                text=texts,
                images=images,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            # Compute similarity scores
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image

            # Convert logits to probabilities using sigmoid
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            # Handle single image case
            if probs.ndim == 0:
                scores = [float(probs)]
            else:
                scores = probs.tolist()

            print(f"Computed scores: {[f'{s:.4f}' for s in scores]}")
            return scores

        except Exception as e:
            print(f"✗ Failed to compute SigLIP scores: {str(e)}")
            return [0.0] * len(images)

    def process(self, svgs: list[str], prompt: str, batch_size: int = 16, top_k: int = 2) -> list[int]:
        """
        Ranks a list of SVGs based on SigLIP similarity scores with the given prompt.

        Args:
            svgs (list[str]): List of SVG strings to rank.
            prompt (str): Text prompt to compare against.
            batch_size (int): Number of images to process in each batch. Default is 16.
            top_k (int): Number of top ranked indices to return. Default is 2.

        Returns:
            list[int]: Indices sorted by score in descending order (top_k only).
        """
        if not svgs:
            print("Warning: No SVGs provided for ranking")
            return []

        if not prompt:
            print("Warning: No prompt provided for ranking")
            return []

        print(
            f"Processing {len(svgs)} SVGs with prompt: '{prompt}' using batch_size={batch_size}")

        # Convert all SVGs to images using robust conversion
        images, valid_indices = prepare_image_for_ranking(svgs)

        if not images:
            print("Warning: No valid images after SVG conversion")
            return []

        print(
            f"Successfully converted {len(images)}/{len(svgs)} SVGs to images")

        # Process images in batches
        all_scores = []

        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            current_batch = images[batch_start:batch_end]

            print(f"Processing batch {batch_start//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size} "
                  f"(images {batch_start+1}-{batch_end})")

            try:
                # Process current batch
                batch_scores = self.score(current_batch, prompt)
                all_scores.extend(batch_scores)

                # Log progress for larger datasets
                if len(images) > batch_size:
                    print(f"Processed {batch_end}/{len(images)} images")

            except Exception as e:
                print(
                    f"✗ Failed to process batch {batch_start//batch_size + 1}: {str(e)}")
                # Add zeros for failed batch to maintain index consistency
                all_scores.extend([0.0] * len(current_batch))

        # Verify score count matches image count
        if len(all_scores) != len(images):
            print(
                f"✗ Score count mismatch: {len(all_scores)} scores for {len(images)} images")
            return []

        # Create full scores array with 0.0 for failed conversions
        full_scores = [0.0] * len(svgs)
        for idx, valid_idx in enumerate(valid_indices):
            full_scores[valid_idx] = all_scores[idx]

        # Sort indices by score in descending order
        indexed_scores = list(enumerate(full_scores))
        sorted_pairs = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, score in sorted_pairs]

        print(
            f"Sorted indices (desc): {sorted_indices[:10]}{'...' if len(sorted_indices) > 10 else ''}")

        # Log top scores
        if full_scores:
            top_count = min(top_k, len(full_scores))
            top_scores = [full_scores[idx]
                          for idx in sorted_indices[:top_count]]
            print(
                f"Top {top_count} scores: {[f'{s:.4f}' for s in top_scores]}")

        # Success message when completed
        print(f"✓ SigLIP ranking complete - Processed {len(svgs)} SVGs")

        return sorted_indices[:top_k]

    @classmethod
    def from_config(cls, cfg):
        """Create SigLipRanker from configuration dictionary."""
        return cls(
            model_path=cfg.get(
                "model_path", "google/siglip-so400m-patch14-384"),
            device=cfg.get("device", "cuda:0")
        )
