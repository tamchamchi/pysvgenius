import logging
import os
from typing import Optional

import clip
import torch
import torch.nn as nn
from PIL import Image

from src.utils.image_utils import prepare_image_for_ranking
from src.utils.logger import get_library_logger

from .base import IRanker


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


class AestheticRanker(IRanker):
    """
    Ranks a list of SVG images by converting them to PNG and scoring
    their visual quality using CLIP embeddings and a trained aesthetic predictor.
    """

    def __init__(
        self,
        model_path: str = None,
        clip_model_path: str = None,
        device: str = "cuda:0",
        logger: Optional[logging.Logger] = None,
    ):
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

        self.model_path = model_path
        self.clip_model_path = clip_model_path
        self.device = device
        self.logger.info(f"Using device: {self.device}")
        self.logger.debug(f"Model path: {self.model_path}")

        self.predictor, self.clip_model, self.preprocessor = self._load()

    def __call__(
        self, svgs: list[str], prompt: str = None, batch_size: int = 16, top_k: int = 2
    ) -> list[int]:
        """
        Make AestheticRanker callable like a function.
        This is a convenience method that delegates to the process() method.

        Args:
            svgs (list[str]): List of SVG to rank.
            prompt (str): Unused in this ranker but included for interface consistency.
            batch_size (int): Number of images to process in each batch. Default is 16.
            top_k (int): Number of top ranked indices to return. Default is 2.

        Returns:
            list[int]: Top k indices sorted by score in descending order

        Example:
            >>> ranker = AestheticRanker()
            >>> images = [svg1, svg2, svg3]
            >>> top_indices = ranker(svgs, batch_size=8, top_k=2)  # Returns top 2 indices
        """
        self.logger.debug(
            f"__call__ invoked with {len(svgs) if svgs else 0} images, "
            f"batch_size={batch_size}, top_k={top_k}"
        )

        return self.process(
            svgs=svgs, prompt=prompt, batch_size=batch_size, top_k=top_k
        )

    def _load(self):
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
                self.clip_model_path, device=self.device)
            self.logger.success("Aesthetic Models loaded successfully")

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
            f"Computing aesthetic scores for batch of {len(images)} images"
        )

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
            f"Computed {len(scores)} aesthetic scores: {[f'{s:.4f}' for s in scores]}"
        )
        return scores

    def process(
        self, svgs: list[str], prompt: str = None, batch_size: int = 16, top_k: int = 2
    ) -> list[int]:
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

        if not svgs:
            self.logger.warning("No svgs provided for ranking")
            return []

        self.logger.info(
            f"Processing {len(svgs)} svgs with batch_size={batch_size}")

        images, _ = prepare_image_for_ranking(svgs)

        all_scores = []

        # Process images in batches
        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            current_batch = images[batch_start:batch_end]

            self.logger.debug(
                f"Processing batch {batch_start // batch_size + 1}/{(len(images) + batch_size - 1) // batch_size} "
                f"(images {batch_start + 1}-{batch_end})"
            )

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
                    f"Failed to process batch {batch_start // batch_size + 1}: {str(e)}"
                )
                # Add zeros for failed batch to maintain index consistency
                all_scores.extend([0.0] * len(current_batch))

        if len(all_scores) != len(images):
            self.logger.error(
                f"Score count mismatch: {len(all_scores)} scores for {len(images)} images"
            )
            return []

        # Sort indices by score in descending order
        indexed_scores = list(enumerate(all_scores))
        sorted_pairs = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, score in sorted_pairs]

        self.logger.debug(
            f"Sorted indices (desc): {sorted_indices[:10]}{'...' if len(sorted_indices) > 10 else ''}"
        )

        # Log top scores
        if all_scores:
            top_count = min(top_k, len(all_scores))
            top_scores = [all_scores[idx]
                          for idx in sorted_indices[:top_count]]
            self.logger.info(
                f"Top {top_count} scores: {[f'{s:.4f}' for s in top_scores]}"
            )

        # Log SUCCESS message when completed
        success_msg = f"Aesthetic ranking complete - Processed {len(images)} images in batches of {batch_size}"
        self.logger.success(success_msg)

        return sorted_indices[:top_k]


if __name__ == "__main__":
    import logging
    import time

    from src._config import MODEL_DIR
    from src.utils.logger import create_console_logger

    logger = create_console_logger(
        "AestheticRanker", logging.DEBUG, use_colors=True)
    model_path = os.path.join(MODEL_DIR, "sac+logos+ava1-l14-linearMSE.pth")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=== AestheticRanker Batch Processing Test ===")

    start = time.time()

    aesthetic_ranker = AestheticRanker(
        model_path=model_path, device=device, logger=logger
    )

    # Load test images
    img1 = Image.open(
        "/home/anhndt/pysvgenius/data/test/image.png").convert("RGB")
    img2 = Image.open(
        "/home/anhndt/pysvgenius/data/test/image01.png").convert("RGB")

    images = [img1, img2]

    # Test SVG
    svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="768" height="768" viewBox="0 0 384 384"><rect width="384" height="384" fill="#98746a"/><polygon points="2.0,0.0 2.0,383.0 140.0,378.0 196.0,8.0 243.0,376.0 382.0,383.0 382.0,0.0" fill="#ab796b"/><polygon points="229.0,242.0 156.0,242.0 141.0,383.0 242.0,380.0" fill="#003951"/><polygon points="166.0,116.0 166.0,219.0 217.0,219.0 217.0,116.0" fill="#5ad4d6"/><polygon points="242.0,223.0 144.0,221.0 142.0,237.0 239.0,240.0" fill="#030615"/><polygon points="149.0,96.0 234.0,99.0 193.0,58.0 198.0,47.0 191.0,36.0 184.0,48.0 190.0,58.0" fill="#f50004"/><polygon points="162.0,100.0 164.0,116.0 220.0,115.0 221.0,100.0" fill="#030615"/><polygon points="162.0,113.0 164.0,219.0" fill="#03253b"/><polygon points="221.0,101.0 218.0,219.0" fill="#03253b"/><polygon points="187.0,134.0 183.0,139.0 183.0,153.0 200.0,153.0 199.0,137.0 194.0,133.0" fill="#fcfdf1"/><polygon points="201.0,245.0 200.0,254.0 219.0,254.0 219.0,245.0" fill="#fcfdf1"/><polygon points="184.0,285.0 162.0,286.0 162.0,293.0 184.0,293.0" fill="#fcfdf1"/><polygon points="210.0,364.0 209.0,374.0 229.0,374.0 231.0,363.0" fill="#fcfdf1"/><polygon points="241.0,382.0 305.0,383.0 303.0,380.0" fill="#03253b"/><polygon points="188.0,137.0 187.0,150.0 196.0,150.0 196.0,139.0 192.0,136.0" fill="#030615"/><polygon points="79.0,383.0 140.0,383.0 140.0,380.0 81.0,380.0" fill="#03253b"/><polygon points="162.0,281.0 184.0,281.0 184.0,276.0 163.0,276.0" fill="#fcfdf1"/><g><circle cx="33.6" cy="30.4" r="9.1" fill="#03253b" /><circle cx="33.6" cy="30.4" r="7.3" fill="#5ad4d6" /><circle cx="33.6" cy="30.4" r="4.7" fill="#03253b" /></g><!-- Circle bytes: outer=53, middle=53, inner=53, total=159, colors: dark=#03253b, light=#5ad4d6, position: top-left --></svg>
"""

    # Create a simple second SVG for comparison
    svg2 = """
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
    <rect width="100" height="100" fill="red"/>
    <circle cx="50" cy="50" r="20" fill="blue"/>
</svg>
"""

    svg_list = [svg, svg2]

    # Test 1: Using score() method
    print("\n--- Test 1: score() method ---")
    start_test = time.time()
    scores = aesthetic_ranker.score(images)
    print(f"Individual scores: {[f'{s:.4f}' for s in scores]}")
    print(f"Score computation time: {time.time() - start_test:.2f}s")

    # Test 2: Using process() method
    print("\n--- Test 2: process() method ---")
    start_test = time.time()
    top_indices_process = aesthetic_ranker.process(
        svg_list, batch_size=16, top_k=2)
    print(f"Top indices from process(): {top_indices_process}")
    print(f"Process time: {time.time() - start_test:.2f}s")

    # Test 3: Using __call__ method (callable interface)
    print("\n--- Test 3: __call__ method (callable interface) ---")
    start_test = time.time()

    # Can call like a function
    top_indices_call = aesthetic_ranker(svg_list, batch_size=16, top_k=2)
    print(f"Top indices from __call__(): {top_indices_call}")

    # Verify results are identical
    print(f"Results identical: {top_indices_process == top_indices_call}")
    print(f"Call time: {time.time() - start_test:.2f}s")

    # Test 4: Different parameters via __call__
    print("\n--- Test 4: __call__ with different parameters ---")

    # Get only top 1
    top_1 = aesthetic_ranker(svg_list, top_k=1)
    print(f"Top 1 index: {top_1}")

    # Use smaller batch size
    top_indices_small_batch = aesthetic_ranker(svg_list, batch_size=1, top_k=2)
    print(f"Small batch top indices: {top_indices_small_batch}")

    # Test 5: Larger dataset
    print("\n--- Test 5: Larger dataset via __call__ ---")
    large_svgs = svg_list * 5  # 10 images
    start_test = time.time()
    top_3 = aesthetic_ranker(large_svgs, batch_size=4, top_k=3)
    print(f"Top 3 from {len(large_svgs)} images: {top_3}")
    print(f"Large dataset time: {time.time() - start_test:.2f}s")

    end = time.time()
    print(f"\nTotal test time: {end - start:.2f} seconds")

    print("\n=== Usage Examples ===")
    print("# Method 1: Traditional method call")
    print("top_indices = ranker.process(images, batch_size=16, top_k=3)")
    print()
    print("# Method 2: Callable interface (more convenient)")
    print("top_indices = ranker(images, batch_size=16, top_k=3)")
    print()
    print("# Method 3: Quick ranking with defaults")
    print("best_2 = ranker(images)  # Uses default batch_size=16, top_k=2")
