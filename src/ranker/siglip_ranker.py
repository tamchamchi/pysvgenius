import logging
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from src.utils.image_utils import prepare_image_for_ranking
from src.utils.logger import create_console_logger, get_library_logger

from .base import IRanker


class SigLipRanker(IRanker):
    def __init__(self, model_path: str = "google/siglip-so400m-patch14-384", device: str = "cuda:0", logger: Optional[logging.Logger] = None):

        if logger is not None:
            self.logger = logger
        else:
            self.logger = get_library_logger(
                f"{__name__}.{self.__class__.__name__}")
        self.device = device
        self.model_path = model_path
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, use_fast=True)
            self.model = AutoModel.from_pretrained(
                self.model_path).to(self.device)

            self.logger.info(f"Using device: {self.device}")
            self.logger.info(f"Model path: {self.model_path}")
            self.logger.success(
                "SIGLIP Model and processor loaded successfully.")
        except Exception as e:
            self.logger.error(
                f"❌ Failed to load model or processor: {type(e).__name__}: {e}")
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

        self.logger.debug(
            f"Computing SigLIP scores for {len(images)} images with prompt: '{prompt}'")

        try:
            inputs = self.processor(
                text=texts,
                images=images,
                padding="max_length",
                return_tensors="pt"  # Fixed typo: was "return_tensor"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image  # Fixed typo: was "logist_per_images"

            # Convert logits to probabilities using sigmoid
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            # Handle single image case
            if probs.ndim == 0:
                scores = [float(probs)]
            else:
                scores = probs.tolist()

            self.logger.debug(
                f"Computed scores: {[f'{s:.4f}' for s in scores]}")
            return scores

        except Exception as e:
            self.logger.error(f"Failed to compute SigLIP scores: {str(e)}")
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
            self.logger.warning("No SVGs provided for ranking")
            return []

        if not prompt:
            self.logger.warning("No prompt provided for ranking")
            return []

        self.logger.info(
            f"Processing {len(svgs)} SVGs with prompt: '{prompt}' using batch_size={batch_size}")

        # Convert all SVGs to images using robust conversion
        images, indexs = prepare_image_for_ranking(svgs)

        if not images:
            self.logger.warning("No valid images after SVG conversion")
            return []

        self.logger.info(
            f"Successfully converted {len(images)}/{len(svgs)} SVGs to images")

        # Process images in batches
        all_scores = []

        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            current_batch = images[batch_start:batch_end]

            self.logger.debug(f"Processing batch {batch_start//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size} "
                              f"(images {batch_start+1}-{batch_end})")

            try:
                # Process current batch
                batch_scores = self.score(current_batch, prompt)
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

        # Create full scores array with 0.0 for failed conversions
        full_scores = [0.0] * len(svgs)
        for idx, valid_idx in enumerate(indexs):
            full_scores[valid_idx] = all_scores[idx]

        # Sort indices by score in descending order
        indexed_scores = list(enumerate(full_scores))
        sorted_pairs = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, score in sorted_pairs]

        self.logger.debug(
            f"Sorted indices (desc): {sorted_indices[:10]}{'...' if len(sorted_indices) > 10 else ''}")

        # Log top scores
        if full_scores:
            top_count = min(top_k, len(full_scores))
            top_scores = [full_scores[idx]
                          for idx in sorted_indices[:top_count]]
            self.logger.info(
                f"Top {top_count} scores: {[f'{s:.4f}' for s in top_scores]}")

        # Log SUCCESS message when completed
        success_msg = f"SigLIP ranking complete - Processed {len(svgs)} SVGs"
        self.logger.success(success_msg)

        return sorted_indices[:top_k]

    def __call__(self, svgs: list[str], prompt: str, batch_size: int = 16, top_k: int = 2) -> list[int]:
        """
        Make SigLipRanker callable like a function.
        This is a convenience method that delegates to the process() method.

        Args:
            svgs (list[str]): List of SVG strings to rank.
            prompt (str): Text prompt to compare against.
            batch_size (int): Number of images to process in each batch. Default is 16.
            top_k (int): Number of top ranked indices to return. Default is 2.

        Returns:
            list[int]: Top k indices sorted by score in descending order

        Example:
            >>> ranker = SigLipRanker()
            >>> svgs = [svg1, svg2, svg3]
            >>> top_indices = ranker(svgs, "a lighthouse", batch_size=8, top_k=2)
        """
        self.logger.debug(f"__call__ invoked with {len(svgs) if svgs else 0} SVGs, "
                          f"prompt='{prompt}', batch_size={batch_size}, top_k={top_k}")

        return self.process(svgs=svgs, prompt=prompt, batch_size=batch_size, top_k=top_k)


if __name__ == "__main__":
    import time

    logger = create_console_logger(
        "SigLipRankerTest", logging.INFO, use_colors=True)

    print("=== SigLipRanker Test ===")

    start_total = time.time()

    ranker = SigLipRanker(logger=logger)

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

    svgs = [svg, svg2]
    prompt = "a lighthouse overlooking the ocean"

    # Test 1: Using process() method
    print("\n--- Test 1: process() method ---")
    print(f"Prompt: '{prompt}'")
    print(f"Number of SVGs: {len(svgs)}")

    start_test = time.time()
    top_indices_process = ranker.process(
        svgs, prompt, batch_size=16, top_k=2)
    print(f"Top indices from process(): {top_indices_process}")
    print(f"Process time: {time.time() - start_test:.2f}s")

    # Test 2: Using __call__ method (callable interface)
    print("\n--- Test 2: __call__ method (callable interface) ---")
    start_test = time.time()

    # Can call like a function
    top_indices_call = ranker(svgs, prompt, batch_size=16, top_k=2)
    print(f"Top indices from __call__(): {top_indices_call}")

    # Verify results are identical
    print(f"Results identical: {top_indices_process == top_indices_call}")
    print(f"Call time: {time.time() - start_test:.2f}s")

    # Test 3: Different parameters
    print("\n--- Test 3: Different parameters ---")

    # Get only top 1
    top_1 = ranker(svgs, prompt, top_k=1)
    print(f"Top 1 index: {top_1}")

    # Different prompt
    different_prompt = "abstract colorful art"
    top_different = ranker(svgs, different_prompt, top_k=2)
    print(f"With prompt '{different_prompt}': {top_different}")

    # Test 4: Test with individual image scoring
    print("\n--- Test 4: Individual image scoring ---")
    try:
        # Convert SVGs to images for direct scoring
        img_list1 = prepare_image_for_ranking([svg])
        img_list2 = prepare_image_for_ranking([svg2])

        if img_list1 and img_list2:
            scores = ranker.score([img_list1[0], img_list2[0]], prompt)
            print(f"Direct image scores: {[f'{s:.4f}' for s in scores]}")
        else:
            print("Failed to convert SVGs to images")
    except Exception as e:
        print(f"Image scoring failed: {e}")

    end_total = time.time()
    print(f"\nTotal test time: {end_total - start_total:.2f} seconds")

    print("\n=== Usage Examples ===")
    print("# Method 1: Traditional method call")
    print("top_indices = ranker.process(svgs, prompt, batch_size=16, top_k=3)")
    print("")
    print("# Method 2: Callable interface (more convenient)")
    print("top_indices = ranker(svgs, prompt, batch_size=16, top_k=3)")
    print("")
    print("# Method 3: Quick ranking with defaults")
    print("best_2 = ranker(svgs, prompt)  # Uses default batch_size=16, top_k=2")
