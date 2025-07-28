
import clip
import torch
import torch.nn as nn
from PIL import Image

from src.common import registry
from src.utils.image_utils import prepare_image_for_ranking

from .base import IRanker


@registry.register_ranker("aesthetic")
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
    ):
        """
        Initialize AestheticRanker.

        Args:
            model_path (str): Path to the trained aesthetic predictor model
            clip_model_path (str): Path to the CLIP model weights
            device (str): Device to run the models on (cuda:0 or cpu)
        """
        # Store configuration parameters
        self.model_path = model_path
        self.clip_model_path = clip_model_path
        self.device = device

        # Load the aesthetic predictor and CLIP model
        self.predictor, self.clip_model, self.preprocessor = self._load()

    def __call__(
        self, svgs: list[str], prompt: str = None, batch_size: int = 16, top_k: int = 2
    ) -> list[int]:
        """
        Make AestheticRanker callable like a function.
        This is a convenience method that delegates to the process() method.

        Args:
            svgs (list[str]): List of SVG strings to rank
            prompt (str): Unused in this ranker but included for interface consistency
            batch_size (int): Number of images to process in each batch. Default is 16
            top_k (int): Number of top ranked indices to return. Default is 2

        Returns:
            list[int]: Top k indices sorted by score in descending order

        Example:
            >>> ranker = AestheticRanker()
            >>> images = [svg1, svg2, svg3]
            >>> top_indices = ranker(svgs, batch_size=8, top_k=2)  # Returns top 2 indices
        """
        # Delegate to the main processing method
        return self.process(
            svgs=svgs, prompt=prompt, batch_size=batch_size, top_k=top_k
        )

    def _load(self):
        """
        Loads the aesthetic predictor and the CLIP model (ViT-L/14).

        Returns:
            tuple: (predictor, clip_model, preprocessor)
                - predictor (nn.Module): Trained aesthetic predictor model
                - clip_model (CLIPModel): Pretrained CLIP model
                - preprocessor (Callable): Image preprocessor function
        """
        try:
            # Load the trained aesthetic predictor state dictionary
            state_dict = torch.load(self.model_path, map_location=self.device)

            # Initialize predictor with 768 dimensions (CLIP ViT-L/14 embedding size)
            predictor = AestheticPredictor(768)
            predictor.load_state_dict(state_dict)
            predictor.to(self.device)
            predictor.eval()  # Set to evaluation mode

            # Load CLIP ViT-L/14 model and preprocessor
            clip_model, preprocessor = clip.load(
                self.clip_model_path, device=self.device)

            return predictor, clip_model, preprocessor

        except Exception as e:
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
            list[float]: List of aesthetic scores normalized to [0, 1] range
        """
        # Handle single image case by wrapping in list
        if isinstance(images, Image.Image):
            images = [images]

        # Return empty list for empty input
        if not images:
            return []

        # Preprocess all images and create batch tensors
        batch_tensors = []
        for img in images:
            # Apply CLIP preprocessing and add batch dimension
            preprocessed = self.preprocessor(img).unsqueeze(0)
            batch_tensors.append(preprocessed)

        # Concatenate into a single batch tensor and move to device
        batch = torch.cat(batch_tensors, dim=0).to(self.device)

        scores = []
        with torch.no_grad():  # Disable gradient computation for inference
            # Extract image features using CLIP encoder
            batch_features = self.clip_model.encode_image(batch)
            # Normalize features to unit vectors
            batch_features /= batch_features.norm(dim=-1, keepdim=True)

            # Predict aesthetic scores using the trained predictor
            batch_scores = self.predictor(batch_features.float())

            # Normalize scores from predictor output to [0, 1] range
            normalized_scores = (batch_scores / 10.0).squeeze().cpu().numpy()

            # Handle single item case (numpy returns scalar for single element)
            if normalized_scores.ndim == 0:
                scores = [float(normalized_scores)]
            else:
                scores = normalized_scores.tolist()

        return scores

    def process(
        self, svgs: list[str], prompt: str = None, batch_size: int = 16, top_k: int = 2
    ) -> list[int]:
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
            list[int]: Indices sorted by aesthetic score in descending order (best first)
        """
        # Ignore prompt parameter (not used in aesthetic ranking)
        _ = prompt

        # Return empty list if no input provided
        if not svgs:
            return []

        # Convert SVG strings to PIL Images for processing
        images, _ = prepare_image_for_ranking(svgs)

        all_scores = []

        # Process images in batches to manage memory usage
        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            current_batch = images[batch_start:batch_end]

            try:
                # Compute aesthetic scores for current batch
                batch_scores = self.score(current_batch)
                all_scores.extend(batch_scores)

            except Exception as e:
                # Add zeros for failed batch to maintain index consistency
                all_scores.extend([0.0] * len(current_batch))

        # Verify that we have scores for all images
        if len(all_scores) != len(images):
            return []

        # Create (index, score) pairs and sort by score in descending order
        indexed_scores = list(enumerate(all_scores))
        sorted_pairs = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, score in sorted_pairs]

        # Return top k indices (best aesthetic scores first)
        return sorted_indices[:top_k], all_scores

    @classmethod
    def from_config(cls, cfg):
        model_path = cfg.get("model_path")
        clip_model_path = cfg.get("clip_model_path", "ViT-L/14")
        device = cfg.get("device")
        return cls(model_path, clip_model_path, device)
