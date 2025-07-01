from .base import ISVGRanker

from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import torch
from src.utils.image_utils import process_svg_to_image


class SigLipRanker(ISVGRanker):
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "google/siglip-so400m-patch14-384"
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)

    def compute_siglip_similarity(self, image: Image.Image, text: str) -> float:
        """
        Compute the similarity score between an image and a text using the SigLIP model.

        Args:
            image (PIL.Image.Image): The input image.
            text (str): The text description.

        Returns:
            float: Cosine similarity between image and text embeddings.
        """
        # Preprocess inputs
        inputs = self.processor(images=image, text=text,
                                return_tensors="pt").to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds  # shape: (1, dim)
            text_embeds = outputs.text_embeds    # shape: (1, dim)

        # Normalize and compute cosine similarity
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        similarity = (image_embeds * text_embeds).sum().item()

        return similarity

    def process(self, svg_list, prompt=None):
        results = []

        for svg in svg_list:
            image = process_svg_to_image(svg_code=svg)
            score = self.compute_siglip_similarity(image=image, text=prompt)
            results.append({
                "svg": svg,
                "score": score
            })
        return results


if __name__ == "__main__":
    ranker = SigLipRanker()
    svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="768" height="768" viewBox="0 0 384 384"><rect width="384" height="384" fill="#98746a"/><polygon points="2.0,0.0 2.0,383.0 140.0,378.0 196.0,8.0 243.0,376.0 382.0,383.0 382.0,0.0" fill="#ab796b"/><polygon points="229.0,242.0 156.0,242.0 141.0,383.0 242.0,380.0" fill="#003951"/><polygon points="166.0,116.0 166.0,219.0 217.0,219.0 217.0,116.0" fill="#5ad4d6"/><polygon points="242.0,223.0 144.0,221.0 142.0,237.0 239.0,240.0" fill="#030615"/><polygon points="149.0,96.0 234.0,99.0 193.0,58.0 198.0,47.0 191.0,36.0 184.0,48.0 190.0,58.0" fill="#f50004"/><polygon points="162.0,100.0 164.0,116.0 220.0,115.0 221.0,100.0" fill="#030615"/><polygon points="162.0,113.0 164.0,219.0" fill="#03253b"/><polygon points="221.0,101.0 218.0,219.0" fill="#03253b"/><polygon points="187.0,134.0 183.0,139.0 183.0,153.0 200.0,153.0 199.0,137.0 194.0,133.0" fill="#fcfdf1"/><polygon points="201.0,245.0 200.0,254.0 219.0,254.0 219.0,245.0" fill="#fcfdf1"/><polygon points="184.0,285.0 162.0,286.0 162.0,293.0 184.0,293.0" fill="#fcfdf1"/><polygon points="210.0,364.0 209.0,374.0 229.0,374.0 231.0,363.0" fill="#fcfdf1"/><polygon points="241.0,382.0 305.0,383.0 303.0,380.0" fill="#03253b"/><polygon points="188.0,137.0 187.0,150.0 196.0,150.0 196.0,139.0 192.0,136.0" fill="#030615"/><polygon points="79.0,383.0 140.0,383.0 140.0,380.0 81.0,380.0" fill="#03253b"/><polygon points="162.0,281.0 184.0,281.0 184.0,276.0 163.0,276.0" fill="#fcfdf1"/><g><circle cx="33.6" cy="30.4" r="9.1" fill="#03253b" /><circle cx="33.6" cy="30.4" r="7.3" fill="#5ad4d6" /><circle cx="33.6" cy="30.4" r="4.7" fill="#03253b" /></g><!-- Circle bytes: outer=53, middle=53, inner=53, total=159, colors: dark=#03253b, light=#5ad4d6, position: top-left --></svg>
"""
    svg_list = [svg]

    score = ranker.process(svg_list, "a lighthouse overlooking the ocean")
    print(score)
