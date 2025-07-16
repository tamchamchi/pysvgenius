import os

import torch

from src._config import AESTHETIC_MODEL_PATH, CLIP_MODEL_PATH
from src.converter import ImageToConverterFactory
from src.generator import TextToImageGeneratorFactory
from src.ranker import SVGRankerFactory
from src.utils import create_console_logger
from src.utils.image_utils import svg_to_png

logger_0 = create_console_logger("Generator")
logger_1 = create_console_logger("Convertor")
logger_2 = create_console_logger("Ranker")

device = torch.device("cuda:0")

generator = TextToImageGeneratorFactory.create(
    "sdxl-turbo", device=device, logger=logger_0
)

convertor = ImageToConverterFactory.create("vtracer", logger=logger_1)

siglip_ranker = SVGRankerFactory.create(
    "siglip", device=device, logger=logger_2)
aesthetic_ranker = SVGRankerFactory.create(
    "aesthetic",
    model_path=AESTHETIC_MODEL_PATH,
    clip_model_path=CLIP_MODEL_PATH,
    device=device,
    logger=logger_2,
)

prompt = "flat color illustration, watercolor painting of a fire xbreathing dragon, inspired by Tom Whalen, vibrant palette, bold outlines, simple shapes, app icon."

output = generator(prompt, num_images=10)

svgs = convertor(output, limit=20000)

top2_svg = siglip_ranker(svgs=svgs, prompt=prompt, top_k=2)

ranked_svg = [svgs[i] for i in top2_svg]

for i, svg in enumerate(ranked_svg):
    image = svg_to_png(svg)
    image.save(f'{i}.png')
