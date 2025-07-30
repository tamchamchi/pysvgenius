from pysvgenius import setup_path
from pysvgenius.converter import load_converter
from pysvgenius.generator import load_generator
from pysvgenius.ranker import load_ranker

setup_path()
generator = load_generator("sdxl-turbo")
converter = load_converter("vtracer-binary-search")

# # Example
prompt = "flat color illustration, app icon, Futuristic skyscraper with neon lights, inspired by Tom Whalen, atmospheric light, soft color palette, bold outlines, golden hour lighting."
images = generator(prompt, num_images=4)
svgs = converter(images, limit=15000)

siglip_ranker = load_ranker("siglip")
idx_1, score = siglip_ranker(svgs, prompt="Futuristic skyscraper with neon lights", top_k=2)
ranked_svg = [svgs[i] for i in idx_1]

aesthetic_ranker = load_ranker("aesthetic")
idx_2, score = aesthetic_ranker(ranked_svg, top_k=1)
