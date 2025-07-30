from pysvgenius import setup_path, load_config
from pysvgenius.converter import load_converter
from pysvgenius.generator import load_generator
from pysvgenius.ranker import load_ranker
from pysvgenius.optimizer import load_optimizer

from PIL import Image
setup_path()
# generator = load_generator("sdxl-turbo")
# converter = load_converter("vtracer-binary-search")

# # # Example
# prompt = "flat color illustration, app icon, Futuristic skyscraper with neon lights, inspired by Tom Whalen, atmospheric light, soft color palette, bold outlines, golden hour lighting."
# images = generator(prompt, num_images=4)
# svgs = converter(images, limit=15000)

# siglip_ranker = load_ranker("siglip")
# idx_1 = siglip_ranker(svgs, prompt="Futuristic skyscraper with neon lights", top_k=2)
# ranked_svg = [svgs[i] for i in idx_1]

# aesthetic_ranker = load_ranker("aesthetic")
# idx_2 = aesthetic_ranker(ranked_svg, top_k=1)

# best_svg = ranked_svg[idx_2[0]]
# origin_image = images[idx_1[idx_2[0]]]

# with open ("skyscraper_svg.svg", "w") as f:
#     f.write(best_svg)

# origin_image.save("skyscraper_image.png")

# config = load_config()
# optimizer = load_optimizer("diffvg")
# with open("/home/anhndt/pysvgenius/skyscraper_svg.svg", "r") as f:
#     svg = f.read()
# image = Image.open("/home/anhndt/pysvgenius/skyscraper_image.png")
# optimizer(svg=svg, image=image, args=config.optimizer_cfg["diffvg"]["args"], limit=15000)
