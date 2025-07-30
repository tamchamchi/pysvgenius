from pysvgenius import load_converter, load_generator, load_optimizer, setup_path, load_config, load_ranker
from PIL import Image
# generator = load_generator("sdxl-turbo")
# converter = load_converter("vtracer-grid-search")

# # # Example
# prompt = "flat color illustration, app icon, a baby dragon ,inspired by Tom Whalen, atmospheric light, soft color palette, bold outlines, golden hour lighting."
# images = generator(prompt, num_images=4)
# svgs = converter(images, limit=15000)
setup_path()
config = load_config()
optimizer = load_optimizer("diffvg")
with open("/home/anhndt/pysvgenius/data/test/results.svg", "r") as f:
    svg = f.read()
image = Image.open("/home/anhndt/pysvgenius/data/test/Screenshot 2025-05-29 175550.png")
optimizer(svg=svg, image=image, args=config.optimizer_cfg["diffvg"]["args"], limit=20000)

# ranker = load_ranker("aesthetic")