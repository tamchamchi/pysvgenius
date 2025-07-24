# from src import load_converter, load_generator
# generator = load_generator("sdxl-turbo")
# converter = load_converter("vtracer-grib-search")

# images = generator("a cartoon cat", num_images=4)
# svgs = converter(images, limit=20000)

# for i, svg in enumerate(svgs):
#     with open(f"{i}.svg", "w") as f:
#         f.write(svg)

from src import load_config
options = ["generator.sdxl-turbo.num_images=6", "generator.sdxl-turbo.seed=42"]
config = load_config(options=options)
print(config.to_dict())
