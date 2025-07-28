from src import load_converter, load_generator, DEFAULT_CONFIG
# generator = load_generator("sdxl-turbo")


from src import load_config
options = ["generator.sdxl-turbo.num_images=6", "generator.sdxl-turbo.seed=42", "generator.sdxl-turbo.prefix=", "generator.sdxl-turbo.suffix="]
config = load_config(options=options)
# print(config.to_dict())
print(config.generator_cfg["sdxl-turbo"])
generator = load_generator("sdxl-turbo", config.generator_cfg["sdxl-turbo"])
converter = load_converter("vtracer-binary-search")

images = generator("a lighthouse", num_images=5)
svgs = converter(images, limit=20000)

for i, svg in enumerate(svgs):
    with open(f"{i}.svg", "w") as f:
        f.write(svg)
