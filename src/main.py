from src import load_converter, load_generator

generator = load_generator("sdxl-turbo")
converter = load_converter("vtracer-grid-search")

# Example
prompt = "flat color illustration, app icon, a baby dragon ,inspired by Tom Whalen, atmospheric light, soft color palette, bold outlines, golden hour lighting."
images = generator(prompt, num_images=4)
svgs = converter(images, limit=15000)
