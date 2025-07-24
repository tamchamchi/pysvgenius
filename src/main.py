from src.common.registry import registry
from src.common.config import Config
from types import SimpleNamespace

generator_cls = registry.get_generator_class("sdxl-turbo")


args = SimpleNamespace(
    cfg_path="/home/anhndt/pysvgenius/configs/configs.yaml",
    options=["generator.sdxl-turbo.num_images=4", "generator.sdxl-turbo.height=512"]
)

config = Config(args=args)
cfg_dict = config.to_dict()

# generator_cfg = cfg_dict["generator"]['sdxl-turbo']
# generator = generator_cls.from_config(generator_cfg)
# images = generator("big dog",num_images = generator_cfg["num_images"])
# print(len(images))

print(config.generator_cfg)