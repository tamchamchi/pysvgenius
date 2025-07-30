from argparse import Namespace
from pathlib import Path

from pysvgenius.common.config import Config
from pysvgenius.common.registry import registry


# Register paths in a central registry for easy access throughout the project
def setup_path():
    # Get the absolute path of the project directory (go up one level from the current file)
    project_dir = Path(__file__).resolve().parents[1]

    # Define subdirectories relative to the project directory
    data_dir = project_dir / "data"
    model_dir = project_dir / "models"
    config_dir = project_dir / "configs"
    
    registry.register_path("project_dir", str(project_dir))
    registry.register_path("data_dir", str(data_dir))
    registry.register_path("model_dir", str(model_dir))
    registry.register_path("config_dir", str(config_dir))
    registry.register_path("default_config_path", str(config_dir / "configs.yaml"))

# Define publicly accessible functions or objects when this module is imported
__all__ = ["load_config", "setup_path"]


def load_config(options=None):
    """
    Load and configure the application settings from the default config file.

    This function loads the main configuration file (configs.yaml) and allows
    runtime overrides through command-line style options. It's the primary
    way to initialize application configuration with custom parameters.

    Args:
        options (List[str], optional): List of configuration override options
            in the format "section.subsection.key=value". These options will
            override the corresponding values in the loaded config file.

            Examples of valid option formats:
            - "generator.sdxl-turbo.num_images=5"
            - "converter.vtracer.limit=20000"
            - "ranker.siglip.device=cuda:1"
            - "optimizer.diffvg.iterations=200"

            Defaults to None (no overrides).

    Returns:
        Config: A configuration object containing all loaded settings.
            The Config object provides:
            - .to_dict(): Convert config to dictionary
            - .generator_cfg: Generator-specific configuration
            - .converter_cfg: Converter-specific configuration
            - .ranker_cfg: Ranker-specific configuration
            - .optimizer_cfg: Optimizer-specific configuration

    Raises:
        FileNotFoundError: If the default config file (configs.yaml) is not found
            at the registered path.
        ValueError: If any of the provided options have invalid format or
            reference non-existent configuration keys.
        RuntimeError: If there are issues parsing the YAML config file or
            applying the option overrides.

    Example:
        Basic usage without overrides:
        >>> config = load_config()
        >>> print(config.to_dict())

        With runtime parameter overrides:
        >>> options = [
        ...     "generator.sdxl-turbo.num_images=8",
        ...     "generator.sdxl-turbo.seed=123",
        ...     "converter.vtracer.limit=15000"
        ... ]
        >>> config = load_config(options=options)
        >>> cfg_dict = config.to_dict()
        >>> print(f"Images to generate: {cfg_dict['generator']['sdxl-turbo']['num_images']}")

        Using config to initialize components:
        >>> config = load_config(["generator.sdxl-turbo.device=cuda:0"])
        >>> generator_cfg = config.generator_cfg['sdxl-turbo']
        >>> generator = SDXLTurboGenerator.from_config(generator_cfg)

    Note:
        - The config file path is automatically resolved from the registry
          (typically: {project_root}/configs/configs.yaml)
        - Options use dot notation to specify nested configuration keys
        - Boolean values should be specified as "true" or "false" strings
        - Numeric values are automatically converted to appropriate types
        - The function creates a new Config instance on each call

    Configuration File Structure:
        The expected YAML structure should contain top-level sections:

        ```yaml
        generator:
          sdxl-turbo:
            model_path: "stabilityai/sdxl-turbo"
            device: "cuda"
            num_images: 3
            seed: 42

        converter:
          vtracer:
            limit: 10000

        ranker:
          siglip:
            device: "cuda"

        optimizer:
          diffvg:
            iterations: 100
            device: "cuda"
        ```

    See Also:
        - Config: The configuration class for detailed API reference
        - registry.get_path(): For understanding path resolution
        - Component.from_config(): For using configs with components
    """
    
    if not registry.get_path("default_config_path"):
        setup_path() 
           
    args = Namespace(
        cfg_path=registry.get_path("default_config_path"),
        options=options if options is not None else [],
    )

    config = Config(args)

    return config


