from omegaconf import OmegaConf

from src.common.registry import registry


class Config:
    def __init__(self, args):

        self.config = {}

        self.args = args

        registry.register("configuration", self)

        user_config = self._build_opt_list(self.args.options)

        config = OmegaConf.load(self.args.cfg_path)

        self.config = OmegaConf.merge(
            config, user_config
        )

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)

        return OmegaConf.from_dotlist(opts_dot_list)

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    def get_config(self):
        return self.config

    @property
    def generator_cfg(self):
        return self.config.generator

    @property
    def converter_cfg(self):
        return self.config.converter

    @property
    def ranker_cfg(self):
        return self.config.ranker

    @property
    def optimizer_cfg(self):
        return self.config.optimizer

    def to_dict(self):
        return OmegaConf.to_container(self.config)


def node_to_dict(node):
    return OmegaConf.to_container(node)
