class Registry:
    mapping = {
        "generator": {},
        "convertor": {},
        "ranker": {},
        "optimizer": {},
        "state": {},
        "paths": {},
    }

    @classmethod
    def register_generator(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from src.common.registry import registry
        """
        def wrap(gen_cls):
            from src.generator.base import IGenerator

            assert issubclass(gen_cls, IGenerator), (
                "All generators must inherit IGenerator class"
            )

            if name in cls.mapping["generator"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["generator"][name]
                    )
                )
            cls.mapping["generator"][name] = gen_cls
            return gen_cls

        return wrap
    

    @classmethod
    def register_path(cls, name, path):
        r"""Register a path to registry with key 'name'

        Args:
            name: Key with which the path will be registered.

        Usage:

            from src.common.registry import registry
        """
        assert isinstance(path, str), "All path must be str."
        if name in cls.mapping["paths"]:
            raise KeyError("Name '{}' already registered.".format(name))
        cls.mapping["paths"][name] = path

    @classmethod
    def register(cls, name, obj):
        r"""Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from src.common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[path[-1]] = obj

    @classmethod
    def get_generator_class(cls, name):
        return cls.mapping["generator"].get(name, None)

    @classmethod
    def list_generator(cls):
        return sorted(cls.mapping["generator"].keys())

    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths"].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for MMF's
                               internal operations. Default: False
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]

        for subname in name:
            value = value.get(subname, default)

            if value is default:
                break
        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].warning(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from src.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()
