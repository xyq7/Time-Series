class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get_module(self, name):
        if name not in self.module_dict:
            print("no such module function, missing importing line ", name)
            return None
        else:
            return self.module_dict[name]

    def _register_module(self, cls):
        name = cls.__name__
        if name in self._module_dict:
            raise ValueError("repeated registered a module", name)
        self._module_dict[name] = cls

    def register(self, cls):
        self._register_module(cls)
        return cls


MODEL_ZONE = Registry("model")