from networks.lightglue.superpoint import SuperPoint as _SuperPoint


class SuperPoint(_SuperPoint):
    # Registered in hloc.extractors so dynamic_load finds it.
    # _SuperPoint inherits from networks...BaseModel which is compatible.
    pass
