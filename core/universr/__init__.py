def __getattr__(name):
    if name == "UniverSR":
        from universr.inference import UniverSR
        return UniverSR
    raise AttributeError(name)
