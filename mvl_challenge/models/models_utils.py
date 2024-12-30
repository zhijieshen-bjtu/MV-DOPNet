from mvl_challenge.models.wrapper_dop_net import WrapperDOPNet

def load_layout_model(cfg):
    """
    Load a layout model estimator and returns an instance of it
    """
    if cfg.model.ly_model == "DOPNet":
        # ! loading HorizonNet
        model = WrapperDOPNet(cfg)
    else:
        raise NotImplementedError("")

    return model
