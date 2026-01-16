from models import CNO2d
from models.FNO import FNO1d, FNO2d, FactorizedFNO2d

def create_model(model_conf, device):
    if model_conf['model_type'] == "FNO1d":
        fno_architecture={
            "modes": model_conf['modes'],
            "width": model_conf['width'],
            "n_layers": model_conf['n_layers'],
            "retrain": model_conf['retrain']
        }
        return FNO1d(
            fno_architecture=fno_architecture,
            device=device
        ).to(device)
    if model_conf['model_type'] == "FNO2d":
        fno_architecture={
            "modes": model_conf['modes'],
            "width": model_conf['width'],
            "n_layers": model_conf['n_layers'],
            "retrain": model_conf['retrain'],
            "padding": model_conf['padding'],
            "include_grid": model_conf['include_grid'],
        }
        return FNO2d(
            fno_architecture=fno_architecture,
            in_channels=model_conf["in_channels"],
            out_channels=model_conf["out_channels"],
            device=device
        ).to(device)
    if model_conf['model_type'] == "FFNO2d":
        fno_architecture={
            "modes": model_conf['modes'],
            "width": model_conf['width'],
            "n_layers": model_conf['n_layers'],
            "retrain": model_conf['retrain'],
            "padding": model_conf['padding'],
            "include_grid": model_conf['include_grid'],
        }
        return FactorizedFNO2d(
            fno_architecture=fno_architecture,
            in_channels=model_conf["in_channels"],
            out_channels=model_conf["out_channels"],
            device=device
        ).to(device)
    if model_conf['model_type'] == "CNO2d":
        return CNO2d.CNO(
            in_dim=model_conf["in_channels"],
            out_dim=model_conf["out_channels"],
            in_size=model_conf['in_size'],
            out_size=model_conf.get('out_size', 1),
            N_layers=model_conf['n_layers']).to(device)
    return None