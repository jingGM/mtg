import torch
from src.trainer import Trainer
from src.configs import get_configs


if __name__ == "__main__":
    cfgs = get_configs()
    trainer = Trainer(cfgs=cfgs)
    torch.autograd.set_detect_anomaly(True)
    trainer.run()
    torch.autograd.set_detect_anomaly(False)
