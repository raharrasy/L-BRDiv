import hydra
import torch
from adhoc_train import AdhocTraining

@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(cfg):
    #args = vars(args)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    torch.set_num_threads(hydra_cfg.launcher.get("cpus_per_task", 4))
    model_trainer = AdhocTraining(cfg)
    model_trainer.run()

if __name__ == '__main__':
    run()
