import json

from omegaconf import DictConfig, OmegaConf


def write_train_config(config: DictConfig, out_dir: str):
    with open(f"{out_dir}/train_config.json", "w", encoding="utf-8") as f:
        config_dict = OmegaConf.to_container(config, resolve=True)
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
