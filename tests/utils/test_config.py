import json
from pathlib import Path

from omegaconf import OmegaConf

from neural_networks.utils.config import write_train_config


def test_write_train_config(tmp_path: Path):
    config = OmegaConf.create(
        {
            "model": {"name": "example_model", "layers": 5, "activation": "relu"},
            "training": {"batch_size": 32, "epochs": 10, "learning_rate": 0.001},
        }
    )
    out_file = tmp_path / "train_config.json"
    write_train_config(config, str(tmp_path))
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        written_config = json.load(f)

    expected_config = OmegaConf.to_container(config, resolve=True)
    assert written_config == expected_config
