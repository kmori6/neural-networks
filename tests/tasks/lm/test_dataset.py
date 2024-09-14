import json
from pathlib import Path

import pytest

from neural_networks.tasks.lm.dataset import CustomDataset


@pytest.fixture()
def json_tmp_file(tmp_path: Path) -> str:
    data = {}
    for i in range(2):
        sample_id = f"sample_id{i}"
        text = "text"
        data[sample_id] = {"text": text}
    json_tmp_path = f"{tmp_path}/data.json"
    with open(json_tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return json_tmp_path


def test_dataset(json_tmp_file: str):
    dataset = CustomDataset(json_tmp_file)
    assert len(dataset) == 2
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample["text"] == "text"
