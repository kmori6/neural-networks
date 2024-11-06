import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io.wavfile import write

from neural_networks.tasks.asr.dataset import CustomDataset


@pytest.fixture()
def json_tmp_file(tmp_path: Path) -> str:
    sample_rate = 16000
    text = "hello world"
    data = {}
    for i in range(2):
        audio_path = f"{tmp_path}/audio{i}.wav"
        audio = np.random.randn(sample_rate)  # 1 second of dummy audio
        write(audio_path, rate=16000, data=audio.astype(np.int16))  # 16kHz, 16-bit PCM
        data[f"utt_id{i}"] = {"audio_path": audio_path, "audio_length": len(audio), "text": text}
    json_tmp_path = f"{tmp_path}/data.json"
    with open(json_tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return json_tmp_path


def test_dataset(json_tmp_file: str):
    dataset = CustomDataset(json_tmp_file)
    assert len(dataset) == 2
    for i in range(len(dataset)):
        sample = dataset[i]
        audio = sample["audio"]
        text = sample["text"]
        isinstance(audio, torch.Tensor)
        assert text == "hello world"
