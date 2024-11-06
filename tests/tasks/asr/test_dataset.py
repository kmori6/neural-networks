import json
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_networks.tasks.asr.dataset import CustomDataset


@pytest.fixture()
def json_tmp_file(tmp_path: Path) -> str:
    sample_rate = 16000
    text = "hello world"
    data = {}
    for i in range(2):
        audio_path = f"{tmp_path}/audio{i}.wav"
        audio = np.random.randn(sample_rate)  # 1 second of dummy audio
        with wave.open(audio_path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)  # 2 bytes (16 bits)
            f.setframerate(sample_rate)
            f.writeframes(audio.tobytes())
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
