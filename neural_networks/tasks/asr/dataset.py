import json

import soundfile
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            data_dict = json.load(f)
        self.sample_list = sorted(data_dict.items(), key=lambda x: x[1]["audio_length"], reverse=True)

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        _, sample_dict = self.sample_list[index]
        np_audio, _ = soundfile.read(sample_dict["audio_path"])  # (sample,)
        torch_audio = torch.from_numpy(np_audio).to(dtype=torch.float32)
        text = sample_dict["text"]
        sample = {"audio": torch_audio, "text": text}
        return sample
