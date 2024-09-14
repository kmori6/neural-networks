import torch

from neural_networks.modules.frontend import Frontend


def test_frontend():
    max_length = 16000
    n_mels = 80
    hop_length = 128
    module = Frontend(hop_length=hop_length, n_mels=n_mels)
    speech = torch.randn(2, max_length)
    speech_length = torch.tensor([max_length // 2, max_length])
    output = module(speech, speech_length)
    assert isinstance(output, tuple) and len(output) == 2
    assert output[0].shape == (2, 1 + max_length // hop_length, n_mels)
    assert output[1].shape == (2, 1 + max_length // hop_length)
    assert torch.equal(output[1].sum(-1), 1 + speech_length // hop_length)
