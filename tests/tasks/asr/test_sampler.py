from neural_networks.tasks.asr.sampler import LengthBucketSampler


def test_shuffle_sampler():
    num_samples = 20
    batch_size = 3
    remainder = num_samples % batch_size
    result = list(LengthBucketSampler(num_samples, batch_size, shuffle=True))

    assert result[:batch_size] == [0, 1, 2]
    assert result[-remainder:] == [18, 19]

    middle_result = result[batch_size:-remainder]
    middle_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    assert sorted(middle_result) == middle_list
    assert middle_result != middle_list


def test_no_shuffle_sampler():
    num_samples = 20
    batch_size = 3
    result = list(LengthBucketSampler(num_samples, batch_size, shuffle=False))

    assert result == list(range(num_samples))
