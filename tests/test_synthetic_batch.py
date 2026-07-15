from pan2.data.synthetic import SyntheticGoalDataset, synthetic_batch


def test_synthetic_batch_keys():
    b = synthetic_batch(3, 8, 64, 4, 23, uint8=True)
    assert b["frames"].shape == (3, 8, 3, 64, 64)
    assert b["frames"].dtype.itemsize == 1
    assert b["goal"].shape == (3, 3, 64, 64)
    assert b["discrete"].shape == (3, 4, 23)


def test_dataset_len():
    ds = SyntheticGoalDataset(length=10, context_len=8, image_size=64, uint8=True)
    assert len(ds) == 10
    item = ds[0]
    assert item["frames"].shape[0] == 8
