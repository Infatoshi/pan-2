"""Shard writer/reader roundtrip tests."""

import json

import numpy as np

from pan2.data.shards import MANIFEST_NAME, ShardDataset, ShardWriter, load_manifest


def _ramp_ep(t, h=4, base=0):
    img = np.zeros((t, h, h, 3), dtype=np.uint8)
    for i in range(t):
        img[i] = (base + i) % 250
    act = np.zeros((t, 25), dtype=np.float32)
    act[:, 3] = 1.0  # forward held, recognizable
    act[:, 23] = 0.2
    return img, act


def test_writer_roundtrip_two_shards(tmp_path):
    eps = [("a", 300), ("b", 200), ("c", 100)]
    srcs = {stem: _ramp_ep(t, base=i * 3) for i, (stem, t) in enumerate(eps)}
    w = ShardWriter(
        tmp_path, image_size=4, target_shard_bytes=300 * 4 * 4 * 3 + 10  # forces split
    )
    for stem, t in eps:
        img, act = srcs[stem]
        w.add_episode(img, act, stem)
    header = w.close()

    assert header["n_shards"] == 2
    assert header["total_frames"] == 600
    assert header["total_episodes"] == 3

    h2, segments = load_manifest(tmp_path)
    assert h2 == header
    # episodes never straddle shards
    assert [s["shard"] for s in segments] == [0, 1, 1]
    for s in segments:
        shard = np.load(tmp_path / f"shard-{s['shard']:05d}.frames.npy")
        act_shard = np.load(tmp_path / f"shard-{s['shard']:05d}.act.npy")
        got = shard[s["offset"] : s["offset"] + s["n_frames"]]
        np.testing.assert_array_equal(got, srcs[s["stem"]][0])
        got_act = act_shard[s["offset"] : s["offset"] + s["n_frames"]]
        np.testing.assert_array_equal(got_act, srcs[s["stem"]][1])
    # manifest parses as jsonl with header first
    first = json.loads(open(tmp_path / MANIFEST_NAME).readline())
    assert first["type"] == "header"


def test_shard_dataset_sampling_and_identity(tmp_path):
    img, act = _ramp_ep(240)
    w = ShardWriter(tmp_path, image_size=4, target_shard_bytes=1 << 30)
    w.add_episode(img, act, "ep0")
    w.close()

    ds = ShardDataset(
        tmp_path,
        context_len=32,
        action_chunk=10,
        image_size=4,
        min_goal_horizon=4,
        max_goal_horizon=40,
    )
    assert len(ds) == ds.windows_per_episode
    for _ in range(25):
        item = ds[0]
        last_ctx = int(item["frames"][-1, 0, 0, 0])
        goal_val = int(item["goal"][0, 0, 0])
        neg_val = int(item["neg"][0, 0, 0])
        gap = goal_val - last_ctx
        assert 4 <= gap <= 40
        assert neg_val - last_ctx > 40
        assert item["discrete"].shape == (10, 23)
        assert item["mouse"].shape == (10, 2)
        # forward column (idx 3 in 23-wide discrete) is held in source act
        assert float(item["discrete"][:, 3].min()) == 1.0


def test_shard_writer_rejects_mixed_act(tmp_path):
    import pytest

    img, act = _ramp_ep(240)
    w = ShardWriter(tmp_path, image_size=4)
    w.add_episode(img, act, "has_act")
    with pytest.raises(ValueError, match="act presence"):
        w.add_episode(img, None, "no_act")
