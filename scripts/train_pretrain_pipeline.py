#!/usr/bin/env python3
"""Pretrain using GPU-resident pipelined loader (~10GB ring)."""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from pan2.config import load_config
from pan2.data.gpu_pipeline import PipelineConfig, PipelinedGpuPretrainLoader
from pan2.train.loop import (
    build_state,
    eval_contrastive,
    load_ckpt,
    save_ckpt,
    train_steps,
)


def _latest_ckpt(ckpt_dir: Path) -> Path | None:
    """Newest pretrain checkpoint by step number (pretrain_step*.pt)."""
    best: tuple[int, Path] | None = None
    for p in ckpt_dir.glob("pretrain_step*.pt"):
        try:
            step = int(p.stem.removeprefix("pretrain_step"))
        except ValueError:
            continue
        if best is None or step > best[0]:
            best = (step, p)
    return best[1] if best else None


def gen_val(val_loader: PipelinedGpuPretrainLoader):
    while True:
        yield next(val_loader)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--budget-gb", type=float, default=10.0)
    p.add_argument("--producers", type=int, default=8)
    p.add_argument("--prefer-source", default="auto",
                   choices=["auto", "shard", "npy", "mp4", "pack"])
    p.add_argument("--raw-dir", default="/data/pan-2/raw")
    p.add_argument("--episodes-dir", default="/data/pan-2/episodes")
    p.add_argument("--pack-index", default="",
                   help="pack_index.npz from scripts/build_pack_index.py "
                        "(required when --prefer-source pack)")
    p.add_argument("--pack-minecraft-only", action="store_true",
                   help="restrict pack items to minecraft-flagged episodes")
    p.add_argument("--native-fps", type=float, default=None,
                   help="source fps for seek/horizon units (default 10 for "
                        "pack, 20 otherwise)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override config train.max_steps (TOTAL steps incl. "
                        "any resumed progress)")
    p.add_argument("--refresh-prob", type=float, default=None,
                   help="override ring slot refresh probability")
    p.add_argument("--resume", default="", choices=["", "auto"],
                   help="'auto': resume model+optim from the newest "
                        "pretrain_step*.pt in ckpt_dir (freeze-tolerant "
                        "relaunch; no-op when none exists)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    cfg.train.stage = "pretrain"
    cfg.train.synthetic = False
    data_sub = cfg.model.frame_subsample
    cfg.model.frame_subsample = 1  # already_subsampled in ring
    state = build_state(cfg)
    cfg.model.frame_subsample = data_sub  # restore for pipeline config below

    if args.resume == "auto":
        ck = _latest_ckpt(Path(cfg.train.ckpt_dir))
        if ck is not None:
            load_ckpt(state, ck)
            print(f"resumed from {ck} at step {state.step}")
        else:
            print("resume auto: no checkpoint found, starting fresh")

    # max_steps is the TOTAL step target: a resumed run finishes the
    # remainder, and rerunning after completion exits fast before any
    # producers spin up (idempotent for the wrapper's retry loop).
    if state.step >= cfg.train.max_steps:
        print(
            f"already at step {state.step} >= max_steps {cfg.train.max_steps}, "
            "nothing to do"
        )
        print("pretrain done")
        return

    native_fps = args.native_fps
    if native_fps is None:
        native_fps = 10.0 if args.prefer_source == "pack" else 20.0

    pcfg = PipelineConfig(
        raw_dir=args.raw_dir,
        episodes_dir=args.episodes_dir,
        batch_size=cfg.train.batch_size,
        context_len=cfg.model.context_len,
        frame_subsample=data_sub,
        image_size=cfg.model.image_size,
        budget_gb=args.budget_gb,
        num_producers=args.producers,
        prefer_source=args.prefer_source,
        device=cfg.train.device,
        min_goal_horizon=cfg.train.min_goal_horizon,
        max_goal_horizon=cfg.train.max_goal_horizon,
        n_hard_negatives=cfg.train.n_hard_negatives,
        native_fps=native_fps,
        pack_index=args.pack_index,
        pack_minecraft_only=args.pack_minecraft_only,
    )
    if args.refresh_prob is not None:
        pcfg.refresh_prob = args.refresh_prob
    pcfg.heldout_frac = cfg.train.heldout_frac
    loader = PipelinedGpuPretrainLoader(pcfg)

    # Held-out val loader: same episode hash split, frozen ring (refresh=0)
    # so every checkpoint scores the same windows. Ring must hold >= 2*bs
    # slots (batches sample distinct slots); size the budget to the batch.
    val_loader = None
    if cfg.train.eval_every > 0 and cfg.train.heldout_frac > 0:
        from pan2.data.gpu_pipeline import _subsample_indices

        t_slot = len(_subsample_indices(cfg.model.context_len, data_sub)) + 1 + max(
            1, cfg.train.n_hard_negatives
        )
        val_budget_gb = max(
            0.3,
            (3 * cfg.train.batch_size * t_slot * 3 * cfg.model.image_size**2)
            / float(2**30),
        )
        vpcfg = dataclasses.replace(
            pcfg,
            split="val",
            budget_gb=val_budget_gb,
            num_producers=2,
            refresh_prob=0.0,
        )
        val_loader = PipelinedGpuPretrainLoader(vpcfg)
        print(
            f"[eval] held-out pool={len(val_loader.items)} episodes, "
            f"{cfg.train.eval_batches} batches every {cfg.train.eval_every} steps"
        )

    def gen():
        while True:
            yield next(loader)

    remaining = cfg.train.max_steps - state.step
    try:
        while remaining > 0:
            chunk = min(cfg.train.log_every, remaining)
            logs = train_steps(state, cfg, gen(), n_steps=chunk)
            st = loader.status()
            print(
                f"step={state.step} loss={logs[-1]['loss']:.4f} "
                f"ring={st['ready']}/{st['capacity']} fills={st['fills']} err={st['errors']}"
            )
            remaining -= chunk
            if val_loader is not None and state.step % cfg.train.eval_every == 0:
                ev = eval_contrastive(state, cfg, gen_val(val_loader), cfg.train.eval_batches)
                chance = 1.0 / (cfg.train.batch_size + cfg.train.n_hard_negatives)
                print(
                    f"eval step={state.step} val_loss={ev['val_loss']:.4f} "
                    f"val_acc={ev['val_acc']:.4f} (chance={chance:.4f})"
                )
            if state.step % cfg.train.ckpt_every == 0:
                save_ckpt(state, Path(cfg.train.ckpt_dir) / f"pretrain_step{state.step}.pt")
        # Step-stamped final save even off the ckpt_every grid, so a
        # rerun's --resume auto lands exactly here and exits fast.
        if val_loader is not None and state.step % cfg.train.eval_every != 0:
            ev = eval_contrastive(state, cfg, gen_val(val_loader), cfg.train.eval_batches)
            chance = 1.0 / (cfg.train.batch_size + cfg.train.n_hard_negatives)
            print(
                f"eval step={state.step} val_loss={ev['val_loss']:.4f} "
                f"val_acc={ev['val_acc']:.4f} (chance={chance:.4f}) [final]"
            )
        save_ckpt(state, Path(cfg.train.ckpt_dir) / f"pretrain_step{state.step}.pt")
        save_ckpt(state, Path(cfg.train.ckpt_dir) / "pretrain_last.pt")
        print("pretrain done")
    finally:
        loader.stop()
        if val_loader is not None:
            val_loader.stop()


if __name__ == "__main__":
    main()
