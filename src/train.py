"""Train the mapping for one stage.

launch.py submits one job per stage and passes every path as a flag.
The job loads the previous stage's mapping (--load-from), trains it, and
saves the checkpoint with the lowest validation loss (--save-to).

One GPU:    python -m src.train --stage 2 ...
Many GPUs:  torchrun --nproc-per-node 4 -m src.train --stage 1 ...
"""
from __future__ import annotations

import argparse
import os
import random

import torch
import torch.distributed as dist

from src.loaders import make_replay_loader, make_train_loaders
from src.model import MappedVLM

SEED = 42
LEARNING_RATE = 2e-5
EPOCHS = 3
GRAD_CLIP = 1.0
# Rows per GPU per batch, and batches per update. Stage 1 runs on 4 GPUs
# (2 x 3 x 4 = 24 rows per update). Stages 2 and 3 run on 1 GPU (4 x 8 = 32).
BATCH_SIZE = {1: 2, 2: 4, 3: 4}
GRAD_ACCUM = {1: 3, 2: 8, 3: 8}
REPLAY_STAGE = 3
REPLAY_EVERY = 3
# Replay's 512-token math targets do not fit on a 40 GB A100 without this.
CHECKPOINT_ACTIVATIONS_STAGES = {3}
LOG_EVERY = 50

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the mapping for one stage.")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--load-from", default="", help="Mapping checkpoint to start from. Empty: start fresh.")
    parser.add_argument("--save-to", required=True, help="Where to save the best mapping checkpoint (.pt).")
    parser.add_argument("--translation-replay", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-steps", type=int, default=0, help="Smoke test: stop each epoch after this many updates.")
    train(parser.parse_args())


def train(args) -> None:
    rank, world_size, device = init_distributed()
    set_seed(args.seed)
    model = build_model(args.backbone, args.encoder, device)
    if args.load_from:
        load_mapping(model, args.load_from)
    if world_size > 1:
        # Every GPU must start from the same mapping.
        for p in model.mapping.parameters():
            dist.broadcast(p.data, src=0)
    if args.stage in CHECKPOINT_ACTIVATIONS_STAGES:
        model.backbone.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.backbone.model.config.use_cache = False

    batch_size = BATCH_SIZE[args.stage]
    train_loader, val_loader = make_train_loaders(args.stage, model, batch_size, args.seed, rank, world_size)
    replay = None
    if args.stage == REPLAY_STAGE:
        replay = replay_batches(make_replay_loader(model, args.translation_replay, batch_size))
    optimizer = torch.optim.AdamW(model.mapping.parameters(), lr=LEARNING_RATE)
    run = init_wandb(args) if rank == 0 else None

    best_val, step = float("inf"), 0
    for epoch in range(EPOCHS):
        step = train_epoch(model, train_loader, replay, optimizer, args, world_size, run, step)
        val_loss = validate(model, val_loader, world_size, args.max_steps)
        if rank == 0:
            print(f"epoch {epoch} | step {step} | val_loss {val_loss:.4f}", flush=True)
            if run:
                run.log({"eval/loss": val_loss}, step=step)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(args.save_to, model, val_loss)
                print(f"saved best checkpoint -> {args.save_to}", flush=True)

    if run:
        run.finish()
    if world_size > 1:
        dist.destroy_process_group()


def init_distributed() -> tuple[int, int, torch.device]:
    """Return (rank, world_size, device). torchrun sets LOCAL_RANK; a plain python run uses one GPU."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, torch.device("cuda")
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(backbone: str, encoder: str, device: torch.device) -> MappedVLM:
    return MappedVLM(backbone, encoder).to(device)


def load_mapping(model: MappedVLM, path: str) -> None:
    model.mapping.load_state_dict(torch.load(path, map_location="cpu")["mapping"])


def save_checkpoint(path: str, model: MappedVLM, val_loss: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"mapping": model.mapping.state_dict(), "val_loss": val_loss}, path)


def replay_batches(loader):
    """Replay batches without end. The loader reshuffles on every pass."""
    while True:
        yield from loader


def train_epoch(model, loader, replay, optimizer, args, world_size, run, step) -> int:
    """One pass over the training rows. Returns the update count so far."""
    model.train()
    accum = GRAD_ACCUM[args.stage]
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        loss = model(batch)
        (loss / accum).backward()
        logs = {"train/loss": loss.item()}
        if replay is not None and i % REPLAY_EVERY == 0:
            replay_loss = model(next(replay))
            (replay_loss / accum).backward()
            logs["train/replay_loss"] = replay_loss.item()

        if (i + 1) % accum == 0:
            update(model, optimizer, world_size)
            step += 1
            if run:
                run.log(logs, step=step)
                if step % LOG_EVERY == 0:
                    print(f"step {step} | " + " | ".join(f"{k} {v:.4f}" for k, v in logs.items()), flush=True)
            if args.max_steps and step % args.max_steps == 0:
                break
    return step


def update(model, optimizer, world_size: int) -> None:
    """Average the mapping's gradients across GPUs, clip them, and take one optimizer step."""
    params = list(model.mapping.parameters())
    if world_size > 1:
        for p in params:
            dist.all_reduce(p.grad)
            p.grad /= world_size
    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def validate(model, loader, world_size: int, max_steps: int) -> float:
    """Mean loss over the validation rows. Each GPU scores its own slice, then the totals are combined."""
    model.eval()
    totals = torch.zeros(2, device=model.backbone.device)  # [sum of batch losses, number of batches]
    for i, batch in enumerate(loader):
        totals += torch.stack([model(batch).float(), torch.ones((), device=totals.device)])
        if max_steps and i + 1 >= max_steps:
            break
    if world_size > 1:
        dist.all_reduce(totals)
    return (totals[0] / totals[1]).item()

if __name__ == "__main__":
    main()
