import math
import random
import argparse
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
import wandb

from model.modeling_tinylm import TinyLMForCausalLM, TinyLMargs
from dataset.dataset_pretrain import PretrainDataset


# =========================================================
# Utils
# =========================================================

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(update_step, total_steps, warmup_steps, base_lr, min_lr):
    if update_step < warmup_steps:
        return base_lr * update_step / warmup_steps

    progress = (update_step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine_decay


@torch.no_grad()
def evaluate(model, loader, device, autocast_ctx, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, (input_ids, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with autocast_ctx:
            loss, _, _ = model(input_ids=input_ids, labels=labels)

        tokens = labels.numel()
        total_loss += loss.item() * tokens
        total_tokens += tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser("Industrial TinyLM Trainer")

    # data
    parser.add_argument("--data_path", type=str, default="data/pretrain_data.bin")
    parser.add_argument("--val_ratio", type=float, default=0.01)

    # training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_seq_len", type=int, default=512)

    # misc
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="TinyLM")
    parser.add_argument("--exp_name", type=str, default="tinylm_run")

    args = parser.parse_args()

    # setup
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if device == "cuda"
        else nullcontext()
    )

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))

    # model
    model_args = TinyLMargs()
    model = TinyLMForCausalLM(model_args).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=(args.dtype == "float16"))

    # dataset
    train_ds = PretrainDataset(args.data_path, max_length=args.max_seq_len, split="train", val_ratio=args.val_ratio)
    val_ds = PretrainDataset(args.data_path, max_length=args.max_seq_len, split="val", val_ratio=args.val_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    total_update_steps = (len(train_loader) * args.epochs) // args.grad_accum_steps

    print(f"Total update steps: {total_update_steps}")

    # =====================================================
    # Training Loop
    # =====================================================

    update_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):

        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, (input_ids, labels) in enumerate(train_loader):

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with autocast_ctx:
                loss, _, aux_loss = model(input_ids=input_ids, labels=labels)
                total_loss = (loss + (aux_loss if aux_loss else 0.0)) / args.grad_accum_steps

            scaler.scale(total_loss).backward()

            if (step + 1) % args.grad_accum_steps == 0:

                # update step++
                update_step += 1

                # lr schedule
                lr = cosine_lr(
                    update_step,
                    total_update_steps,
                    args.warmup_steps,
                    args.lr,
                    args.min_lr,
                )

                for g in optimizer.param_groups:
                    g["lr"] = lr

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # logging
                if args.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "update_step": update_step
                    }, step=update_step)

                if update_step % 100 == 0:
                    print(f"Step {update_step} | loss {loss.item():.4f} | lr {lr:.2e}")

                # =====================================
                # Evaluation
                # =====================================
                if update_step % args.eval_interval == 0:

                    val_loss, val_ppl = evaluate(
                        model,
                        val_loader,
                        device,
                        autocast_ctx,
                        max_batches=200  # 控制验证成本
                    )

                    print(f"[Eval @ {update_step}] val_loss={val_loss:.4f} | ppl={val_ppl:.2f}")

                    if args.use_wandb:
                        wandb.log({
                            "val/loss": val_loss,
                            "val/ppl": val_ppl
                        }, step=update_step)

                    # best checkpoint
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), "best_model.pt")
                        print(f"New best model saved @ step {update_step}")

    # save final
    torch.save(model.state_dict(), "last_model.pt")
    print("Training finished.")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
