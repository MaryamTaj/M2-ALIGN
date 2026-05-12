# coding=utf-8
"""Stage 1 training script for MindMerger (NLLB encoder → mapping → Qwen3-VL)."""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
from datetime import datetime

import deepspeed
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, NllbTokenizer

from evaluation import evaluate_ppl
from modeling_mindmerger import MindMerger
from tools.deepspeed_config import get_train_ds_config
from tools.input_features import llm_input_features, mt_input_features
from tools.read_datasets import MathDataset, read_lego, read_math_train, read_nllb, read_x_csqa_train, read_xnli_train
from tools.utils import save_model, set_seed

try:
    import wandb
except ImportError:
    wandb = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> logging.Logger:
    """Create a logger that writes to stdout and a timestamped file in *log_dir*.

    Args:
        log_dir: Directory in which to create the log file (created if absent).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"training_{timestamp}.log")

    logger = logging.getLogger("stage1_training")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def _load_wandb_key_from_tokens() -> bool:
    """Load WANDB_API_KEY from a ``.tokens`` file if present.

    Searches the current working directory and its parent for a file
    named ``.tokens`` containing lines of the form ``WANDB_API_KEY=...``
    or ``export WANDB_API_KEY=...``.  Sets the environment variable if
    a non-empty value is found.

    Returns:
        ``True`` if a key was found and set, ``False`` otherwise.
    """
    for candidate in (
        os.path.join(os.getcwd(), ".tokens"),
        os.path.join(os.path.dirname(os.getcwd()), ".tokens"),
    ):
        if not os.path.isfile(candidate):
            continue
        with open(candidate, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() != "WANDB_API_KEY":
                    continue
                value = value.strip().strip('"').strip("'")
                if value:
                    os.environ["WANDB_API_KEY"] = value
                    return True
    return False


def _init_wandb_or_disable(args, config: dict) -> bool:
    """Initialise Weights & Biases with online/offline fallback.

    Never raises; returns ``False`` if W&B cannot be initialised so
    callers can safely skip all ``wandb.*`` calls.

    Args:
        args: Parsed argument namespace with attributes ``wandb_mode``,
            ``wandb_project``, ``wandb_run_name``, and
            ``wandb_init_timeout``.
        config: Hyperparameter dict logged to the W&B run.

    Returns:
        ``True`` if W&B was successfully initialised, ``False`` otherwise.
    """
    if wandb is None:
        return False

    requested_mode = (args.wandb_mode or "auto").lower()
    if requested_mode not in {"auto", "online", "offline"}:
        print(f"Invalid wandb_mode={args.wandb_mode}; expected auto|online|offline. Disabling W&B.")
        return False

    if requested_mode in {"auto", "online"} and not os.environ.get("WANDB_API_KEY"):
        _load_wandb_key_from_tokens()

    if requested_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=config,
                settings=wandb.Settings(init_timeout=args.wandb_init_timeout),
            )
            print("Initialized Weights & Biases in offline mode.")
            return True
        except Exception as exc:
            print(f"wandb offline init failed ({exc}); disabling Weights & Biases logging.")
            return False

    if not os.environ.get("WANDB_API_KEY"):
        if requested_mode == "online":
            print("WANDB_API_KEY not found; disabling Weights & Biases logging.")
            return False
        os.environ["WANDB_MODE"] = "offline"
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=config,
                settings=wandb.Settings(init_timeout=args.wandb_init_timeout),
            )
            print("WANDB_API_KEY not found; initialized W&B in offline mode.")
            return True
        except Exception as exc:
            print(f"wandb offline init failed ({exc}); disabling Weights & Biases logging.")
            return False

    try:
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config,
            settings=wandb.Settings(init_timeout=args.wandb_init_timeout),
        )
        return True
    except Exception as exc:
        if requested_mode == "online":
            print(f"wandb online init failed ({exc}); disabling Weights & Biases logging.")
            return False
        try:
            os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=config,
                settings=wandb.Settings(init_timeout=args.wandb_init_timeout),
            )
            print(f"wandb online init failed ({exc}); fell back to offline mode.")
            return True
        except Exception as offline_exc:
            print(
                f"wandb init failed ({exc}); offline fallback failed ({offline_exc}); "
                "disabling Weights & Biases logging."
            )
            return False


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(args, logger: logging.Logger) -> None:
    """Run the Stage 1 mapping-layer training loop.

    Loads the dataset for the requested *task* and *stage_name*, builds the
    :class:`MindMerger` model, wraps it with DeepSpeed, then trains for
    *epoch_num* epochs with periodic perplexity-based checkpointing.

    Args:
        args: Parsed argument namespace from :func:`argparse.ArgumentParser`.
        logger: Logger for training progress and checkpoint messages.
    """
    llm_path = args.llm_path
    mt_path = args.mt_path

    train_num = args.train_num
    stage_name = args.stage_name
    task = args.task
    augmentation = args.augmentation
    save_name = args.save_name
    result_path_base = f"./results/{save_name}/{task}/{stage_name}/"
    output_model_path_base = f"./outputs/{save_name}/{task}/{stage_name}/"

    languages: list[str]
    if stage_name == "mapping":
        if "nllb_corpus" in task:
            languages = [x.strip() for x in args.nllb_languages.split(",") if x.strip()]
            train_set = read_nllb(args.nllb_data_dir, train_num, languages)
        elif "math" in task:
            languages = ["Bengali", "Thai", "Swahili", "Japanese", "Chinese", "German", "French", "Russian", "Spanish"]
            train_set = read_lego(train_num, languages)
        elif "csqa" in task:
            languages = ["Urdu", "Hindi", "Swahili", "Japanese", "Vietnamese", "Polish", "Chinese",
                         "Flemish", "Russian", "Italian", "German", "Portuguese", "French", "Spanish", "Arabic"]
            train_set = read_lego(train_num, languages)
        else:
            languages = ["Swahili", "Urdu", "Hindi", "Thai", "Arabic", "Turkish", "Greek",
                         "Vietnamese", "Chinese", "Russian", "Bulgarian", "German", "French", "Spanish"]
            train_set = read_lego(train_num, languages)
        task = "translation"
    else:
        if "math" in task:
            train_set = read_math_train(train_num)
        elif "csqa" in task:
            train_set = read_x_csqa_train()
        else:
            train_set = read_xnli_train()
        languages = []

    val_set = train_set[: args.val_size]
    train_set = train_set[args.val_size:]
    train_set = MathDataset(train_set, task)
    val_set = MathDataset(val_set, task)

    lr = args.lr
    epoch_num = args.epoch_num
    max_seq_len = args.max_seq_len
    max_gen_len = args.max_gen_len
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    train_micro_batch_size_per_gpu = args.train_micro_batch_size_per_gpu
    gpu_num = torch.cuda.device_count()
    gradient_accumulation = train_batch_size // (train_micro_batch_size_per_gpu * gpu_num)
    assert train_micro_batch_size_per_gpu * gpu_num * gradient_accumulation == train_batch_size
    ds_config = get_train_ds_config(train_batch_size, train_micro_batch_size_per_gpu, lr, gradient_accumulation)

    os.makedirs(output_model_path_base, exist_ok=True)
    os.makedirs(result_path_base, exist_ok=True)

    if "nllb-200" in mt_path or "nllb" in mt_path:
        tokenizer_m2m = NllbTokenizer.from_pretrained(mt_path)
    else:
        tokenizer_m2m = AutoTokenizer.from_pretrained(mt_path, use_fast=False)

    try:
        tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    except (ValueError, ImportError) as e:
        logger.warning("Falling back to slow LLM tokenizer for %s: %s", llm_path, e)
        tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, use_fast=False)

    tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"

    config_summary = {
        "llm_path": llm_path,
        "mt_path": mt_path,
        "lr": lr,
        "epoch_num": epoch_num,
        "gradient_accumulation": gradient_accumulation,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "max_seq_len": max_seq_len,
        "max_gen_len": max_gen_len,
        "train_batch_size": train_batch_size,
        "result_path": result_path_base,
        "output_model_path": output_model_path_base,
        "languages": languages,
    }
    logger.info("Training config:\n%s", json.dumps(config_summary, indent=2))

    use_wandb = args.use_wandb and args.local_rank == 0
    if use_wandb:
        use_wandb = _init_wandb_or_disable(args, {
            "llm_path": llm_path,
            "mt_path": mt_path,
            "stage_name": stage_name,
            "task": task,
            "train_num_per_language": train_num,
            "train_batch_size": train_batch_size,
            "lr": lr,
            "max_seq_len": max_seq_len,
            "max_gen_len": max_gen_len,
            "languages": languages,
        })

    if stage_name != "mapping" and args.init_checkpoint is None:
        args.init_checkpoint = f"./outputs/{save_name}/{task}/mapping/pytorch_model.bin"

    model = MindMerger(
        mt_path, llm_path, max_gen_len,
        tokenizer_llm.bos_token_id,
        tokenizer_llm.pad_token_id,
    )
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        model.mapping.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info("Mapping layer initialised from: %s", args.init_checkpoint)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, _, __ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=parameters,
        training_data=None,
    )

    train_sampler = DistributedSampler(train_set)
    val_sampler = SequentialSampler(val_set)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_micro_batch_size_per_gpu,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=1,
        drop_last=False,
    )

    global_rank = torch.distributed.get_rank()
    best_perplexity = evaluate_ppl(
        model, val_loader, tokenizer_llm, tokenizer_m2m,
        max_seq_len, max_gen_len, langs_map, augmentation,
    )
    logger.info("Initial validation perplexity: %.4f", best_perplexity)
    if use_wandb:
        wandb.log({"eval/perplexity": best_perplexity, "train/global_step": 0})

    eval_step = 2000
    global_step = 0
    for epoch in range(epoch_num):
        model.train()
        tr_loss, nb_tr_steps = 0.0, 0
        step_count = 0
        step_trange = tqdm(train_loader)
        for train_step in step_trange:
            sources = train_step["source"]
            prompts = train_step["prompt"]
            targets = train_step["target"]
            source_languages = train_step["source_language"]

            input_ids_m2m, attention_mask_m2m = mt_input_features(
                sources, tokenizer_m2m, max_seq_len, source_languages, langs_map
            )
            labels, mask_label = llm_input_features(
                targets, tokenizer_llm, max_gen_len, add_bos_token=False, add_eos_token=True
            )

            input_ids_prompt, mask_prompt = None, None
            if augmentation:
                input_ids_prompt, mask_prompt = llm_input_features(
                    prompts, tokenizer_llm, max_gen_len, add_bos_token=False, add_eos_token=False
                )

            loss = model(
                input_ids_m2m, attention_mask_m2m,
                input_ids_prompt=input_ids_prompt, mask_prompt=mask_prompt,
                labels=labels, mask_label=mask_label,
            )
            loss = loss.mean()
            tr_loss += loss.item()
            nb_tr_steps += 1
            model.backward(loss)
            model.step()

            avg_loss = tr_loss / nb_tr_steps
            step_trange.set_postfix_str(f"Epoch:{epoch} loss:{avg_loss:.4f}")
            global_step += 1
            if use_wandb:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                })

            if step_count % eval_step == 0 and step_count > 0:
                perplexity = evaluate_ppl(
                    model, val_loader, tokenizer_llm, tokenizer_m2m,
                    max_seq_len, max_gen_len, langs_map, augmentation,
                )
                logger.info("Step %d | val_ppl=%.4f", global_step, perplexity)
                if use_wandb:
                    wandb.log({"eval/perplexity": perplexity, "train/global_step": global_step})
                if global_rank == 0 and perplexity < best_perplexity:
                    best_perplexity = perplexity
                    save_model(output_model_path_base, model.mapping)
                    logger.info("New best checkpoint saved (ppl=%.4f)", best_perplexity)
            step_count += 1

        perplexity = evaluate_ppl(
            model, val_loader, tokenizer_llm, tokenizer_m2m,
            max_seq_len, max_gen_len, langs_map, augmentation,
        )
        logger.info("Epoch %d complete | val_ppl=%.4f", epoch, perplexity)
        if use_wandb:
            wandb.log({
                "eval/perplexity": perplexity,
                "train/epoch": epoch + 1,
                "train/global_step": global_step,
            })
        if global_rank == 0 and perplexity < best_perplexity:
            best_perplexity = perplexity
            save_model(output_model_path_base, model.mapping)
            logger.info("New best checkpoint saved (ppl=%.4f)", best_perplexity)

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: train the MindMerger mapping layer.")
    parser.add_argument("--llm_path", type=str, default="../LLMs/Llama-2-7b-hf/")
    parser.add_argument("--mt_path", type=str, default="../LLMs/mt5-xl/")
    parser.add_argument("--save_name", type=str, default="MindMerger")
    parser.add_argument("--task", type=str, default="translation")
    parser.add_argument("--stage_name", type=str, default="mapping")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch_num", type=int, default=3)
    parser.add_argument("--train_num", type=int, default=100000)
    parser.add_argument("--train_batch_size", type=int, default=24)
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--val_size", type=int, default=3000)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--augmentation", type=ast.literal_eval, default=False)
    parser.add_argument("--nllb_data_dir", type=str, default="./data/nllb")
    parser.add_argument(
        "--nllb_languages", type=str, default="French",
        help="Comma-separated source languages for task nllb_corpus.",
    )
    parser.add_argument("--use_wandb", type=ast.literal_eval, default=False)
    parser.add_argument("--wandb_project", type=str, default="mindmerger-stage1")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="auto")
    parser.add_argument("--wandb_init_timeout", type=int, default=90)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(0)

    logger = setup_logging(os.path.join(os.path.dirname(__file__), "logs"))

    langs_map_nllb = {
        "Swahili": "swh_Latn",
        "Yoruba": "yor_Latn",
        "Wolof": "wol_Latn",
        "French": "fra_Latn",
    }
    langs_map_m2m = {
        "English": "en", "Swahili": "sw", "Chinese": "zh", "Bengali": "bn",
        "German": "de", "Spanish": "es", "French": "fr", "Japanese": "ja",
        "Russian": "ru", "Thai": "th", "Greek": "el", "Telugu": "te",
        "Arabic": "ar", "Bulgarian": "bg", "Croatian": "hr", "Hungarian": "hu",
        "Italian": "it", "Lithuanian": "lt", "Macedonian": "mk", "Polish": "pl",
        "Portuguese": "pt", "Albanian": "sq", "Serbian": "sr", "Turkish": "tr",
        "Vietnamese": "vi", "Hindi": "hi", "Flemish": "nl", "Urdu": "ur",
    }

    langs_map = langs_map_nllb if "nllb" in args.mt_path else langs_map_m2m

    main(args, logger)
