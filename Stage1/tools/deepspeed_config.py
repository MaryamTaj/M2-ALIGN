def get_train_ds_config(
    train_batch_size: int = 1,
    train_micro_batch_size_per_gpu: int = 1,
    lr: float = 2e-5,
    gradient_accumulation_steps: int = 1,
    offload: bool = True,
    stage: int = 2,
    enable_hybrid_engine: bool = False,
    inference_tp_size: int = 1,
    release_inference_cache: bool = False,
    pin_parameters: bool = True,
    tp_gather_partition_size: int = 8,
    max_out_tokens: int = 512,
) -> dict:
    """Build a DeepSpeed training configuration dictionary.

    Returns a config suitable for passing to ``deepspeed.initialize``.
    Uses ZeRO stage 2 with BF16 mixed precision and AdamW.  Parameter
    and optimizer states are offloaded to CPU when *offload* is ``True``.

    Args:
        train_batch_size: Global effective batch size (micro-batch ×
            gradient-accumulation × data-parallel degree).
        train_micro_batch_size_per_gpu: Per-GPU micro-batch size before
            gradient accumulation.
        lr: AdamW learning rate.
        gradient_accumulation_steps: Number of micro-steps to accumulate
            before a weight update.
        offload: Offload parameters and optimizer states to CPU to reduce
            GPU memory usage.
        stage: ZeRO optimisation stage (default: 2).
        enable_hybrid_engine: Enable DeepSpeed hybrid inference engine.
        inference_tp_size: Tensor-parallel degree for the hybrid engine.
        release_inference_cache: Release KV cache after inference.
        pin_parameters: Pin model parameters for faster CPU↔GPU transfer.
        tp_gather_partition_size: Partition size for tensor-parallel
            gather operations.
        max_out_tokens: Maximum number of tokens generated during
            hybrid-engine inference.

    Returns:
        A dict in the DeepSpeed JSON config format.
    """
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {"device": device},
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False,
    }
    return {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "steps_per_print": 2000,
        "zero_optimization": zero_opt_dict,
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
    }
