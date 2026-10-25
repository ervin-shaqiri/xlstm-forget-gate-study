import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from xlstm import (
    xLSTMLMModel,
    xLSTMLMModelConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
)
from xlstm.blocks.slstm.cell import sLSTMCell_vanilla
from xlstm.blocks.slstm.src.vanilla import slstm_forward
from .cell_variants import POINTWISE_REGISTRY


def normalize_variant_name(variant: str) -> str:
    """Normalizes CLI or legacy variant flags into canonical directory keys."""
    v = variant.strip().lower()
    if v in ("enabled", "fgate_enabled", "no_stabilizer", "true", "on"):
        return "fgate_enabled"
    if v in ("disabled", "fgate_disabled", "no_forget", "false", "off"):
        return "fgate_disabled"
    if v == "baseline":
        return "baseline"
    raise ValueError(f"Unknown variant '{variant}'. Expected 'enabled', 'disabled', or 'baseline'.")


def build_model(
    variant: str = "fgate_enabled",
    vocab_size: int = 3,
    context_length: int = 64,
    embedding_dim: int = 64,
    num_heads: int = 1,
) -> xLSTMLMModel:
    """
    Constructs an sLSTM language model configured with the requested cell pointwise variant.
    By default uses the decoupled stabilizer (m_t = 0) with forget gate enabled or disabled.
    """
    if variant not in POINTWISE_REGISTRY:
        raise ValueError(f"Unknown variant '{variant}'. Options: {list(POINTWISE_REGISTRY.keys())}")

    cfg = xLSTMLMModelConfig(
        vocab_size=vocab_size,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=num_heads)
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="vanilla",
                num_heads=num_heads,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
        ),
        context_length=context_length,
        num_blocks=1,
        embedding_dim=embedding_dim,
        slstm_at=[0],
    )
    model = xLSTMLMModel(cfg)
    pointwise_fn = POINTWISE_REGISTRY[variant]

    for _, module in model.named_modules():
        if isinstance(module, sLSTMCell_vanilla):
            module.pointwise = pointwise_fn

            def make_patched_impl(cell_ref):
                def patched_impl(training: bool, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
                    states_all, _, g = slstm_forward(
                        input,
                        state,
                        cell_ref._recurrent_kernel,
                        cell_ref._bias,
                        cell_ref.pointwise,
                        constants=cell_ref.config.constants,
                    )
                    cell_ref.g_acts = g
                    cell_ref.states_all = states_all
                    return states_all

                return patched_impl

            module._impl = make_patched_impl(module)

    return model


def count_parameters(model: torch.nn.Module, variant: str = "fgate_enabled") -> dict[str, int]:
    """
    Returns total allocated parameters and active parameters.
    When the forget gate is disabled (f_t = 1.0), the forget-gate projections
    (W_f, R_f, b_f) do not influence forward dynamics and are excluded from active params.
    """
    total_params = sum(p.numel() for p in model.parameters())
    norm_var = normalize_variant_name(variant)
    fgate_params = 0
    for name, p in model.named_parameters():
        if "fgate" in name:
            fgate_params += p.numel()
        elif "_recurrent_kernel_" in name or "_bias_" in name:
            # 1 of the 4 gates in sLSTMCell belongs to the forget gate
            fgate_params += p.numel() // 4

    active_params = total_params - fgate_params if norm_var == "fgate_disabled" else total_params
    return {
        "total_params": total_params,
        "fgate_params": fgate_params,
        "active_params": active_params,
    }

