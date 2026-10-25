import torch
from torch.nn.functional import logsigmoid


def slstm_pointwise_fgate_enabled(
    Wx: torch.Tensor,
    Ry: torch.Tensor,
    b: torch.Tensor,
    states: torch.Tensor,
    constants: dict[str, float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decoupled-Stabilizer sLSTM with Forget Gate Enabled (no_stabilizer):
    - Stabilizer m_t is removed (m_t = 0) so i_t and f_t are decoupled.
    - Forget gate is standard sigmoid(f_raw), independent of i_raw.
    - Input gate is pure exp(i_raw) (clamped at 15.0 to prevent fp32 overflow).
    """
    raw = Wx + Ry + b
    y, c, n, m = torch.unbind(states.view(4, states.shape[1], -1), dim=0)
    iraw, fraw, zraw, oraw = torch.unbind(raw.view(raw.shape[0], 4, -1), dim=1)

    ogate = torch.sigmoid(oraw)
    fgate = torch.sigmoid(fraw)
    igate = torch.exp(torch.clamp(iraw, max=15.0))

    cnew = fgate * c + igate * torch.tanh(zraw)
    nnew = fgate * n + igate
    ynew = ogate * cnew / (nnew + 1e-6)
    mnew = torch.zeros_like(iraw)

    return torch.stack((ynew, cnew, nnew, mnew), dim=0), torch.stack((igate, fgate, zraw, ogate), dim=0)


def slstm_pointwise_fgate_disabled(
    Wx: torch.Tensor,
    Ry: torch.Tensor,
    b: torch.Tensor,
    states: torch.Tensor,
    constants: dict[str, float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decoupled-Stabilizer sLSTM with Forget Gate Disabled (no_forget):
    - Stabilizer m_t is removed (m_t = 0).
    - Forget gate is permanently fixed to 1.0 (no exponential decay of past states).
    - Input gate is pure exp(i_raw) (clamped at 15.0).
    - Normalizer n_t accumulates sum(exp(i_raw)), inducing causal softmax dilution.
    """
    raw = Wx + Ry + b
    y, c, n, m = torch.unbind(states.view(4, states.shape[1], -1), dim=0)
    iraw, fraw, zraw, oraw = torch.unbind(raw.view(raw.shape[0], 4, -1), dim=1)

    ogate = torch.sigmoid(oraw)
    fgate = torch.ones_like(iraw)
    igate = torch.exp(torch.clamp(iraw, max=15.0))

    cnew = c + igate * torch.tanh(zraw)
    nnew = n + igate
    ynew = ogate * cnew / (nnew + 1e-6)
    mnew = torch.zeros_like(iraw)

    return torch.stack((ynew, cnew, nnew, mnew), dim=0), torch.stack((igate, fgate, zraw, ogate), dim=0)


def slstm_pointwise_baseline(
    Wx: torch.Tensor,
    Ry: torch.Tensor,
    b: torch.Tensor,
    states: torch.Tensor,
    constants: dict[str, float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standard sLSTM with coupled stabilizer m_t:
    m_t = max(i_raw, m_{t-1} + log(sigmoid(f_raw)))
    i_gate = exp(i_raw - m_t)
    f_gate = exp(m_{t-1} + log(sigmoid(f_raw)) - m_t)
    """
    raw = Wx + Ry + b
    y, c, n, m = torch.unbind(states.view(4, states.shape[1], -1), dim=0)
    iraw, fraw, zraw, oraw = torch.unbind(raw.view(raw.shape[0], 4, -1), dim=1)

    logfplusm = m + logsigmoid(fraw)
    if torch.all(n == 0.0):
        mnew = iraw
    else:
        mnew = torch.max(iraw, logfplusm)

    ogate = torch.sigmoid(oraw)
    igate = torch.minimum(torch.exp(iraw - mnew), torch.ones_like(iraw))
    fgate = torch.minimum(torch.exp(logfplusm - mnew), torch.ones_like(iraw))

    cnew = fgate * c + igate * torch.tanh(zraw)
    nnew = fgate * n + igate
    ynew = ogate * cnew / (nnew + 1e-6)

    return torch.stack((ynew, cnew, nnew, mnew), dim=0), torch.stack((igate, fgate, zraw, ogate), dim=0)


# Aliases for backward compatibility
slstm_pointwise_no_stabilizer = slstm_pointwise_fgate_enabled
slstm_pointwise_no_forget = slstm_pointwise_fgate_disabled

POINTWISE_REGISTRY = {
    "fgate_enabled": slstm_pointwise_fgate_enabled,
    "enabled": slstm_pointwise_fgate_enabled,
    "no_stabilizer": slstm_pointwise_fgate_enabled,
    "fgate_disabled": slstm_pointwise_fgate_disabled,
    "disabled": slstm_pointwise_fgate_disabled,
    "no_forget": slstm_pointwise_fgate_disabled,
    "baseline": slstm_pointwise_baseline,
}

