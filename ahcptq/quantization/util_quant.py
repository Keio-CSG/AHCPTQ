import torch


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max):
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant

def fake_logquantize_per_tensor_affine(x, scale, quant_min, quant_max, tau=2):
    levels = quant_max - quant_min + 1
    x = torch.clamp(x,1e-20,None)
    x_int = round_ste(-1 * (x/scale).log2() * tau)
    softmax_mask = ((x_int >= levels))
    x_q = torch.clamp(x_int, 0, levels - 1)
    X = scale * 2 ** (-1 * x_q / tau )
    X[softmax_mask] = torch.Tensor([0.0])

    return X

def fake_quantize_per_channel_affine(x, scale, zero_point, ch_axis, quant_min, quant_max):
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = scale.reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnable_per_tensor_affine_training(x, scale, zero_point, quant_min, quant_max, grad_factor):
    scale = grad_scale(scale, grad_factor)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnable_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnableplus_per_tensor_affine_training(x, scale, zero_point, quant_min, quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    scale = grad_scale(scale, grad_factor)
    zero_point = grad_scale(zero_point, grad_factor)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnableplus_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = grad_scale(zero_point, grad_factor).reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_hybrid_quantize_per_tensor_affine(x, fp_min, quant_min, quant_max, scale_log, scale_uni, grid_rate):
    levels = quant_max - quant_min + 1
    levels_log = levels * grid_rate
    levels_uni = levels - levels_log
    xq = x.clone()
    xq = xq - fp_min
    mask_log = (xq <= scale_log)
    mask_uni = ~mask_log

    xq[mask_log] = torch.clamp(xq[mask_log], 1e-20, None)
    xq[mask_log] = round_ste(-1 * (xq[mask_log] / scale_log).log2())
    softmax_mask = (xq >= levels_log)
    xq[mask_log] = torch.clamp(xq[mask_log], 0, levels_log - 1)
    xq[mask_log] = scale_log * 2 ** (-1 * xq[mask_log])
    xq[softmax_mask] = torch.Tensor([0.0])

    xq[mask_uni] = round_ste((xq[mask_uni] - scale_log) / scale_uni)
    xq[mask_uni] = torch.clamp(xq[mask_uni], quant_min, levels_uni - 1)
    xq[mask_uni] = xq[mask_uni] * scale_uni + scale_log

    xq = xq + fp_min
    return xq


def grad_scale(t, scale):
    return (t - (t * scale)).detach() + (t * scale)
