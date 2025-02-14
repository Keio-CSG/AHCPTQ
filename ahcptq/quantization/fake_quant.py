import pandas as pd
import torch
import torch.nn as nn
from .observer import ObserverBase
from .util_quant import (
    fake_quantize_per_channel_affine,
    fake_quantize_per_tensor_affine,
    fake_quantize_learnable_per_tensor_affine_training,
    fake_quantize_learnable_per_channel_affine_training,
    fake_quantize_learnableplus_per_channel_affine_training,
    fake_quantize_learnableplus_per_tensor_affine_training,
    fake_hybrid_quantize_per_tensor_affine
)
from math import sqrt
from .util_quant import round_ste
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

import logging
logger = logging.getLogger('ahcptq')
class QuantizeBase(nn.Module):

    def __init__(self, observer=ObserverBase, bit=8, symmetric=False, ch_axis=-1):
        super().__init__()
        self.observer = observer(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.observer_enabled = 0
        self.fake_quant_enabled = 0
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max
        self.drop_prob = 1.0

    def set_bit(self, bit):
        self.observer.set_bit(bit)
        self.bit = bit
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max

    def set_name(self, name):
        self.name = name

    @torch.jit.export
    def calculate_qparams(self):
        return self.observer.calculate_qparams()

    @torch.jit.export
    def disable_observer(self):
        self.observer_enabled = 0

    @torch.jit.export
    def enable_observer(self):
        self.observer_enabled = 1

    @torch.jit.export
    def disable_fake_quant(self):
        self.fake_quant_enabled = 0

    @torch.jit.export
    def enable_fake_quant(self):
        self.fake_quant_enabled = 1

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'symmetric={}, bit={}, ch_axis={}, quant_min={}, quant_max={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.symmetric, self.bit, self.ch_axis,
                   self.quant_min, self.quant_max)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                if name == 'scale':
                    if isinstance(self.scale, nn.Parameter):
                        self.scale.data = torch.ones_like(val.to(self.scale.device))
                    else:
                        self.scale.resize_(val.shape)
                else:
                    assert name == 'zero_point'
                    if isinstance(self.zero_point, nn.Parameter):
                        self.zero_point.data = torch.ones_like(val.to(self.zero_point.device))
                    else:
                        self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'zero_point'
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class FixedFakeQuantize(QuantizeBase):

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.drop_prob = 1.0

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = X

            if self.ch_axis != -1:
                X = fake_quantize_per_channel_affine(
                    X, self.scale.data, self.zero_point.data.int(), self.ch_axis,
                    self.quant_min, self.quant_max)
            else:
                X = fake_quantize_per_tensor_affine(
                    X, self.scale.item(), self.zero_point.item(),
                    self.quant_min, self.quant_max)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob

        return X


class LSQFakeQuantize(QuantizeBase):

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.use_grad_scaling = use_grad_scaling
        self.drop_prob = 1.0

    def forward(self, X, value=None):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.resize_(_zero_point.shape)

            self.scale.data.copy_(_scale)
            self.zero_point.copy_(_zero_point)
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())

        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = X
            if self.ch_axis != -1:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnable_per_channel_affine_training(
                    X, self.scale, self.zero_point.data.int(), self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnable_per_tensor_affine_training(
                    X, self.scale, self.zero_point.item(), self.quant_min, self.quant_max, grad_factor)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob
        return X

from .observer import * 
class LSQSignFakeQuantize(LSQFakeQuantize):

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.use_grad_scaling = use_grad_scaling
        self.drop_prob = 1.0

        self.sign = None
        self.is_bimodal = None


    def judge_bimodal(self, data_inp):
        
        data = data_inp[0].flatten().cpu().numpy()
        kde = gaussian_kde(data)
        x = np.linspace(min(data), max(data), self.global_num)
        y = kde(x)
        peaks, _ = find_peaks(y, height = self.peak_height*sum(y), distance=self.peak_distance)
        self.is_bimodal = len(peaks) == 2
        
        if self.is_bimodal:
            data_inp = data_inp.transpose(0, -1).flatten(1, -1)
            self.sign = torch.tensor([torch.sign(chan_data.mean()) \
                    for chan_data in data_inp]).cuda()

    def forward(self, X):
        if self.is_bimodal is None:

            self.judge_bimodal(X)

        if self.observer_enabled == 1:

            self.observer(X.detach()) # TODO
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.data.copy_(_scale)
            self.zero_point.copy_(_zero_point)
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())


        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = X
            if self.ch_axis != -1:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnable_per_channel_affine_training(
                    X, self.scale, self.zero_point.data.int(), self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0

                X = fake_quantize_learnable_per_tensor_affine_training(
                    X, self.scale, self.zero_point.item(), self.quant_min, self.quant_max, grad_factor)
                
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob
        return X



class LSQPlusFakeQuantize(QuantizeBase):

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.zero_point = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.use_grad_scaling = use_grad_scaling
        self.drop_prob = 1.0

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.data = torch.zeros_like(_zero_point.float())
            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point.float())
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())
            self.zero_point.data.clamp_(self.quant_min, self.quant_max).float()

        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = X
            if self.ch_axis != -1:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnableplus_per_channel_affine_training(
                    X, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnableplus_per_tensor_affine_training(
                    X, self.scale, self.zero_point, self.quant_min, self.quant_max, grad_factor)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob
        return X

class LSQPlusSignFakeQuantize(LSQPlusFakeQuantize):

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True):
        # if observer not in [AvgMinMaxObserver, MinMaxObserver]:
        #     observer = AvgMinMaxObserver
        observer
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.zero_point = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.use_grad_scaling = use_grad_scaling
        self.drop_prob = 1.0
        
        # self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        # self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        # self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        # self.use_grad_scaling = use_grad_scaling
        # self.drop_prob = 1.0
        self.observer2sign = True
        print('ob2sing',self.observer2sign)

        self.gamma = 0.8
        self.is_A_two_peak = None
        self.a_sign = None

        self.only4sign = True

    def _judge_two_peak(self, A):
        A = A.transpose(0, -1).flatten(1, -1).cuda()
        def judge(channel_tensor : torch.Tensor):
            channel_tensor = channel_tensor.clone().reshape(-1)
            total_num = channel_tensor.numel()
            pos_num = (channel_tensor > 0).nonzero().numel()
            neg_num = total_num - pos_num
            asy_rate = torch.tensor((pos_num - neg_num) / total_num).abs()
            return 1 if asy_rate >= self.gamma else 0
        
        a_num = 0
        a_sign = []
        for a_channel in A:
            a_num = a_num + judge(a_channel)
            a_sign.append(torch.sign((a_channel.min() + a_channel.max())))
        
        channel_dim = A.shape[0]
        if (a_num / channel_dim) > self.gamma:
            # self.is_A_two_peak = True
            new_a_sign = torch.tensor(a_sign).cuda().detach()
            if self.a_sign is None or new_a_sign.equal(self.a_sign):
                self.is_A_two_peak = True
                self.a_sign = new_a_sign
            else:
                self.is_A_two_peak = False
        else:
            self.is_A_two_peak = False

    def forward(self, X):
        if self.only4sign:
            if self.is_A_two_peak is None:
                self._judge_two_peak(X)
        if self.observer_enabled == 1:
            # if self.is_A_two_peak is None:
            #     self._judge_two_peak(X)
            # elif self.is_A_two_peak:
            #     self._judge_two_peak(X)

            if self.is_A_two_peak:
                X_in = X.detach()*self.a_sign[None,None,:] #TODO
            else:
                X_in = X.detach()

            self.observer.sign = self.a_sign if self.observer2sign else None
            self.observer(X_in) # TODO
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.data = torch.zeros_like(_zero_point.float())
                # self.zero_point.resize_(_zero_point.shape)

            self.scale.data.copy_(_scale)
            self.zero_point.copy_(_zero_point.float())

            
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())
            self.zero_point.data.clamp_(self.quant_min, self.quant_max).float()


        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = X
            if self.ch_axis != -1:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnableplus_per_channel_affine_training(
                    X, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0

                if self.is_A_two_peak:
                    X = X*self.a_sign[None,None,:]

                X = fake_quantize_learnableplus_per_tensor_affine_training(
                    X, self.scale, self.zero_point, self.quant_min, self.quant_max, grad_factor)
                
                if self.is_A_two_peak:
                    X = X/self.a_sign[None,None,:]

            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob
        return X



class AdaRoundFakeQuantize(QuantizeBase):
    """
    self.adaround=True: turn on up or down forward
    self.adaround=False: turn on round-to-nearest forward
    based on the FixedFakeQuantize
    """

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.adaround = False
        self.gamma, self.zeta = -0.1, 1.1

    def init(self, weight_tensor: torch.Tensor, round_mode):
        self.adaround = True
        self.round_mode = round_mode
        self.init_alpha(x=weight_tensor.data.clone().detach())

    def init_alpha(self, x: torch.Tensor):
        if self.ch_axis != -1:
            new_shape = [1] * len(x.shape)
            new_shape[self.ch_axis] = x.shape[self.ch_axis]
            scale = self.scale.data.reshape(new_shape)
        else:
            scale = self.scale.data
        x_floor = torch.floor(x / scale)
        if self.round_mode == 'learned_hard_sigmoid':
            rest = (x / scale) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = torch.nn.Parameter(alpha)
        else:
            raise NotImplementedError

    def rectified_sigmoid(self):
        """generate rounding mask.
        """
        return ((self.zeta - self.gamma) * torch.sigmoid(self.alpha) + self.gamma).clamp(0, 1)

    def adaround_forward(self, X, hard_value=False):
        if self.ch_axis != -1:
            new_shape = [1] * len(X.shape)
            new_shape[self.ch_axis] = X.shape[self.ch_axis]
            scale = self.scale.data.reshape(new_shape)
            zero_point = self.zero_point.data.int().reshape(new_shape)
        else:
            scale = self.scale.item()
            zero_point = self.zero_point.item()
        X = torch.floor(X / scale)
        if hard_value:
            X += (self.alpha >= 0).float()
        else:
            X += self.rectified_sigmoid()
        X += zero_point
        X = torch.clamp(X, self.quant_min, self.quant_max)
        X = (X - zero_point) * scale
        return X

    def get_hard_value(self, X):
        X = self.adaround_forward(X, hard_value=True)
        return X

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled == 1:
            if not self.adaround:
                if self.ch_axis != -1:
                    X = fake_quantize_per_channel_affine(
                        X, self.scale.data, self.zero_point.data.int(), self.ch_axis,
                        self.quant_min, self.quant_max)
                else:
                    X = fake_quantize_per_tensor_affine(
                        X, self.scale.item(), self.zero_point.item(),
                        self.quant_min, self.quant_max)
            else:
                if not hasattr(self, 'alpha'):
                    raise NotImplementedError
                if self.round_mode == 'learned_hard_sigmoid':
                    X = self.adaround_forward(X)
                else:
                    raise NotImplementedError
        return X



class AdaptiveGranularityQuantize(QuantizeBase):

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)

        self.tau = 2
        self.inited = False
        self.value = None
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        
    def ori_forward(self, x: torch.Tensor):

        if self.inited is False:
            self.scale = torch.nn.Parameter(self.init_quantization_scale(x))
            self.inited = True

        x_dequant = self.quantize(x, self.scale)
        return x_dequant
    
    def forward(self, X, value=None):
        if self.observer_enabled == 1:
            if value is None:
                value = self.value
            self.observer(X.detach(),value=value)

        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = X

            X = self.ori_forward(X)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob
        return X

    def init_quantization_scale(self, x: torch.Tensor):
        tau_errors = self.observer.tau_errors
        _, min_error_idx = torch.min(tau_errors, dim=0)
        
        scale = self.observer.best_tau_scales[min_error_idx]
        self.tau = self.observer.taus[min_error_idx]
        
        return scale

    def quantize(self, x, scale):
        levels = self.observer.quant_max - self.observer.quant_min + 1
        x = torch.clamp(x, 1e-20, None)
        x_int = round_ste(-1 * (x/scale).log2() * self.tau)

        softmax_mask = x_int >= levels
        x_q = torch.clamp(x_int, 0, levels - 1)
        X = scale * 2 ** (-1 * x_q / self.tau)
        X[softmax_mask] = torch.Tensor([0.0])

        return X


class GroupLSQFakeQuantize(QuantizeBase):
    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.grouped_scales = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.grouped_zero_points = torch.tensor([0], dtype=torch.float)
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.use_grad_scaling = use_grad_scaling
        self.drop_prob = 1.0
        self.grouped = False
        self.labels = None
        self.ori_num_channel = None

    def map_vector(self):
        labels = self.labels.long()
        mapped_scale = self.grouped_scales[labels]
        mapped_zero_point = self.grouped_zero_points[labels]
        return mapped_scale, mapped_zero_point

    def group_channel(self, num_groups):
        if not self.grouped:
            scale = self.grouped_scales.to(self.grouped_scales.device)
            zero_point = self.grouped_zero_points.to(self.grouped_scales.device)
        if self.labels is not None:
            scale, zero_point = self.map_vector()

        vector_params = torch.cat([scale.unsqueeze(1), zero_point.unsqueeze(1)], dim=1).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=num_groups, random_state=0).fit(vector_params)
        cluster_centers = torch.tensor(kmeans.cluster_centers_, device=self.grouped_scales.device)

        new_grouped_scales = cluster_centers[:, 0]
        new_grouped_zero_points = cluster_centers[:, 1]
        new_grouped_zero_points = torch.round(new_grouped_zero_points)
        self.grouped_scales = torch.nn.Parameter(new_grouped_scales)
        self.grouped_zero_points = torch.nn.Parameter(new_grouped_zero_points)
        self.labels = torch.tensor(kmeans.labels_, device=self.grouped_scales.device)

        self.grouped = True

    def forward(self, X):
        if self.ch_axis == 'det':
            self.ch_axis = len(X.shape) - 1
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.grouped_scales.device), _zero_point.to(self.grouped_scales.device)
            self.grouped_scales.data = torch.ones_like(_scale)
            self.grouped_zero_points.resize_(_zero_point.shape)

            self.grouped_scales.data.copy_(_scale)
            self.grouped_zero_points.copy_(_zero_point)
        else:
            self.grouped_scales.data.abs_()
            self.grouped_scales.data.clamp_(min=self.eps.item())

        if self.fake_quant_enabled == 1:
            if not self.grouped:
                scale = self.grouped_scales.to(self.grouped_scales.device)
                zero_point = self.grouped_zero_points.to(self.grouped_scales.device)
            if self.ori_num_channel is None:
                self.ori_num_channel = self.grouped_scales.shape[0]
            if self.labels is not None:
                scale, zero_point = self.map_vector()
            if self.drop_prob < 1.0:
                x_orig = X
            if self.use_grad_scaling:
                grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
            else:
                grad_factor = 1.0
            X = fake_quantize_learnable_per_channel_affine_training(X, scale, zero_point.data, self.ch_axis,
                                                                    self.quant_min, self.quant_max, grad_factor)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob
        return X


class HybridQuantize(QuantizeBase):
    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.inited = False
        self.weight = None
        self.bias = None
        self.fp_min = None
        self.grid_rate = None
        self.scale_log = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float, device='cuda'))
        self.scale_uni = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float, device='cuda'))

    def init_quantization(self):
        loss_list = self.observer.loss_list
        avg_loss_list = {}
        for losses in loss_list:
            for loss, range_rate, grid_rate in losses:
                key = (range_rate, grid_rate)
                if key not in avg_loss_list:
                    avg_loss_list[key] = []
                avg_loss_list[key].append(loss)
        avg_loss_list = {k: sum(v) / len(v) for k, v in avg_loss_list.items()}
        best_params = min(avg_loss_list, key=avg_loss_list.get)
        best_range_rate, best_grid_rate = best_params

        range = self.observer.max_val - self.observer.min_val
        self.fp_min = self.observer.min_val
        self.grid_rate = best_grid_rate
        scale_log = range * best_range_rate
        scale_uni = range * (1 - best_range_rate) / ((self.observer.quant_max - self.observer.quant_min) * (1 - best_grid_rate))
        self.scale_log = torch.nn.Parameter(torch.tensor([scale_log], dtype=torch.float, device='cuda'))
        self.scale_uni = torch.nn.Parameter(torch.tensor([scale_uni], dtype=torch.float, device='cuda'))

    def quantize_activation(self, x: torch.Tensor):
        xq = fake_hybrid_quantize_per_tensor_affine(x, self.fp_min, self.observer.quant_min, self.observer.quant_max,
                                                    self.scale_log, self.scale_uni, self.grid_rate)
        return xq

    def forward(self, x):
        if self.observer_enabled == 1:
            self.observer(x.detach(), self.weight, self.bias)
        if self.fake_quant_enabled == 1:
            if self.inited is False:
                self.init_quantization()
                self.inited = True
            if self.drop_prob < 1.0:
                x_orig = x
            x = self.quantize_activation(x)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(x) < self.drop_prob, x, x_orig)
                return x_prob
        return x
