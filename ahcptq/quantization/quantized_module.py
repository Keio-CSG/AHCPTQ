from torch import nn
from .observer import MSEFastObserver, MinMaxObserver, AvgMinMaxObserver, MSEObserver, AvgMSEObserver, AvgMSEFastObserver, LogAvgMSEFastObserver, SignAvgMSEFastObserver, PCTObserver, AvgMinMaxGroupObserver, HybridParamObserver
from .fake_quant import LSQPlusSignFakeQuantize, AdaRoundFakeQuantize, FixedFakeQuantize, LSQFakeQuantize, LSQPlusFakeQuantize, LSQSignFakeQuantize, AdaptiveGranularityQuantize, GroupLSQFakeQuantize, HybridQuantize
import torch.nn.functional as F

ObserverDict = {
    'MinMaxObserver':           MinMaxObserver,                                    # noqa: E241
    'AvgMinMaxObserver':        AvgMinMaxObserver,                                 # noqa: E241
    'MSEObserver':              MSEObserver,                                       # noqa: E241
    'AvgMSEObserver':           AvgMSEObserver,                                    # noqa: E241
    'MSEFastObserver':          MSEFastObserver,                                   # noqa: E241
    'AvgMSEFastObserver':       AvgMSEFastObserver,                                # noqa: E241
    'LogAvgMSEFastObserver':    LogAvgMSEFastObserver,
    'SignAvgMSEFastObserver':    SignAvgMSEFastObserver,
    'PCTObserver':  PCTObserver,
    'AvgMinMaxGroupObserver': AvgMinMaxGroupObserver,
    'HybridParamObserver': HybridParamObserver
}

FakeQuantizeDict = {
    'FixedFakeQuantize':     FixedFakeQuantize,                                    # noqa: E241
    'LSQFakeQuantize':       LSQFakeQuantize,                                      # noqa: E241
    'LSQSignFakeQuantize':       LSQSignFakeQuantize,                              # noqa: E241
    'LSQPlusFakeQuantize':   LSQPlusFakeQuantize,                                  # noqa: E241
    'LSQPlusSignFakeQuantize':   LSQPlusSignFakeQuantize,                          # noqa: E241
    'AdaRoundFakeQuantize':  AdaRoundFakeQuantize,                                 # noqa: E241
    'AdaptiveGranularityQuantize': AdaptiveGranularityQuantize,                    # noqa: E241
    'GroupLSQFakeQuantize': GroupLSQFakeQuantize,
    'HybridQuantize': HybridQuantize
}

def update_specialized_quantizer_config(base_config, quantizer_name):
    import copy
    specialized_config = copy.deepcopy(base_config)

    update_keys = {
        'group': {'quantizer': 'GroupLSQFakeQuantize',
                  'observer': 'AvgMinMaxGroupObserver'},
        'hybrid': {'quantizer': 'HybridQuantize',
                   'observer': 'HybridParamObserver'}
    }[quantizer_name]
    specialized_config.update(update_keys)
    return specialized_config

def ActivationQuantizer(a_qconfig, detect_ch_axis=False):
    if detect_ch_axis:
        quantizer = FakeQuantizeDict[a_qconfig.quantizer](ObserverDict[a_qconfig.observer], bit=a_qconfig.bit,
                                                          symmetric=a_qconfig.symmetric, ch_axis='det')
    else:
        quantizer = FakeQuantizeDict[a_qconfig.quantizer](ObserverDict[a_qconfig.observer], bit=a_qconfig.bit,
                                                          symmetric=a_qconfig.symmetric, ch_axis=a_qconfig.ch_axis)
    return quantizer

def SignActivationQuantizer(a_qconfig):
    return FakeQuantizeDict['LSQSignFakeQuantize'](ObserverDict[a_qconfig.observer], bit=a_qconfig.bit,
                                                 symmetric=a_qconfig.symmetric, ch_axis=a_qconfig.ch_axis)

def WeightQuantizer(w_qconfig):
    return FakeQuantizeDict[w_qconfig.quantizer](
            ObserverDict[w_qconfig.observer],
            bit=w_qconfig.bit,
            symmetric=w_qconfig.symmetric,
            ch_axis=w_qconfig.ch_axis)


class QuantizedOperator():
    pass


class QConv2d(QuantizedOperator, nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 groups,
                 bias,
                 padding_mode,
                 w_qconfig):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input, gamma=None):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)


class QLinear(QuantizedOperator, nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias,
                 w_qconfig):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input, gamma = None):
        if gamma is None:
            return F.linear(input, self.weight_fake_quant(self.weight), self.bias)
        else:
            fused_weight = self.weight.mul(gamma.unsqueeze(1))
            fused_bias = self.bias.mul(gamma)
            return F.linear(input, self.weight_fake_quant(fused_weight), fused_bias)


class QEmbedding(QuantizedOperator, nn.Embedding):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx,
                 max_norm,
                 norm_type,
                 scale_grad_by_freq,
                 sparse,
                 _weight,
                 w_qconfig):
        super().__init__(num_embeddings=num_embeddings,
                         embedding_dim=embedding_dim,
                         padding_idx=padding_idx,
                         max_norm=max_norm,
                         norm_type=norm_type,
                         scale_grad_by_freq=scale_grad_by_freq,
                         sparse=sparse,
                         _weight=_weight)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input):
        return F.embedding(
            input, self.weight_fake_quant(self.weight), self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


module_type_to_quant_weight = {
    nn.Linear: QLinear,
    nn.Conv2d: QConv2d,
    nn.Embedding: QEmbedding,
}


def get_module_args(module):
    if isinstance(module, nn.Linear):
        return dict(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None
            )
    elif isinstance(module, nn.Conv2d):
        return dict(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            )
    elif isinstance(module, nn.Embedding):
        return dict(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            _weight=None,
        )
    else:
        raise NotImplementedError


def Quantizer(module, config, detect_ch_axis=False, sign=False):
    if module is None:
        if detect_ch_axis:
            return ActivationQuantizer(a_qconfig=config, detect_ch_axis=detect_ch_axis)
        if sign:
            return SignActivationQuantizer(a_qconfig=config)
            # return LogSqrt2Quantize(a_qconfig=config)
        return ActivationQuantizer(a_qconfig=config)
    module_type = type(module)
    if module_type in module_type_to_quant_weight:
        kwargs = get_module_args(module)
        qmodule = module_type_to_quant_weight[module_type](**kwargs, w_qconfig=config)
        qmodule.weight.data = module.weight.data.clone()
        if getattr(module, 'bias', None) is not None:
            qmodule.bias.data = module.bias.data.clone()
        return qmodule
    return module


class QuantizedModule(nn.Module):
    def __init__(self):
        super().__init__()


class QuantizedLayer(QuantizedModule):
    def __init__(self, module, activation, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.module = Quantizer(module, w_qconfig)
        self.activation = activation
        if qoutput:
            self.layer_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        x = self.module(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.qoutput:
            x = self.layer_post_act_fake_quantize(x)
        return x


class PreQuantizedLayer(QuantizedModule):
    def __init__(self, module, activation, w_qconfig, a_qconfig, type='normal', qinput=True):
        super().__init__()
        self.qinput = qinput
        self.module = Quantizer(module, w_qconfig)
        self.activation = activation
        detect_ch_axis = False
        if type == 'group':
            a_qconfig = update_specialized_quantizer_config(a_qconfig, 'group')
            detect_ch_axis = True
        elif type == 'hybrid':
            a_qconfig = update_specialized_quantizer_config(a_qconfig, 'hybrid')
        elif type == 'normal':
            a_qconfig = a_qconfig
        else:
            raise NotImplementedError
        if qinput:
            self.layer_pre_act_fake_quantize = Quantizer(None, a_qconfig, detect_ch_axis)
        if type == 'hybrid':
            self.layer_pre_act_fake_quantize.weight = module.weight
            self.layer_pre_act_fake_quantize.bias = module.bias

    def forward(self, x, gamma = None):
        if self.qinput:
            x = self.layer_pre_act_fake_quantize(x)
        x = self.module(x, gamma)
        if self.activation is not None:
            x = self.activation(x)
        return x

class QuantizedMatMul(QuantizedModule):
    def __init__(self, a_qconfig, qinput=True):
        super().__init__()
        self.qinput = qinput
        if qinput:
            self.a_layer_pre_act_fake_quantize = Quantizer(None, a_qconfig)
            self.b_layer_pre_act_fake_quantize = Quantizer(None, a_qconfig)
    
    def forward(self, inputs):
        a, b = inputs
        if self.qinput:
            a = self.a_layer_pre_act_fake_quantize(a)
            b = self.b_layer_pre_act_fake_quantize(b)
        x = a @ b
        return x


class QuantizedBlock(QuantizedModule):
    def __init__(self):
        super().__init__()
