import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import pylab
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# ---- Spike-Aware Channel-Spatial Attention (ECA) ----
class EnhancedCoordinateAttention(nn.Module):
    """
    Enhanced Coordinate Attention with:
    - Dynamic reduction ratio
    - Multi-scale feature fusion
    - Residual connection
    """

    def __init__(self, in_channels, reduction=16, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        # Dynamic reduction based on input channels
        reduction = max(8, min(32, in_channels // reduction))
        mid_channels = max(8, in_channels // reduction)

        # Height attention branch
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_h1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn_h1 = nn.BatchNorm2d(mid_channels)
        self.conv_h2 = nn.Conv2d(mid_channels, in_channels, kernel_size=1)

        # Width attention branch
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_w1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn_w1 = nn.BatchNorm2d(mid_channels)
        self.conv_w2 = nn.Conv2d(mid_channels, in_channels, kernel_size=1)

        # Channel attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        # Height attention
        x_h = self.pool_h(x)  # (b, c, h, 1)
        x_h = self.conv_h1(x_h)
        x_h = self.bn_h1(x_h)
        x_h = self.relu(x_h)
        x_h = self.conv_h2(x_h)  # (b, c, h, 1)
        x_h = self.sigmoid(x_h)

        # Width attention
        x_w = self.pool_w(x)  # (b, c, 1, w)
        x_w = self.conv_w1(x_w)
        x_w = self.bn_w1(x_w)
        x_w = self.relu(x_w)
        x_w = self.conv_w2(x_w)  # (b, c, 1, w)
        x_w = self.sigmoid(x_w)

        # Channel attention
        x_c = self.gap(x).view(b, c)  # (b, c)
        x_c = self.relu(self.fc1(x_c))
        x_c = self.sigmoid(self.fc2(x_c)).view(b, c, 1, 1)  # (b, c, 1, 1)

        # Combine all attentions
        out = x * x_h * x_w * x_c

        if self.use_residual:
            out = out + identity

        return out

# ---- Temporal Attention for SNNs ----
class EnhancedTemporalAttention(nn.Module):
    """
    Enhanced temporal attention with multi-head mechanism
    """

    def __init__(self, T: int, num_heads: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.T = T
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Multi-head attention
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(T, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, T)
            ) for _ in range(num_heads)
        ])

        self.attention_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feats):
        # feats: [T, B, C]
        T, B, C = feats.shape

        # Compute attention for each head
        summary = feats.mean(dim=2).transpose(0, 1)  # [B, T]

        attention_maps = []
        for head in self.heads:
            attn = head(summary)  # [B, T]
            attention_maps.append(attn.unsqueeze(1))

        # Combine multi-head attention
        attention_maps = torch.cat(attention_maps, dim=1)  # [B, num_heads, T]
        weighted_attention = torch.sum(attention_maps * self.attention_weights.view(1, -1, 1), dim=1)
        attention_weights = self.softmax(weighted_attention).transpose(0, 1).unsqueeze(2)  # [T, B, 1]

        # Apply attention
        out = (feats * attention_weights).sum(dim=0)  # [B, C]
        return out

# ---- Surrogate spike autograd (triangle surrogate) ----
class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, gamma=1.0):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.gamma = gamma
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        gamma = ctx.gamma
        diff = (input - threshold) / gamma
        grad_surrogate = (1.0 - diff.abs()).clamp(min=0.0) / gamma
        return grad_output * grad_surrogate, None, None


# ---- Adaptive / Parametric LIF Neuron ----
class AdaptivePLIFNeuron(nn.Module):
    """
    Parametric LIF with learnable decay and optionally learnable threshold.
    Supports optional membrane normalization for stability.
    """

    def __init__(self, v_th: float = 1.0, decay_init: float = 0.9, learn_th: bool = True, norm_mem: bool = True):
        super().__init__()
        self.norm_mem = norm_mem
        self.spike_fn = SpikeFunction.apply
        # learnable decay in (0,1) via sigmoid
        self.decay_param = nn.Parameter(torch.logit(torch.tensor(decay_init)))
        if learn_th:
            self.v_th = nn.Parameter(torch.tensor(v_th))
        else:
            self.register_buffer('v_th', torch.tensor(v_th))

    @property
    def decay(self):
        # sigmoid to keep decay in (0,1)
        return torch.sigmoid(self.decay_param)

    def forward(self, syn_input, mem=None, spike=None):
        if mem is None:
            mem = torch.zeros_like(syn_input)
            spike = torch.zeros_like(syn_input)
        # parametric decay
        d = self.decay
        mem = d * (mem - self.v_th * spike) + syn_input

        if self.norm_mem:
            # channel-wise normalization to keep magnitudes stable
            denom = mem.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-6
            mem = mem / denom

        out = self.spike_fn(mem, self.v_th, 1.0)
        return mem, out


# ---- Spike Normalization (homeostasis) ----
class SpikeNorm(nn.Module):
    def __init__(self, target_rate=0.1, momentum=0.9, max_scale=2.0):
        super().__init__()
        self.target = float(target_rate)
        self.momentum = float(momentum)
        self.max_scale = float(max_scale)
        # running estimate of spike rate (scalar)
        self.register_buffer('running_rate', torch.tensor(self.target))

    def forward(self, spike):
        # spike: (B,C,H,W) or similar
        rate = spike.mean().detach()
        self.running_rate = self.momentum * self.running_rate + (1.0 - self.momentum) * rate
        scale = (self.target / (self.running_rate + 1e-8)).clamp(max=self.max_scale)
        return spike * scale

class BiSNN(nn.Module):
    def __init__(self,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 features: int = 1,
                 time_steps: int = 4,
                 use_bn: bool = True,
                 spike_dropout: float = 0.0,
                 noise_std: float = 0.0,
                 refine_spatial: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.features = features
        self.num_layers = num_layers
        self.time_steps = time_steps
        self.use_bn = use_bn
        self.spike_dropout = spike_dropout
        self.noise_std = noise_std
        self.refine_spatial = refine_spatial

        # input projection (from pooled channels -> hidden)
        self.input_proj = nn.Linear(features, hidden_size, bias=False)
        if use_bn:
            self.bn_proj = nn.BatchNorm1d(hidden_size)
        else:
            self.bn_proj = None

        # stack of small parametric LIF neurons
        # expects AdaptivePLIFNeuron defined in the same module/file
        self.lif_neurons = nn.ModuleList([
            AdaptivePLIFNeuron(v_th=0.5, decay_init=0.85, learn_th=True, norm_mem=True)
            for _ in range(num_layers)
        ])

        # output projection from spike-hidden to features
        self.output_proj = nn.Linear(hidden_size, features, bias=False)

        # optional small depthwise conv to refine spatial map (keeps model light)
        if refine_spatial:
            # depthwise: groups=features keeps params small and per-channel
            self.spatial_refine = nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, padding=1, groups=features, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            )
        else:
            self.spatial_refine = None

        # learnable per-channel scale to soften binary modulation (initialized small)
        self.scale = nn.Parameter(torch.ones(features) * 0.5)

        # state buffers
        self.mem_states = None
        self.spike_states = None

    def reset_states(self):
        self.mem_states = [None] * self.num_layers
        self.spike_states = [None] * self.num_layers

    def extra_repr(self):
        return f'hidden_size={self.hidden_size}, num_layers={self.num_layers}, time_steps={self.time_steps}, features={self.features}'

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.size()

        # global pooling -> [B, C]
        x_feat = F.adaptive_avg_pool2d(x, 1).view(b, c)

        # if input features != expected features, dynamically project
        if x_feat.size(1) != self.features:
            # lazy dynamic projection
            if not hasattr(self, 'dynamic_proj'):
                self.dynamic_proj = nn.Linear(x_feat.size(1), self.features, bias=False).to(x.device)
            x_feat = self.dynamic_proj(x_feat)

        # input proj -> hidden
        h_vec = self.input_proj(x_feat)  # [B, hidden]
        if self.bn_proj is not None:
            # BatchNorm1d expects [B, hidden]
            h_vec = self.bn_proj(h_vec)
        h_vec = F.relu(h_vec)

        # initialize states
        if self.mem_states is None:
            self.reset_states()

        final_spike = None
        for t in range(self.time_steps):
            current = h_vec
            for i in range(self.num_layers):
                mem, spike = self.lif_neurons[i](current, self.mem_states[i], self.spike_states[i])

                # training-time stochastic perturbation and dropout on spikes
                if self.training and self.noise_std > 0.0:
                    mem = mem + torch.randn_like(mem) * (self.noise_std)

                if self.training and self.spike_dropout > 0.0:
                    spike = F.dropout(spike, p=self.spike_dropout, training=True)

                # update states (detach to avoid exploding BPTT across long sequences)
                self.mem_states[i] = mem.detach()
                self.spike_states[i] = spike.detach()

                current = spike

            final_spike = current

        # map spike [B, hidden] -> binary-like vector in {-1, +1}
        binary_output = 2.0 * final_spike - 1.0  # [B, hidden] -> [-1, +1]

        # project to features
        if self.hidden_size != self.features:
            binary_output = self.output_proj(binary_output)

        # soft per-channel scaling to reduce hard jumps and allow continuous modulation
        # scale shape: [features] -> [1, features]
        scaled = torch.tanh(self.scale.view(1, -1) * binary_output) * 0.5  # in (-0.5, 0.5)
        mod_map = 1.0 + scaled  # centered around 1.0

        # reshape and expand to spatial
        spatial = mod_map.view(b, self.features, 1, 1).expand(b, self.features, h, w)

        # small refinement conv (depthwise) to add spatial diversity without many params
        if self.spatial_refine is not None:
            spatial = self.spatial_refine(spatial)
            # bring refinement output to a gentle multiplicative map around 1 via tanh
            spatial = 1.0 + 0.25 * torch.tanh(spatial)

        return spatial
#二进制激活
class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        # Calculates the sign of the input x: if x is positive, it returns 1; if x is negative, it returns -1; if x is 0, it returns 0
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))#-1*1+0
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))#0+0
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))#1+0
        out = out_forward.detach() - out3.detach() + out3

        return out

# Binary weighting: key algorithmic principles
class MetaConv2d(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        # rename parameter to 'weight' so it will match common naming and be affected by weight-noise/clamp utilities
        self.weight = nn.Parameter(torch.randn((out_chn, in_chn, kernel_size, kernel_size)) * 0.01,
                                   requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, meta_net=None):
        # Basic convolution
        out = F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

        # If no meta network is provided, return normal conv output
        if meta_net is None:
            return out

        # Try to obtain modulation map from meta_net using the current input
        try:
            mod = meta_net(x)  # expected shape: [B, F, H, W] (F may be 1 or equal to out channels)
        except Exception:
            # if meta_net cannot be called for some reason, fall back to plain conv output
            return out

        if mod is None:
            return out

        # Ensure spatial size matches conv output; if not, upsample mod to match
        b, out_ch, h, w = out.shape
        if mod.shape[2:] != (h, w):
            mod = F.interpolate(mod, size=(h, w), mode='bilinear', align_corners=False)

        # Adapt channel dimension
        if mod.shape[1] == out_ch:
            mod_map = mod
        elif mod.shape[1] == 1:
            mod_map = mod.expand(-1, out_ch, -1, -1)
        else:
            # reduce extra channels by mean and expand
            mod_map = mod.mean(dim=1, keepdim=True).expand(-1, out_ch, -1, -1)

        # Convert modulation map to a gentle scale around 1: use tanh so values are in (-1,1), then add 1
        mod_scale = 1.0 + torch.tanh(mod_map)

        # Apply modulation (per-sample, per-channel, per-spatial-location)
        out = out * mod_scale
        return out

class SNNMultiShortcutBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, lif_kwargs=None,
                 freeze_bn=False, use_attention: bool = False,
                 use_spikenorm: bool = True,
                 attention_type: str = "eca",
                 use_res_gate: bool = True,
                 gate_type: str = "lightweight"):
        super().__init__()
        self.binary_activation = BinaryActivation()
        self.binary_conv1 = MetaConv2d(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.binary_conv2 = MetaConv2d(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.binary_conv3 = MetaConv2d(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.nonlinear = nn.PReLU(planes)
        self.downsample = downsample
        lif_kwargs = lif_kwargs or {}
        self.lif = AdaptivePLIFNeuron(**lif_kwargs)
        self.freeze_bn = freeze_bn
        self.attention_type = attention_type

        self.use_res_gate = use_res_gate
        self.gate_type = gate_type
        if use_res_gate:
            self._init_res_gate(planes, gate_type)

        # spike norm
        self.use_spikenorm = use_spikenorm
        if use_spikenorm:
            self.spike_norm = SpikeNorm()
        else:
            self.spike_norm = None

        # attention module: only use in deeper layers
        self.use_attention = use_attention
        if use_attention:
            self.attention = EnhancedCoordinateAttention(planes, reduction=16)
        else:
            self.attention = None

    def _init_res_gate(self, planes: int, gate_type: str):
        """Initialise the residual gating module"""
        if gate_type == "lightweight":
            # The most lightweight channel gating
            self.res_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(planes, max(planes // 8, 4), 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(planes // 8, 4), planes, 1, bias=False),
                nn.Sigmoid()
            )

        elif gate_type == "simple":
            # Minimalist Door Control - Global Scalar Weights
            self.res_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(planes, 1, 1, bias=False),
                nn.Sigmoid()
            )

        elif gate_type == "adaptive":
            # Adaptive Gate Control – Balancing the Main Path and Residuals
            self.res_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(planes, max(planes // 4, 8), 1, bias=False),
                nn.ReLU(inplace=True),
                # Output two weights
                nn.Conv2d(max(planes // 4, 8), 2, 1, bias=False),
                nn.Softmax(dim=1)
            )

        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

        # Trainable global residual weights
        self.res_weight = nn.Parameter(torch.tensor(1.0))

    def forward_time(self, x, mem, spike, meta_net):
        residual = x
        out = self.binary_activation(x)

        out1 = self.bn1(self.binary_conv1(out, meta_net))
        out2 = self.bn2(self.binary_conv2(out1, meta_net))
        out3 = self.bn3(self.binary_conv3(out2, meta_net))
        out_sum = out1 + out2 + out3  # multiple shortcuts fusion

        # apply attention if present
        if self.attention is not None:
            out_sum = self.attention(out_sum)

        # If mem/spike missing or shape mismatch, reinit to match out_sum
        if mem is None or spike is None or mem.shape != out_sum.shape:
            mem = torch.zeros_like(out_sum, device=out_sum.device, dtype=out_sum.dtype)
            spike = torch.zeros_like(out_sum, device=out_sum.device, dtype=out_sum.dtype)

        # integrate & fire
        mem, spk = self.lif(out_sum, mem, spike)

        # apply spike normalization/homeostasis
        if self.use_spikenorm and self.spike_norm is not None:
            spk = self.spike_norm(spk)

        # apply downsample to residual if necessary to match shapes
        if self.downsample is not None:
            residual = self.downsample(x)

        # Application of residual gating
        if self.use_res_gate and hasattr(self, 'res_gate'):
            if residual.shape != spk.shape:
                # Adjust the shape of the residuals to match the main path
                if residual.shape[2:] != spk.shape[2:]:
                    residual = F.adaptive_avg_pool2d(residual, spk.shape[2:])
                if residual.shape[1] != spk.shape[1]:
                    padding = spk.shape[1] - residual.shape[1]
                    residual = F.pad(residual, (0, 0, 0, 0, 0, padding))

            if self.gate_type == "lightweight" or self.gate_type == "simple":
                gate_weight = self.res_gate(spk)
                gated_residual = residual * gate_weight

            elif self.gate_type == "adaptive":
                gate_weights = self.res_gate(spk)
                alpha = gate_weights[:, 0:1, :, :]
                beta = gate_weights[:, 1:2, :, :]
                gated_residual = residual * beta
                spk = spk * alpha
            else:
                gated_residual = residual

            # Application of learnable global weights
            # Restricted to the range 0–1
            weight = torch.sigmoid(self.res_weight)
            out = spk + weight * gated_residual
        else:
            out = spk + residual

        out = self.nonlinear(out)

        # optionally freeze batchnorm behavior
        if self.freeze_bn:
            self.bn1.eval()
            self.bn2.eval()
            self.bn3.eval()

        return out, mem, spk

class BiSpikeNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, time_steps=4,
                 lif_decay: float = 0.9, detach_mem: bool = True,
                 auto_reset: bool = True, freeze_bn: bool = False, weight_clamp: bool = False,
                 use_spikenorm: bool = True, use_temporal_att: bool = True,
                 weight_noise_std: float = 0.0, use_attention_layers: tuple = (2, 3)):
        super().__init__()
        self.time_steps = time_steps
        self.detach_mem = detach_mem
        self.auto_reset = auto_reset
        self.freeze_bn = freeze_bn
        self.weight_clamp = weight_clamp
        self.use_spikenorm = use_spikenorm
        self.use_temporal_att = use_temporal_att
        self.weight_noise_std = float(weight_noise_std)
        self.use_attention_layers = use_attention_layers

        self.inplanes = 64
        # keep ImageNet-style head (7x7 conv + maxpool) but using 3x3 for reduced params
        self.conv1 = nn.Conv2d(7, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.nonlinear = nn.PReLU(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        lif_kwargs = {'v_th': 0.5, 'decay_init': lif_decay, 'learn_th': True, 'norm_mem': True}
        self.layer1 = self._make_layer(block, 64, layers[0], lif_kwargs=lif_kwargs,
                                       freeze_bn=freeze_bn, layer_idx=0)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, lif_kwargs=lif_kwargs,
                                       freeze_bn=freeze_bn, layer_idx=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, lif_kwargs=lif_kwargs,
                                       freeze_bn=freeze_bn, layer_idx=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, lif_kwargs=lif_kwargs,
                                       freeze_bn=freeze_bn, layer_idx=3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        try:
            self.meta_net = BiSNN()
        except Exception:
            self.meta_net = None

        # temporal aggregator
        if self.use_temporal_att:
            self.temporal_agg = EnhancedTemporalAttention(self.time_steps)
        else:
            self.temporal_agg = None

        self._init_states = False
        self.mem_states = [None, None, None, None]
        self.spike_states = [None, None, None, None]

    def _make_layer(self, block, planes, blocks, stride=1, lif_kwargs=None,
                    freeze_bn=False, layer_idx=0, attention_type="eca"):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        for i in range(blocks):
            # Only apply attention in specified layers
            use_attention = layer_idx in self.use_attention_layers
            layers.append(block(
                self.inplanes if i == 0 else planes,
                planes,
                stride if i == 0 else 1,
                downsample if i == 0 else None,
                lif_kwargs,
                freeze_bn,
                use_attention=use_attention,
                use_spikenorm=self.use_spikenorm,
                # Passing attention types
                attention_type=attention_type
            ))
            self.inplanes = planes * block.expansion
        return nn.ModuleList(layers)

    def reset_states(self):
        self._init_states = False
        self.mem_states = [None, None, None, None]
        self.spike_states = [None, None, None, None]

    def apply_weight_clamp(self):
        for name, p in self.named_parameters():
            if p.requires_grad and p.dim() >= 2:
                with torch.no_grad():
                    p.clamp_(-1.0, 1.0)

    def apply_weight_noise(self):
        if self.training and self.weight_noise_std > 0.0:
            with torch.no_grad():
                for name, p in self.named_parameters():
                    if 'weight' in name and p.requires_grad:
                        p.add_(torch.randn_like(p) * self.weight_noise_std)

    def forward(self, x, return_logits: bool = False):
        if self.auto_reset:
            self.reset_states()

        # optionally inject small weight noise for robustness
        if self.weight_noise_std > 0.0:
            self.apply_weight_noise()

        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.nonlinear(feat)
        feat = self.maxpool(feat)

        if not self._init_states:
            self._init_states = True
            self.mem_states = [None, None, None, None]
            self.spike_states = [None, None, None, None]

        outputs = []
        for t in range(self.time_steps):
            cur = feat if t == 0 else feat.detach()

            for idx, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                if self.mem_states[idx] is None:
                    self.mem_states[idx] = None
                    self.spike_states[idx] = None

                mem = self.mem_states[idx]
                spk = self.spike_states[idx]

                for blk in layer:
                    cur, mem, spk = blk.forward_time(cur, mem, spk, self.meta_net)

                self.mem_states[idx] = mem
                self.spike_states[idx] = spk

            pooled = self.avgpool(cur)
            v = pooled.view(pooled.size(0), -1)
            logits = self.fc(v)
            outputs.append(logits)

            if self.detach_mem:
                for i in range(len(self.mem_states)):
                    if self.mem_states[i] is not None:
                        self.mem_states[i] = self.mem_states[i].detach()
                    if self.spike_states[i] is not None:
                        self.spike_states[i] = self.spike_states[i].detach()

        out_seq = torch.stack(outputs, dim=0)  # [T, B, C]

        # temporal aggregation
        if self.temporal_agg is not None:
            out = self.temporal_agg(out_seq)  # [B, C]
        else:
            out = out_seq.mean(dim=0)

        if self.weight_clamp:
            self.apply_weight_clamp()

        if return_logits:
            return out
        return F.log_softmax(out, dim=1)


# ---- main ----
def bispikenet(time_steps=2, use_attention_layers=(1, 2), **kwargs):
    return BiSpikeNet(BiSpikeNet, [2, 2, 0, 2],
                                time_steps=time_steps,
                                use_attention_layers=use_attention_layers,
                                **kwargs)
