import torch
# from dataset1 import StockDataset
from torch.utils.data import DataLoader,random_split
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import os
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import optuna

import math
# from PatchTST_backbone import PatchTST_backbone, TSTEncoder
# from PatchTST_layers import series_decomp

# from RevIN import RevIN
from typing import Callable, Optional
from torch import Tensor
import random
from transformers import AutoModel, AutoTokenizer
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def set_seed(seed=42):
    torch.manual_seed(seed)  # 为 CPU 设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前 GPU 设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU，设置所有 GPU 的随机种子
    np.random.seed(seed)  # 设置 numpy 的随机种子
    random.seed(seed)  # 设置 python 的随机种子
    torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法一致
    torch.backends.cudnn.benchmark = False  # 如果输入数据的大小不变，设置为 False 可能会提升性能

# 调用设置种子
set_seed(2)
# from mamba_ssm import Mamba

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class InferenceParams:
    def __init__(self):
        # 存储每一层的缓存状态
        self.key_value_memory_dict = {}
        # 当前生成到第几个 token
        self.seqlen_offset = 0
        
    def update_offset(self):
        self.seqlen_offset += 1

    def reset(self):
        self.key_value_memory_dict.clear()
        self.seqlen_offset = 0

class Mamba1(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = False

        # self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj = nn.Sequential(nn.Linear(self.d_inner*2, self.d_inner, bias=bias, **factory_kwargs),nn.ReLU(),nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs))
        self.inference_params = InferenceParams()
    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        # print(hidden_states.shape)
        batch, seqlen, dim = hidden_states.shape
        # inference_params = self.inference_params
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            # print(conv_state.shape,ssm_state.shape)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out
        # conv_state = torch.zeros(batch, dim, seqlen, device=hidden_states.device)
        # ssm_state = torch.zeros(batch, seqlen, dim, device=hidden_states.device)

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # print(x.shape,z.shape,"198")
            # Compute short convolution
            if conv_state is not None:
                x_padded = F.pad(x, (self.d_conv - x.shape[-1], 0))  # 填充x的最后一个维度
                # print(self.d_conv - x.shape[-1],x_padded.shape)
                # conv_state.copy_(x_padded.mean(dim=0))
                conv_state = x_padded.mean(dim=0)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
            # Feature interaction in the SSM state update
            # Initialize S_t as a zero matrix (or could use a learnable init)
            S_t = torch.zeros_like(A)  # Assuming A is (d_state, d_state)
            # Compute the new S_t with the interaction term
            S_t.copy_(A)
            # Compute the new S_t with the interaction term (A @ S_t @ A.T + B @ x @ x.T)
            S_t_weighted =  A.T @ S_t # Ensure this operation is valid by aligning dimensions
            x_interaction = B @ x.T.permute(2,0,1)
            x_interaction = torch.matmul(S_t_weighted.unsqueeze(0),x_interaction)
            
            x_interaction = torch.matmul(x_interaction, x.T.permute(2,1,0))
            # Compute the output using the new S_t
            C_T = C.T.permute(2, 0, 1)  # 调整C.T的维度为 [20315, 16, 5]
            S_t_reshaped = S_t.T.unsqueeze(0)  # 将S_t的维度从[64, 16]变为[1, 16, 64]
            tr_term = torch.matmul(C_T, S_t_reshaped)
            y_t = tr_term + (self.D @ x).unsqueeze(-1)
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                # ssm_state.copy_(last_state.mean(dim=0))
                ssm_state = last_state.mean(dim=0)
            y_t = y_t.permute(0,2,1)
            y = torch.cat([y,y_t],dim=1)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        # print("248**",hidden_states.shape)
        # dtype = hidden_states.dtype
        # # assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        # xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        # x, z = xz.chunk(2, dim=-1)  # (B D)

        # # Conv step
        # if causal_conv1d_update is None:
        #     conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
        #     print(x.shape,conv_state.shape)
        #     conv_state[:, :, -1] = x
        #     x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
        #     if self.conv1d.bias is not None:
        #         x = x + self.conv1d.bias
        #     x = self.act(x).to(dtype=dtype)
        # else:
        #     x = causal_conv1d_update(
        #         x,
        #         conv_state,
        #         rearrange(self.conv1d.weight, "d 1 w -> d w"),
        #         self.conv1d.bias,
        #         self.activation,
        #     )

        # x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        # dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # # Don't add dt_bias here
        # dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        # A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # # SSM step
        # print(selective_state_update,"277")
        # if selective_state_update is None:
        #     # Discretize A and B
        #     dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
        #     dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        #     dB = torch.einsum("bd,bn->bdn", dt, B)
        #     S_t = torch.einsum("bd,bn->bdn", dA, dB) + torch.einsum("bd,dn->bdn", dB, x)  # Cross-feature interaction
        #     ssm_state = S_t.mean(dim=0)
        #     # ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        #     y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        #     y = y + self.D.to(dtype) * x
        #     y = y * self.act(z)  # (B D)
        #     print("288")
        # else:
        #     y = selective_state_update(
        #         ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
        #     )

        # out = self.out_proj(y)
        # # print("274",ssm_state)
        # return out.unsqueeze(1), conv_state, ssm_state
        dtype = hidden_states.dtype
        # 不再有时间步限制，允许多个时间步
        # 原本代码会在这里assert hidden_states.shape[1] == 1, 被去掉
        xz = self.in_proj(hidden_states)  # (B T D)
        x, z = xz.chunk(2, dim=-1)  # (B T D)
        # print(xz.shape,x.shape,z.shape,hidden_states.shape,conv_state.shape)
        
        if causal_conv1d_update is None:
            # Ensure conv_state has an additional dimension for time step
            conv_state = conv_state.unsqueeze(-1).repeat(x.shape[0], 1, 1, hidden_states.shape[1])  # (B D W T)

            # Roll conv_state (shift by -1 along the time dimension)
            conv_state = torch.roll(conv_state, shifts=-1, dims=-2)  # Update state (B D W T)
            
            x = x.permute(0, 2, 1)
            # print(conv_state.shape, x.shape)
            # Update conv_state with new information from x (last dimension is sequence length)
            conv_state[:, :, :, :] = x.unsqueeze(-2)  # Update entire sequence
            conv_state = conv_state[:,:,:,:].mean(dim=-1).mean(dim=0)
            # Apply convolution
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-2)  # (B T D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)

        else:
            # Use causal_conv1d_update for multi-time step updates
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )
        x_db = self.x_proj(x)  # (B T dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B T d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))  # (B T d_inner d_state)
            dB = torch.einsum("bd,bn->bdn", dt, B)  # (B T d_state)
            
            # 计算 S_t 时，按时间步处理
            S_t = torch.einsum("bd,bn->bdn", dA, dB) + torch.einsum("bd,dn->bdn", dB, x)  # Cross-feature interaction
            ssm_state = S_t.mean(dim=0)  # 对多个时间步的状态进行平均

            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)
        else:
            # 保留 selective_state_update 函数
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out, conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        # if 0 in inference_params.key_value_memory_dict:
            # print(inference_params.key_value_memory_dict.keys())
            # print(torch.sum(inference_params.key_value_memory_dict[0][0]),torch.sum(inference_params.key_value_memory_dict[0][1]))
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                # batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                # batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.inference_params = InferenceParams()
    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        # print(hidden_states.shape)
        batch, seqlen, dim = hidden_states.shape
        # inference_params = self.inference_params
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            # print(conv_state.shape,ssm_state.shape)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out
        # conv_state = torch.zeros(batch, dim, seqlen, device=hidden_states.device)
        # ssm_state = torch.zeros(batch, seqlen, dim, device=hidden_states.device)

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # print(x.shape,z.shape,"198")
            # Compute short convolution
            if conv_state is not None:
                x_padded = F.pad(x, (self.d_conv - x.shape[-1], 0))  # 填充x的最后一个维度
                # print(self.d_conv - x.shape[-1],x_padded.shape)
                # conv_state.copy_(x_padded.mean(dim=0))
                conv_state = x_padded.mean(dim=0)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            

            # Feature interaction in the SSM state update
            # Initialize S_t as a zero matrix (or could use a learnable init)
            S_t = torch.zeros_like(A)  # Assuming A is (d_state, d_state)
            S_t.copy_(A)
            # Compute the new S_t with the interaction term
            # Compute the new S_t with the interaction term (A @ S_t @ A.T + B @ x @ x.T)
            S_t_weighted =  A.T @ S_t  # Ensure this operation is valid by aligning dimensions
            x_interaction = B @ x.T.permute(2,0,1)
            x_interaction = torch.matmul(S_t_weighted.unsqueeze(0),x_interaction)
            x_interaction = torch.matmul(x_interaction, x.T.permute(2,1,0))
            # Compute the output using the new S_t
            C_T = C.T.permute(2, 0, 1)  # 调整C.T的维度为 [20315, 16, 5]
            S_t_reshaped = S_t.T.unsqueeze(0)  # 将S_t的维度从[64, 16]变为[1, 16, 64]
            tr_term = torch.matmul(C_T, S_t_reshaped)
            y_t = tr_term + (self.D @ x).unsqueeze(-1)
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                # ssm_state.copy_(last_state.mean(dim=0))
                ssm_state = last_state.mean(dim=0)
            # print(y.shape)
            # y = y + y_t.permute(0,2,1)
            # y = y_t.permute(0,2,1)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
            # print(out.shape)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        # print("248**",hidden_states.shape)
        # dtype = hidden_states.dtype
        # # assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        # xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        # x, z = xz.chunk(2, dim=-1)  # (B D)

        # # Conv step
        # if causal_conv1d_update is None:
        #     conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
        #     print(x.shape,conv_state.shape)
        #     conv_state[:, :, -1] = x
        #     x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
        #     if self.conv1d.bias is not None:
        #         x = x + self.conv1d.bias
        #     x = self.act(x).to(dtype=dtype)
        # else:
        #     x = causal_conv1d_update(
        #         x,
        #         conv_state,
        #         rearrange(self.conv1d.weight, "d 1 w -> d w"),
        #         self.conv1d.bias,
        #         self.activation,
        #     )

        # x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        # dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # # Don't add dt_bias here
        # dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        # A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # # SSM step
        # print(selective_state_update,"277")
        # if selective_state_update is None:
        #     # Discretize A and B
        #     dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
        #     dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        #     dB = torch.einsum("bd,bn->bdn", dt, B)
        #     S_t = torch.einsum("bd,bn->bdn", dA, dB) + torch.einsum("bd,dn->bdn", dB, x)  # Cross-feature interaction
        #     ssm_state = S_t.mean(dim=0)
        #     # ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        #     y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        #     y = y + self.D.to(dtype) * x
        #     y = y * self.act(z)  # (B D)
        #     print("288")
        # else:
        #     y = selective_state_update(
        #         ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
        #     )

        # out = self.out_proj(y)
        # # print("274",ssm_state)
        # return out.unsqueeze(1), conv_state, ssm_state
        dtype = hidden_states.dtype
        # 不再有时间步限制，允许多个时间步
        # 原本代码会在这里assert hidden_states.shape[1] == 1, 被去掉
        xz = self.in_proj(hidden_states)  # (B T D)
        x, z = xz.chunk(2, dim=-1)  # (B T D)
        print(xz.shape,x.shape,z.shape,hidden_states.shape,conv_state.shape)
        
        if causal_conv1d_update is None:
            # Ensure conv_state has an additional dimension for time step
            conv_state = conv_state.unsqueeze(-1).repeat(x.shape[0], 1, 1, hidden_states.shape[1])  # (B D W T)

            # Roll conv_state (shift by -1 along the time dimension)
            conv_state = torch.roll(conv_state, shifts=-1, dims=-2)  # Update state (B D W T)
            
            x = x.permute(0, 2, 1)
            # print(conv_state.shape, x.shape)
            # Update conv_state with new information from x (last dimension is sequence length)
            conv_state[:, :, :, :] = x.unsqueeze(-2)  # Update entire sequence
            conv_state = conv_state[:,:,:,:].mean(dim=-1).mean(dim=0)
            # Apply convolution
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-2)  # (B T D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)

        else:
            # Use causal_conv1d_update for multi-time step updates
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )
        x_db = self.x_proj(x)  # (B T dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B T d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))  # (B T d_inner d_state)
            dB = torch.einsum("bd,bn->bdn", dt, B)  # (B T d_state)
            
            # 计算 S_t 时，按时间步处理
            S_t = torch.einsum("bd,bn->bdn", dA, dB) + torch.einsum("bd,dn->bdn", dB, x)  # Cross-feature interaction
            ssm_state = S_t.mean(dim=0)  # 对多个时间步的状态进行平均

            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)
        else:
            # 保留 selective_state_update 函数
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out, conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        # if 0 in inference_params.key_value_memory_dict:
            # print(inference_params.key_value_memory_dict.keys())
            # print(torch.sum(inference_params.key_value_memory_dict[0][0]),torch.sum(inference_params.key_value_memory_dict[0][1]))
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                # batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                # batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

class Add_Norm(nn.Module):
    def __init__(self, d_model, dropout, residual, drop_flag=0):
        super(Add_Norm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.drop_flag = drop_flag

    def forward(self, x, res):
        if self.residual:
            x = x + res
        x = self.norm(x)
        if self.drop_flag:
            x = self.dropout(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, mamba_forward, mamba_backward=None, d_model=128, d_ff=256, dropout=0.2, 
                 activation="relu", bi_dir=False, residual=True):
        super(EncoderLayer, self).__init__()
        self.bi_dir = bi_dir
        self.mamba_forward = mamba_forward
        self.residual = residual
        self.addnorm_for = Add_Norm(d_model, dropout, residual)

        if self.bi_dir and mamba_backward is not None:
            self.mamba_backward = mamba_backward
            self.addnorm_back = Add_Norm(d_model, dropout, residual)
        
        self.ffn = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        )
        self.addnorm_ffn = Add_Norm(d_model, dropout, residual)
        self.inference_params = InferenceParams()
    def forward(self, x):
        # x: [B, S, D]
        # print(self.inference_params)
        output_forward = self.mamba_forward(x)
        output_forward = self.addnorm_for(output_forward, x)

        if self.bi_dir:
            output_backward = self.mamba_backward(x.flip(dims=[1])).flip(dims=[1])
            output_backward = self.addnorm_back(output_backward, x)
            output = output_forward + output_backward
        else:
            output = output_forward
    
        temp = output
        output = self.ffn(output.transpose(-1, 1)).transpose(-1, 1)
        output = self.addnorm_ffn(output, temp)
        return output

class Encoder(nn.Module):
    def __init__(self, mamba_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.norm = norm_layer
        self.inference_params = InferenceParams()
    def forward(self, x):
        for mamba_block in self.mamba_layers:
            # print(mamba_block,x.shape)
            x = mamba_block(x)

        if self.norm is not None:
            x = self.norm(x)
        return x
    
class MambaTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, bi_dir=False):
        super(MambaTimeSeriesModel, self).__init__()
        # print("???878",self.input_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.inference_params = InferenceParams()
        # 定义 Mamba 编码器层
        mamba_layers = []
        for _ in range(num_layers):
            mamba_forward = Mamba(
                d_model=input_dim,
                d_state=16,
                d_conv=4,
                expand=2,
                layer_idx=0
            )
            mamba_backward = Mamba(
                d_model=input_dim,
                d_state=16,
                d_conv=4,
                expand=2,
                layer_idx=1
            ) if bi_dir else None

            layer = EncoderLayer(
                mamba_forward=mamba_forward,
                mamba_backward=mamba_backward,
                d_model=input_dim,
                d_ff=hidden_dim,
                bi_dir=bi_dir
            )
            mamba_layers.append(layer)
        self.encoder = Encoder(mamba_layers)
        mamba_layers = []
        for _ in range(num_layers):
            mamba_forward = Mamba1(
                d_model=input_dim,
                d_state=16,
                d_conv=4,
                expand=2,
                layer_idx=0
            )
            mamba_backward = Mamba(
                d_model=input_dim,
                d_state=16,
                d_conv=4,
                expand=2,
                layer_idx=1
            ) if bi_dir else None

            layer = EncoderLayer(
                mamba_forward=mamba_forward,
                mamba_backward=mamba_backward,
                d_model=input_dim,
                d_ff=hidden_dim,
                bi_dir=bi_dir
            )
            mamba_layers.append(layer)
        self.encoder1 = Encoder(mamba_layers)
        # 输出层：使用全局平均池化（兼容不同长度）
        self.output_layer = nn.Linear(input_dim, 2)

    def forward(self, x):
        # print("???939",self.input_dim)
        # x: [B, S, D]
        # x = torch.cat([self.encoder(x) ,self.encoder1(x)],dim=-1)
        x = self.encoder1(x)
        # x = self.encoder1(x)
        # print(self.encoder)
        # print(x[0])
        # 输出层
        # print(self.output_layer,"?????",x.shape)
        output = self.output_layer(x)  # [B, 1]
        return output,x

class Configs(object):
    ab = 0
    modes = 32
    mode_select = 'random'
    # version = 'Fourier'
    version = 'Wavelets'
    moving_avg = [12, 24]
    L = 1
    base = 'legendre'
    cross_activation = 'tanh'
    seq_len = 125
    label_len = 48
    pred_len = 1
    output_attention = True
    enc_in = 3
    dec_in = 3
    d_model = 16
    embed = 'timeF'
    dropout = 0.05
    freq = 'h'
    factor = 1
    n_heads = 8
    d_ff = 16
    e_layers = 2
    d_layers = 1
    c_out = 1
    activation = 'gelu'
    wavelet = 0
    fc_dropout = 0.2
    head_dropout = 0
    individual = 1
    patch_len = 7
    stride = 2
    padding_patch = 'end'
    revin = True
    affine = True
    subtract_last = True
    decomposition = 0
    kernel_size = 3

# def calculate_mcc(y_true, y_pred):
#     """
#     计算 MCC (Matthew's Correlation Coefficient) 指标。
    
#     Args:
#         y_true (list or np.ndarray): 真实标签。
#         y_pred (list or np.ndarray): 预测标签。
    
#     Returns:
#         float: MCC 值。
#     """
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
#     # print(y_true,y_pred)
#     # print(tn, fp, fn, tp)
#     numerator = (tp * tn) - (fp * fn)
#     denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
#     return numerator / denominator if denominator != 0 else 0.0

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch[0]['time']) ==0:
        return None
    if len(batch) == 0:
        return None
    times = [item["time"] for item in batch]
    stock_features = [item["stock_features"] for item in batch]
    labels = [item["label"] for item in batch]
    texts = [item["texts"] for item in batch]
    amplis = [item["ampl"] for item in batch]
    return {
        "time": times,
        "stock_features": torch.nn.utils.rnn.pad_sequence(stock_features, batch_first=True, padding_value=0),
        "texts": torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0),
        "label": torch.stack(labels),
        "amp": torch.stack(amplis)
    }
def calculate_mcc(y_true, y_pred):
    """
    计算 MCC (Matthew's Correlation Coefficient) 指标。
    
    Args:
        y_true (list or np.ndarray): 真实标签。
        y_pred (list or np.ndarray): 预测标签。
    
    Returns:
        float: MCC 值。
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator != 0 else 0.0


class TimeAxisAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.lnorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.tensor, rt_attn=False):
        # x: (D, W, L)
        o, (h, _) = self.lstm(x) # o: (D, W, H) / h: (1, D, H)
        score = torch.bmm(o, h.permute(1, 2, 0)) # (D, W, H) x (D, H, 1)
        tx_attn = torch.softmax(score, 1).squeeze(-1)  # (D, W)
        context = torch.bmm(tx_attn.unsqueeze(1), o).squeeze(1)  # (D, 1, W) x (D, W, H)
        normed_context = self.lnorm(context)
        if rt_attn:
            return normed_context, tx_attn
        else:
            return normed_context, None
            
class DataAxisAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, drop_rate=0.1):
        super().__init__()
        self.multi_attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            nn.ReLU(),
            nn.Linear(4*hidden_size, hidden_size)
        )
        self.lnorm1 = nn.LayerNorm(hidden_size)
        self.lnorm2 = nn.LayerNorm(hidden_size)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, hm: torch.tensor, rt_attn=False):
        # Forward Multi-head Attention
        residual = hm
        # hm_hat: (D, H), dx_attn: (D, D) 
        hm_hat, dx_attn = self.multi_attn(hm, hm, hm)
        hm_hat = self.lnorm1(residual + self.drop_out(hm_hat))

        # Forward FFN
        residual = hm_hat
        # hp: (D, H)
        hp = torch.tanh(hm + hm_hat + self.mlp(hm + hm_hat))
        hp = self.lnorm2(residual + self.drop_out(hp))

        if rt_attn:
            return hp, dx_attn
        else:
            return hp, None

class TimeTextModel(nn.Module):
    def __init__(self, stock_feature_dim, hidden_dim):
        super(TimeTextModel,self).__init__()
        self.stock_lstm = nn.LSTM(128, hidden_dim, batch_first=True, bidirectional=False)
        self.mlp = nn.Sequential(nn.Linear(64+64,64),nn.ReLU(),nn.Linear(64,2))
        self.ln = nn.Linear(11+5,128)
        self.test_mamba = MambaTimeSeriesModel(64, 5, 2, False)
        
    def forward(self, stock_features, window_feat):
        stock_features = stock_features.squeeze(0)
        stock_features = torch.cat([stock_features,window_feat],dim=-1)
        stock_features = self.ln(stock_features)
        
        stock_output, _ = self.stock_lstm(stock_features)

        pred1, emb = self.test_mamba(stock_output)
        emb1 = (emb[:,-1,:]).squeeze(1).unsqueeze(0)
        # ,(stock_output[:,-1,:]).squeeze(1).unsqueeze(0)
        output = self.mlp(torch.cat([emb1],dim=-1))
        return output


class ContextNormalization(nn.Module):
    """
    According to Eq. 3 from [1]
    """
    def __init__(self, num_stocks, hidden_dim):
        super(ContextNormalization, self).__init__()
        # Gamma and beta have the same size as the input h_{u i}
        self.gamma = nn.Parameter(torch.ones(num_stocks, hidden_dim))
        self.beta = nn.Parameter(torch.zeros(num_stocks, hidden_dim))

    def forward(self, x):
        # x: [batch_size, num_stocks, hidden_dim]
        # Calculate the mean and standard deviation across all stocks and the entire hidden space
        # for each batch element separately.
        # This aggregates statistics across the 'num_stocks' and 'hidden_dim' dimensions.
        mean = x.mean(dim=[1, 2], keepdim=True)  # Taking mean across stocks and hidden_dim
        std = x.std(dim=[1, 2], keepdim=True)    # Standard deviation across the same dimensions
        
        # Normalize the input x using the calculated mean and std.
        normalized_x = (x - mean) / (std + 1e-9)
        # print(self.gamma.shape,normalized_x.shape,self.beta.shape,"1171**",x.shape)
        return self.gamma * normalized_x + self.beta


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, lstm_out):
        # print(lstm_out.shape)
        # lstm_out shape: [batch_size * num_stocks, seq_len, hidden_dim]
        # Extract the last hidden state as the query vector
        h_T = lstm_out[:, -1, :].unsqueeze(2)  # Shape: [batch_size * num_stocks, hidden_dim, 1]
        
        # Compute attention scores
        # Here, we perform batch matrix multiplication between lstm_out and h_T for each sequence
        # and apply softmax to get attention weights
        # print(lstm_out.shape,h_T.shape)
        attention_scores = torch.bmm(lstm_out, h_T).squeeze(2)  # [batch_size * num_stocks, seq_len]
        alpha_i = F.softmax(attention_scores, dim=1).unsqueeze(2)  # [batch_size * num_stocks, seq_len, 1]
        
        # Compute context vector as a weighted sum of LSTM outputs
        context_vector = torch.sum(alpha_i * lstm_out, dim=1)  # [batch_size * num_stocks, hidden_dim]
        
        return context_vector
    
class AttentiveLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_stocks, num_layers=1,mamba_ly=2):
        super(AttentiveLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_transform = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,bidirectional=True)
        self.attention = Attention(hidden_dim)  
        self.mamba_ct = MambaTimeSeriesModel(hidden_dim, 5, mamba_ly, False)
        self.context_norm = ContextNormalization(num_stocks, hidden_dim*2)

    def forward(self, x):
        batch_size, seq_len, num_stocks, input_dim = x.size()
        x_reshaped = x.reshape(batch_size * num_stocks, seq_len, input_dim)
        transformed_features = torch.tanh(self.feature_transform(x_reshaped))
        lstm_out, _ = self.lstm(transformed_features)
        pred, mamba_out = self.mamba_ct(transformed_features)
        lstm_out1 = torch.cat([lstm_out,mamba_out],dim=-1)
        # Apply attention mechanism
        context_vector = self.attention(lstm_out)
        # Reshape context_vector back to the original batch and stocks structure
        context_vector = context_vector.view(batch_size, num_stocks, self.hidden_dim*2)
        # Apply context normalization
        normalized_context_vectors = self.context_norm(context_vector)
        return normalized_context_vectors


    
class DataAxisSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(DataAxisSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == self.hidden_dim, "hidden_dim must be divisible by num_heads"

        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        num_stocks, batch_size, _ = x.size()
        
        # Eq. (6)
        Q = self.Wq(x).view(num_stocks, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        K = self.Wk(x).view(num_stocks, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        V = self.Wv(x).view(num_stocks, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        
        # Calculating attention scores
        # Eq. (7)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim ** 0.5
        S = F.softmax(energy, dim=-1)

        out = torch.matmul(S, V).permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, num_stocks, self.hidden_dim)

        # MLP is applied to the sum of the original input and the aggregated attention output
        # Eq. (8) from [1]
        out_mlp = self.mlp(x.reshape(batch_size, num_stocks, self.hidden_dim) + out)
        out_final = torch.tanh(x.reshape(batch_size, num_stocks, self.hidden_dim) + out + out_mlp) # [batch_size, num_stocks, hidden_dim]

        return out_final


class DTML(nn.Module):
    def __init__(self, beta_hyp, input_dim, hidden_dim, num_stocks, num_heads, num_layers,mamba_ly):
        super(DTML, self).__init__()
        self.beta_hyp = beta_hyp  # Hyperparameter determining the weight of the global market context
        self.beta_hyp = nn.Parameter(torch.tensor(beta_hyp)).to(device)
        # Updated LSTM model with normalization and attention mechanism
        self.stock_lstm_model = AttentiveLSTM(input_dim, hidden_dim, num_stocks, num_layers, mamba_ly)
        # Direct use of DataAxisSelfAttention
        self.self_attention = DataAxisSelfAttention(hidden_dim*2, num_heads)  # Assuming a single head of attention
        self.final_linear = nn.Linear(hidden_dim*2, 1)  # Linear layer for prediction
    def forward(self, x):
        normalized_context_vectors = self.stock_lstm_model(x) # (batch_size, seq_len, num_stocks, input_dim) -> (batch_size, num_stocks, hidden_dim)
        # Extract the global market context for the global_context_inde
        global_market_context = torch.mean(normalized_context_vectors,dim=1).unsqueeze(1)
        # Calculate multi-level contexts for each stock (Eq. 4)
        # h^m_u = h^c_u + beta * h^i + self.beta_hyp * global_market_context
        multi_level_contexts = normalized_context_vectors + self.beta_hyp * global_market_context
        # Apply self-attention mechanism and non-linear transformation
        multi_level_contexts = multi_level_contexts.permute(1, 0, 2)  # Preparing data for self-attention
        attention_output = self.self_attention(multi_level_contexts) # [batch_size, num_stocks, hidden_dim]
        multi_level_contexts = multi_level_contexts.permute(1, 0, 2)
        final_predictions = self.final_linear(multi_level_contexts)
        return final_predictions, attention_output
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict
class FuseContext1(nn.Module):
    def __init__(self, dims):
        super(FuseContext1, self).__init__()
        # self.embedding = nn.Embedding(59042, 200)
        # embed = 200
        # self.text_fc = nn.Linear(144, hidden_dim)
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(1, 48, (k, embed)) for k in (2, 3, 4)]
        # )
        # self.conv = nn.Conv1d(
        #     in_channels=11 * hidden_dim,  # 合并 num_docs 和 hidden_dim
        #     out_channels=hidden_dim*2,  # 输出通道
        #     kernel_size=3,
        #     padding=3 // 2  # 保持长度不变
        # )
        # self.hidden_dim = hidden_dim
        # self.pool = nn.MaxPool1d(3)
        self.model = AutoModel.from_pretrained("/home/zhaokx/method2/bertweet")
        self.mlp = nn.Sequential(nn.Linear(dims*3+hidden_dim//2+10,dims+hidden_dim),nn.ReLU(),nn.Linear(dims+hidden_dim,1))
    def conv_and_pool(self, x, conv):
        """卷积 + 最大池化"""
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    def forward(self, x_emb, texts_input_ids,text_attn_mask):
        tra_tokenized = {
            "input_ids": texts_input_ids,
            "attention_mask": text_attn_mask,
        }
        with torch.no_grad():
            outputs = self.model(**inputs)
        print(outputs.shape)
        # batch_size, num_step, num_docs, max_words = texts.size()
        # # print(texts,texts.shape)
        # texts = texts.to(device)
        # # 填充值向量
        # pad_vector = torch.full((40,), 59041).to(device)  # 形状为 (40,)
        # # 生成掩码
        # mask = torch.all(texts == pad_vector.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device), dim=-1)
        # mask = ~mask
        # mask_expanded = mask.unsqueeze(-1)  # (1, 398, 10, 1)
        
        # # print(mask.shape, mask)
        # texts = texts.reshape(-1, max_words)
        
        # xtext = self.embedding(texts.to(device)).unsqueeze(1)
        
        # text_repr = torch.cat([self.conv_and_pool(xtext, conv) for conv in self.convs], 1)
        # text_repr2 = self.text_fc(text_repr)

        # text_repr = text_repr2.view(batch_size, num_step, num_docs, -1)
        # mask_expanded = mask_expanded.expand_as(text_repr)  # (1, 398, 10, 64)
        # text_repr_masked_mean = torch.where(mask_expanded, text_repr, torch.tensor(0.0).to(device)) 
        # valid_count = mask.sum(dim=-1, keepdim=True)  # (1, 398, 1, 1)
        # valid_count = valid_count.clamp(min=1)  # 防止除以零
        # # print(text_repr_masked_mean,valid_count)
        # # 求均值/ valid_count  # (1, 398, 64)
        # mean_result = text_repr_masked_mean.sum(dim=-2) 
        # # print(mean_result.shape,"1312")
        # # print(text_repr.shape)
        # text_repr = text_repr[:,:,:10,:]
        # text_repr = torch.cat([mean_result, text_repr.view(batch_size, num_step, 10 * text_repr2.shape[-1])],dim=-1).permute(0,2,1)
        # # print(text_repr.shape,"1316",self.conv(text_repr).shape)
        # output = self.conv(text_repr).permute(0,2,1)
        # output = self.pool(output)
        # # print(output.shape,x_emb.shape)
        # # text_repr = text_repr2.view(1, batch_size, num_docs, -1)
        # # text_repr = text_repr.view(1, batch_size, num_docs * text_repr2.shape[-1]).permute(0,2,1)  # [B, seq_length , hidden_dim * num_docs] # [B, hidden_dim * num_docs, seq_length]
        final_predictions = self.mlp(torch.cat([x_emb,output],dim=-1))
        return final_predictions

class CrossAttentionFusion1(nn.Module):
    def __init__(self, x_emb_dim, text_dim, num_heads=4):
        super(CrossAttentionFusion1,self).__init__()
        self.query = nn.Linear(x_emb_dim, x_emb_dim)  # Query: 时序特征
        self.key = nn.Linear(text_dim, x_emb_dim)      # Key: 文本特征
        self.value = nn.Linear(text_dim, x_emb_dim)    # Value: 文本特征
        self.multihead_attn = nn.MultiheadAttention(x_emb_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(x_emb_dim*3)

    def forward(self, x_emb, text_feat):
        # x_emb: [batch, seq_len, x_emb_dim] (时序特征)
        # text_feat: [batch, seq_len, text_dim] (文本特征)
        
        q = self.query(x_emb)  # [batch, seq_len, x_emb_dim]
        k = self.key(text_feat)  # [batch, seq_len, x_emb_dim]
        v = self.value(text_feat)  # [batch, seq_len, x_emb_dim]
        
        # 交叉注意力：时序特征查询文本特征
        attn_output, _ = self.multihead_attn(q, k, v)
        # print(x_emb.shape,attn_output.shape)
        fused = self.norm(torch.cat([x_emb ,attn_output],dim=-1))  # 残差连接
        
        return fused
# class FuseContext(nn.Module):
#     def __init__(self, dims):
#         super(FuseContext, self).__init__()
#         self.embedding = nn.Embedding(59042, 200)
#         embed = 200
#         self.ratio_embed = nn.Sequential(
#             nn.Linear(1, embed//2),
#             nn.ReLU(),

#             nn.Linear(embed//2, embed)
#         )

#         self.sum_embed = nn.Sequential(
#             nn.Linear(1, embed//2),
#             nn.ReLU(),
#             nn.Linear(embed//2, embed)
#         )
#         self.text_fc = nn.Linear(144, hidden_dim)
#         self.text_fc_sc = nn.Linear(150, 50)
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1, 48, (k, embed)) for k in (2, 3, 4)]
#         )
#         self.convs1 = nn.ModuleList(
#             [nn.Conv2d(1, 50, (k, 50)) for k in (2, 3, 4)]
#         )
#         self.conv = nn.Conv1d(
#             in_channels=20 * hidden_dim,  # 合并 num_docs 和 hidden_dim
#             out_channels=hidden_dim*2,  # 输出通道
#             kernel_size=3,
#             padding=3 // 2  # 保持长度不变
#         )
#         self.conv2 = nn.Conv1d(
#             in_channels=20 * hidden_dim,  # 合并 num_docs 和 hidden_dim
#             out_channels=hidden_dim*2,  # 输出通道
#             kernel_size=3,
#             padding=3 // 2  # 保持长度不变
#         )
#         self.conv1 = nn.Conv1d(
#             in_channels=500,  # 合并 num_docs 和 hidden_dim
#             out_channels=hidden_dim*2,  # 输出通道
#             kernel_size=3,
#             padding=3 // 2  # 保持长度不变
#         )
#         self.conv_sc = nn.Conv1d(
#             in_channels=20 * 50,  # 合并 num_docs 和 hidden_dim
#             out_channels=50*2,  # 输出通道
#             kernel_size=3,
#             padding=3 // 2  # 保持长度不变
#         )
#         self.hidden_dim = hidden_dim
#         self.pool = nn.MaxPool1d(2)
#         self.mlp = nn.Sequential(nn.Linear(392,200),nn.LeakyReLU(0.03),nn.Linear(200,150),nn.LeakyReLU(0.03),nn.Linear(150,50),nn.LeakyReLU(0.03),nn.Linear(50,1))
#         self.score_emb = nn.Linear(10,50)
#         self.window_size = 5
#         self.ln = nn.Linear(242,242)
#         self.stock_lstm1 = nn.LSTM(242, 200, batch_first=True, bidirectional=False)
#         self.score_pad_emb = nn.Embedding(11,50)
#         self.cross_attn = CrossAttentionFusion1(x_emb_dim=192, text_dim=50)
#     def conv_and_pool(self, x, conv):
#         """卷积 + 最大池化"""
#         x = F.relu(conv(x)).squeeze(3)
#         x = F.dropout(x, p=0.2, training=self.training)
#         x = F.max_pool1d(x, x.size(2)).squeeze(2)
#         return x
#     def forward(self, x_emb, texts, ratios, sums, texts2, score, score_pad):
#         batch_size, num_step, num_docs, max_words = texts.size()
#         texts = texts.reshape(-1, max_words)
#         xtext = self.embedding(texts.to(device)).unsqueeze(1)
#         ratios = ratios.reshape(-1, max_words, 1)
#         sums = sums.reshape(-1, max_words, 1)
#         ratio_embed = self.ratio_embed(ratios.to(device)).unsqueeze(1)
#         sum_embed = self.sum_embed(sums.to(device)).unsqueeze(1)
#         # xtext = xtext + ratio_embed
#         xtext = ratio_embed
#         text_repr = torch.cat([self.conv_and_pool(xtext, conv) for conv in self.convs], 1)
#         text_repr2 = self.text_fc(text_repr)
#         text_repr = text_repr2.view(batch_size, num_step, num_docs, -1)
#         text_repr1 = text_repr[:,:,:20,:]
#         text_repr = torch.cat([text_repr1.view(batch_size, num_step, 20 * text_repr2.shape[-1])],dim=-1).permute(0,2,1)
#         output = self.conv(text_repr).permute(0,2,1)
#         output_txt = self.pool(output)
#         batch_size, num_step, num_docs, max_words = score_pad.permute(1,0,2,3).size()
#         score_pad = score_pad[:,:,:20,:]
#         score_pad = score_pad.reshape(-1, max_words).long()
#         score_pad_emb = self.score_pad_emb(score_pad).unsqueeze(1)
#         score_pad_emb = torch.cat([self.conv_and_pool(score_pad_emb, conv) for conv in self.convs1], 1)
#         sc_repr2 = self.text_fc_sc(score_pad_emb)
#         sc_repr = sc_repr2.view(batch_size, num_step, 20, -1)
#         score_pad_emb = torch.cat([sc_repr.view(batch_size, num_step, 20 * sc_repr2.shape[-1])],dim=-1).permute(0,2,1)
#         output = self.conv_sc(score_pad_emb).permute(0,2,1)
#         output = self.pool(output)
#         score_seq = self.score_emb(score)
#         # text_feat = torch.cat([score_seq, output.permute(1,0,2)], dim=-1)  # 合并文本特征
#         fused_feat = self.cross_attn(x_emb, score_seq) 
#         # text_feat = torch.cat([score_seq, output.permute(1,0,2)], dim=-1)  # 合并文本特征
#         # print(x_emb.shape,score_seq.shape,output.permute(1,0,2).shape)
#         # fused_feat = cross_attn(x_emb, text_feat)
#         three_emb = torch.cat([x_emb,score_seq],dim=-1)
#         # print(three_emb.shape)
#         # three_emb = fused_feat
#         three_emb = self.ln(three_emb)
#         lstm_outputs = []
#         for i in range(self.window_size - 1):
#             window = three_emb[:,:i+1:,:]
#             stock_output, _ = self.stock_lstm1(window)
#             lstm_outputs.append(stock_output[:,-1,:]) 
#         for i in range(self.window_size - 1, three_emb.size(1)):
#             window = three_emb[:, i - self.window_size + 1:i + 1, :]
#             stock_output, _ = self.stock_lstm1(window)
#             lstm_outputs.append(stock_output[:, -1, :])
#         stock_output1 = torch.stack(lstm_outputs, dim=1)
#         final_predictions = self.mlp(torch.cat([x_emb,stock_output1],dim=-1))
#         return final_predictions, torch.cat([x_emb,stock_output1],dim=-1)
class FuseContext(nn.Module):
    def __init__(self, dims):
        super(FuseContext, self).__init__()
        self.embedding = nn.Embedding(59042, 200)
        embed = 200
        self.mlp = nn.Sequential(nn.Linear(392,200),nn.LeakyReLU(0.03),nn.Linear(200,150),nn.LeakyReLU(0.03),nn.Linear(150,50),nn.LeakyReLU(0.03),nn.Linear(50,1))
        self.score_emb = nn.Linear(10,50)
        self.window_size = 5
        self.stock_lstm1 = nn.LSTM(242, 200, batch_first=True, bidirectional=False)
    def conv_and_pool(self, x, conv):
        """卷积 + 最大池化"""
        x = F.relu(conv(x)).squeeze(3)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    def forward(self, x_emb, texts, ratios, sums, texts2, score, score_pad):
        score_seq = self.score_emb(score)
        three_emb = torch.cat([x_emb,score_seq],dim=-1)
        lstm_outputs = []
        for i in range(self.window_size - 1):
            window = three_emb[:,:i+1:,:]
            stock_output, _ = self.stock_lstm1(window)
            lstm_outputs.append(stock_output[:,-1,:]) 
        for i in range(self.window_size - 1, three_emb.size(1)):
            window = three_emb[:, i - self.window_size + 1:i + 1, :]
            stock_output, _ = self.stock_lstm1(window)
            lstm_outputs.append(stock_output[:, -1, :])
        stock_output1 = torch.stack(lstm_outputs, dim=1)
        final_predictions = self.mlp(torch.cat([x_emb,stock_output1],dim=-1))
        return final_predictions
def objective(trial):
    set_seed(43)
    best_test_acc = 0
    try:
        layers = [1,2,3]
        heads = [1,2,3]
        dims = [32,48,64,96,128]
        pos_weis = [0.7,0.8,0.9,1.0]
        layers = trial.suggest_int("layers",1,4)
        heads = trial.suggest_int("heads",1,4)
        dims = trial.suggest_categorical("dims",[32,48,64])
        pos_weis = trial.suggest_categorical("pos_weis",[0.7,0.8,0.9,1.0])
        learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
        mamba_ly = trial.suggest_categorical("mamba_ly",[1,2,3])
        
        gl_wei = trial.suggest_categorical("gl_wei",[0.1,0.2,0.3])

        # layers = 3
        # heads = 4
        # dims = 32
        # pos_weis = 0.8
        # learning_rate = 0.005
        # gl_wei = 0.2
        # mamba_ly = 2

        dtml = DTML(gl_wei, 16, dims, max1+1, heads, layers,mamba_ly).to(device)
        # dtml = torch.load("/home/zhaokx/method2/ckpt_new/ftse1003_4_32_0.8_0.005_0.2_model_time_m.pth",weights_only=False,map_location=device).to(device)
        optimizer = optim.Adam(dtml.parameters(), lr=learning_rate)
        # stocknet 0.1, 16, 48, max1+1, 2, 2 0.8
        # pos_weight = torch.tensor([0.5]).to(device)
        # pos_weight = torch.tensor([0.6]).to(device))
        # ftse 0.1, 16, 32, max1+1, 2, 2 0.8
        # ni225 0.1, 16, 32, max1+1, 2, 1 0.85
        # kdd17 0.1, 16, 32, max1+1, 2, 2 0.8
        # print(layers,heads,dims,pos_weis,learning_rate,gl_wei,mamba_ly)
        criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([pos_weis]).to(device))
        # criterion = nn.CrossEntropyLoss(weight = torch.tensor([0.47,0.53], dtype=torch.float32).to(device))
        best = 0
        loss_last = 0
        idx = 1000
        best_test_acc = 0 
        best_test_mcc = 0
        for epoch in range(num_epochs):
            print(epoch)
            correct2 = 0
            dtml.train()
            correct = 0
            count = 0
            correct = 0
            for j in range(0,len(expanded_tensor),idx):
                # print(j)
                optimizer.zero_grad()
                data_train = []
                label_train = []
                dt_train = expanded_tensor[j:j+idx,:,:].to(device)
                # dt_text = expanded_txt[j:j+idx,:,:,:].to(device)
                pred, train_emb = dtml(dt_train)
                dt_label = expanded_label[j:j+idx,:].to(device)
                lb_mask = []
                for i in range(len(dt_label)):
                    data_train.append(pred[i][(dt_label[i].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                    label_train.append(dt_label[i][(dt_label[i].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                data_train1 = torch.cat(data_train, dim=0)
                label_train1 = torch.cat(label_train, dim=0).squeeze(-1)
                predicted = (torch.sigmoid(data_train1) > 0.5).int()
                # predicted = torch.argmax(data_train1[(label_train1!=torch.tensor(2).to(device))],dim=1)
                # correct += (label_train1.to(device)[(label_train1!=torch.tensor(2).to(device))] == predicted.to(device)).sum().item()
                # count += len(predicted)
                loss = criterion(data_train1[(label_train1!=torch.tensor(2).to(device))].squeeze(1), label_train1.float()[(label_train1!=torch.tensor(2).to(device))])
                loss.backward()
                optimizer.step()
            train_emb1 = train_emb.reshape(-1, train_emb.size(2))  # [251 * 24, 128]
            indices = torch.arange(train_emb1.size(1)).repeat(train_emb1.size(0))  # [251 * 24]
            dtml.eval()
            data_eval = []
            label_eval = []
            
            with torch.no_grad():
                pred, eval_emb = dtml(expanded_tensor1.to(device))
                for i in range(len(expanded_label1)):
                    data_eval.append(pred[i][(expanded_label1[i].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                    label_eval.append(expanded_label1[i].to(device)[(expanded_label1[i].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                data_eval1 = torch.cat(data_eval, dim=0).to(device)
                label_eval1 = torch.cat(label_eval, dim=0).squeeze(-1).to(device)
                loss1 = criterion(data_eval1.squeeze(), label_eval1.float())

                eval_emb1 = eval_emb.reshape(-1, eval_emb.size(2))  # [251 * 24, 128]
                indices1 = torch.arange(eval_emb1.size(1)).repeat(eval_emb1.size(0))  # [251 * 24]
                predicted1 = (torch.sigmoid(data_eval1) > 0.5).int()
                # predicted1 = torch.argmax(data_eval1,dim=1)
                correct3 = (label_eval1.to(device) == predicted1.to(device).squeeze(1)).sum().item()
                data_test = []
                label_test = []
                pred, test_emb = dtml(expanded_tensor2.to(device))
                for i in range(len(expanded_label2)):
                    data_test.append(pred[i][(expanded_label2[i]!=torch.tensor(2)).squeeze(1).to(device)])
                    label_test.append(expanded_label2[i].to(device)[(expanded_label2[i]!=torch.tensor(2)).squeeze(1).to(device)])
                data_test2 = torch.cat(data_test, dim=0)
                label_test2 = torch.cat(label_test, dim=0).squeeze(-1)
                predicted = (torch.sigmoid(data_test2) > 0.5).int()
                # predicted = torch.argmax(data_test2,dim=1)
                correct2 = (label_test2.to(device) == predicted.to(device).squeeze(1)).sum().item()
                test_emb1 = test_emb.reshape(-1, test_emb.size(2))  # [251 * 24, 128]
                # if abs(loss-loss_last)<0.01:
                    # print(loss)
                if correct3/len(predicted1)>best:
                    best = correct3/len(predicted1)
                    torch.save(dtml,"./../checkpoints/"+dir1+"/"+dir1+"_".join([str(b) for b in [layers,heads,dims,pos_weis,learning_rate,gl_wei,mamba_ly]])+"_model_time_con.pth")
                    print("***",correct2/len(predicted),calculate_mcc(label_test2.detach().cpu().numpy(),predicted.detach().cpu().numpy()),correct3/len(predicted1),calculate_mcc(label_eval1.detach().cpu().numpy(),predicted1.detach().cpu().numpy()))
                    best_test_acc = correct2/len(predicted)
                loss_last = loss1
        print(best_test_acc)
    except Exception as e:
        # raise e
        print(e)
        pass
    return best_test_acc
import torch

def deduplicate_sequences(x, pad_vector):
    """
    Args:
        x: Tensor of shape [B, L, D], e.g., [398, 10, 40]
        pad_vector: Tensor of shape [D], e.g., [40]
    Returns:
        Tensor of shape [B, L, D] (填充后的去重结果)
    """
    B, L, D = x.shape
    device = x.device
    
    # 1. 过滤 pad_vector 并去重（每个序列独立处理）
    unique_seqs = []
    for seq in x:  # seq shape: [L, D]
        # 找出非 pad 的向量
        is_not_pad = ~torch.all(seq == pad_vector, dim=1)  # [L]
        non_pad_vecs = seq[is_not_pad]  # [N, D], N ≤ L
        
        # 去重
        unique_vecs = torch.unique(non_pad_vecs, dim=0)  # [M, D], M ≤ N
        
        # 填充到固定长度 L
        if len(unique_vecs) < L:
            padding = pad_vector.expand(L - len(unique_vecs), D).to(device)
            unique_vecs = torch.cat([unique_vecs, padding], dim=0)  # [L, D]
        
        unique_seqs.append(unique_vecs)
    
    # 2. 堆叠成 [B, L, D]
    return torch.stack(unique_seqs, dim=0)  # [398, 10, 40]

import torch

def deduplicate_sequences_fast(x, pad_vector):
    """
    向量化优化的去重+填充函数（比循环版本快5-10倍）
    Args:
        x: Tensor of shape [B, L, D], e.g., [398, 10, 40]
        pad_vector: Tensor of shape [D], e.g., [40]
    Returns:
        Tensor of shape [B, L, D] (去重后填充的结果)
    """
    B, L, D = x.shape
    device = x.device
    
    # 1. 标记非pad向量 (B, L)
    is_not_pad = ~torch.all(x == pad_vector, dim=2)  # [B, L]
    
    # 2. 为每个序列生成唯一ID (利用哈希技巧)
    # 将每个D维向量映射为一个唯一的整数ID
    flat_x = x.view(-1, D)  # [B*L, D]
    unique_ids = torch.zeros(B * L, dtype=torch.long, device=device)
    _, unique_idx = torch.unique(flat_x, return_inverse=True, dim=0)
    unique_ids = unique_idx.view(B, L)  # [B, L]
    
    # 3. 对每个序列，保留非pad且首次出现的向量
    output = pad_vector.repeat(B, L, 1)  # 初始化为全pad [B, L, D]
    for b in range(B):
        # 获取当前序列的非pad位置和唯一ID
        mask = is_not_pad[b]  # [L]
        ids = unique_ids[b]  # [L]
        
        # 找到每个唯一ID的第一次出现位置
        _, first_occur_idx = torch.unique(ids[mask], return_inverse=True)
        first_mask = torch.zeros_like(mask)
        first_mask[torch.where(mask)[0][first_occur_idx]] = True
        
        # 填充结果
        output[b][first_mask] = x[b][first_mask]
    
    return output
def test(expanded_label,expanded_label1,expanded_label2,expanded_txt1,expanded_txt2,expanded_tensor1,expanded_tensor2,expanded_ratio1,expanded_ratio2,expanded_sum1,expanded_sum2,expanded_txt_2,expanded_txt1_2,expanded_txt2_2, expanded_txt_sc, expanded_txt1_sc, expanded_txt2_sc,expanded_sc_pad,expanded_sc1_pad,expanded_sc2_pad):
    
    layers = 2
    heads = 4
    dims = 64
    pos_weis = 0.9
    learning_rate = 0.0005
    mamba_ly = 1
    # 2 3 32 1.0 0.0005 0.1
    gl_wei = 0.3
    # dtml = DTML(gl_wei, 16, dims, max1+1, heads, layers,mamba_ly).to(device)
    # dir1 = 'ni225'
    
    # dtml = torch.load("/home/zhaokx/method2/ckpt_new/kdd173_3_32_1.0_0.005_0.3_model_time_m.pth",weights_only=False,map_location=device).to(device)
    dtml = torch.load("/home/zhaokx/stock_text_pred/ckpt_new/stocknet3_1_64_1.0_0.0005_0.1_model_time_m.pth",weights_only=False,map_location=device).to(device)
    
    print(dtml)
    context_model = FuseContext(dims).to(device)
    # context_model = torch.load("context_model.pth", weights_only=False, map_location=device)
    optimizer = optim.Adam(dtml.parameters(), lr=learning_rate)
    # optimizer1 = optim.RMSprop(context_model.parameters(), lr=0.0001)
    optimizer1 = optim.RMSprop(context_model.parameters(), lr=0.0001)
    # stocknet 0.1, 16, 48, max1+1, 2, 2 0.8
    # pos_weight = torch.tensor([0.5]).to(device)
    # pos_weight = torch.tensor([0.6]).to(device))
    # ftse 0.1, 16, 32, max1+1, 2, 2 0.8
    # ni225 0.1, 16, 32, max1+1, 2, 1 0.85
    # kdd17 0.1, 16, 32, max1+1, 2, 2 0.8
    # print(layers,heads,dims,pos_weis,learning_rate,gl_wei)
    criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([pos_weis]).to(device))
    # criterion = nn.CrossEntropyLoss(weight = torch.tensor([0.47,0.53], dtype=torch.float32).to(device))
    best = 0
    loss_last = 0
    idx = 2000
    best_test_acc = 0 
    best_test_mcc = 0
    expanded_txt1 = expanded_txt1.permute(1,0,2,3)
    expanded_txt2 = expanded_txt2.permute(1,0,2,3)
    # print(expanded_txt1.shape,expanded_ratio1.shape,expanded_sum1.shape)
    expanded_ratio1 = expanded_ratio1.permute(1,0,2,3)
    expanded_sum1 = expanded_sum1.permute(1,0,2,3)

    expanded_ratio2 = expanded_ratio2.permute(1,0,2,3)
    expanded_sum2 = expanded_sum2.permute(1,0,2,3)

    # expanded_tensor1 = expanded_tensor1.permute(2,1,0,3)
    # print(expanded_tensor.shape,expanded_tensor1.shape)
    # print(expanded_txt2_2.shape,expanded_txt.shape,expanded_txt1.shape,"1650")
    # expanded_txt_2 = expanded_txt.permute(0,2,1)
    expanded_txt1_2 = expanded_txt1_2.permute(1,0,2)
    expanded_txt2_2 = expanded_txt2_2.permute(1,0,2)
    expanded_txt_sc = expanded_txt_sc.permute(1,0,2)
    expanded_txt1_sc = expanded_txt1_sc.permute(1,0,2)
    expanded_txt2_sc = expanded_txt2_sc.permute(1,0,2)
    # print(expanded_txt_sc.shape,expanded_sc_pad.shape,"1672")
    # expanded_sc_pad = expanded_sc_pad.permute(1,0,2,3)
    expanded_sc1_pad = expanded_sc1_pad.permute(1,0,2,3)
    expanded_sc2_pad = expanded_sc2_pad.permute(1,0,2,3)
    best_eval = 0
    pad_vector = torch.full((40,), 59041).to(device) 
    for epoch in range(num_epochs):
        print(epoch)
        correct2 = 0
        dtml.eval()
        context_model.train()
        correct = 0
        count = 0
        correct = 0
        emb_out = None
        for j in range(0,len(expanded_tensor),idx):
            # optimizer.zero_grad()
            data_train = []
            label_train = []
            dt_train = expanded_tensor[j:j+idx,:,:].to(device)
            dt_text = expanded_txt[j:j+idx,:,:,:].to(device)
            dt_ratio = expanded_ratio[j:j+idx,:,:,:].to(device)
            dt_sum = expanded_sum[j:j+idx,:,:,:].to(device)
            dt_text2 = expanded_txt_2[j:j+idx,:,:].to(device)
            dt_sc_pad = expanded_sc_pad[j:j+idx,:,:,:].to(device)
            # dt_text2 = expanded_txt2[j:j+idx,:,:]
            # scaler = torch.cuda.amp.GradScaler()
            # with torch.cuda.amp.autocast():
            pred, train_emb = dtml(dt_train)
            emb_out = train_emb
            # print(dt_train.shape,"1461",dt_text.shape,train_emb.shape)
            dt_label = expanded_label[j:j+idx,:].to(device)
            lb_mask = []
            for i in range(len(dt_label)):
                data_train.append(pred[i][(dt_label[i].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                label_train.append(dt_label[i][(dt_label[i].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
            data_train1 = torch.cat(data_train, dim=0)
            label_train1 = torch.cat(label_train, dim=0).squeeze(-1)
            predicted = (torch.sigmoid(data_train1) > 0.5).int()
            valid_indices = torch.where(label_train1 != 2)[0]
            filtered_data = data_train1[valid_indices].squeeze(1)
            filtered_labels = label_train1.float()[valid_indices]
            loss = criterion(filtered_data, filtered_labels)
            # print(filtered_labels.shape,predicted.shape)
            print((filtered_labels.to(device) == predicted.squeeze(1)).sum().item()/len(predicted),"1514")
            # loss.backward()
            # optimizer.step()

        ###
        # print("1491",dt_text.shape,train_emb.shape,dt_label.shape)
        dt_text = dt_text.permute(1,0,2,3)
        dt_ratio = dt_ratio.permute(1,0,2,3)
        dt_sum = dt_sum.permute(1,0,2,3)
        train_emb = train_emb.permute(1,0,2)
        dt_label = dt_label.permute(1,0,2)
        dt_text2 = dt_text2.permute(1,0,2)
        dt_sc_pad = dt_sc_pad.permute(1,0,2,3)

        # print("1494",dt_text.shape,train_emb.shape,dt_label.shape)
        corre = 0
        total = 0
        print(dt_text.shape,dt_text2.shape,expanded_txt_sc.shape,dt_sc_pad.shape,"1751")
        for k in range(len(dt_label)):
            optimizer1.zero_grad()
            # dt_text[k] = deduplicate_sequences(dt_text[k], pad_vector)
            out = context_model(train_emb[k].detach().unsqueeze(0), dt_text[k].unsqueeze(0), dt_ratio[k].unsqueeze(0),dt_sum[k].unsqueeze(0),dt_text2[k].unsqueeze(0),expanded_txt_sc[k].unsqueeze(0).to(device),dt_sc_pad[k].unsqueeze(0).to(device))
            # print("1502",train_emb[k].shape,dt_text[k].shape,dt_text[k])
            label = dt_label[k][(dt_label[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)]
            # print(dt_label.shape,out.sh)
            loss = criterion(out[0][(dt_label[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)], label)
            # _, predicted = torch.max(out.squeeze(0), 0)
            predicted = (torch.sigmoid(out[0][(dt_label[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)]) > 0.5).int()
            loss.backward()
            optimizer1.step()
            # print(label.shape, predicted.shape)
            corre += (label.squeeze(1).to(device) == predicted.squeeze(1)).sum().item()
            total += label.shape[0]
        print("pre",corre/total,corre,total)
        # for k in range(0,len(dt_label),5):
        #     optimizer1.zero_grad()
        #     out = context_model(train_emb[k:k+5].detach(), dt_text[k:k+5])
        #     # print("1502",train_emb[k].shape,dt_text[k].shape)
        #     dt_label1 = dt_label[k:k+5,:]
        #     for i in range(len(dt_label1)):
        #         data_train.append(out[i][(dt_label1[i].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
        #         label_train.append(dt_label1[i][(dt_label1[i].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
        #     # label = dt_label[k][(dt_label[k:k+5].to(device)!=torch.tensor(2).to(device)).squeeze(1)]
        #     data_train1 = torch.cat(data_train, dim=0)
        #     label_train1 = torch.cat(label_train, dim=0)
        #     loss = criterion(data_train1, label_train1)
        #     # _, predicted = torch.max(out.squeeze(0), 0)
        #     # predicted = (torch.sigmoid(out[0][(dt_label[k:k+5].to(device)!=torch.tensor(2).to(device)).squeeze(1)]) > 0.5).int()
        #     loss.backward()
        #     optimizer1.step()
        #     # print(label.shape, predicted.shape)
        #     # corre += (label.squeeze(1).to(device) == predicted.squeeze(1)).sum().item()
        #     # total += label.shape[0]
        # # print("pre",corre/total,corre,total)


        # train_emb1 = train_emb.reshape(-1, train_emb.size(2))  # [251 * 24, 128]
        # indices = torch.arange(train_emb1.size(1)).repeat(train_emb1.size(0))  # [251 * 24]
        dtml.eval()
        context_model.eval()
        data_eval = []
        label_eval = []
        pred_all = []
        label_all = []
        # expanded_tensor1 = expanded_tensor1.permute()
        # print(expanded_tensor2.shape,expanded_label2.shape)
        # print("?",1480,expanded_tensor1.shape,expanded_label1.shape)
        with torch.no_grad():
            pred, eval_emb = dtml(expanded_tensor1.to(device))
            for i in range(len(expanded_label1)):
                data_eval.append(pred[i][(expanded_label1[i].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                label_eval.append(expanded_label1[i].to(device)[(expanded_label1[i].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
            
            data_eval1 = torch.cat(data_eval, dim=0).to(device)
            label_eval1 = torch.cat(label_eval, dim=0).squeeze(-1).to(device)
            loss1 = criterion(data_eval1.squeeze(), label_eval1.float())
            # print("1581",data_eval1,data_eval1.shape)
            eval_emb = eval_emb.permute(1,0,2)
            
            expanded_label22 = expanded_label2.permute(1,0,2)
            expanded_label11 = expanded_label1.permute(1,0,2)

            corre = 0
            total = 0
            pred = []
            # print("input1590",eval_emb.shape,expanded_txt1.shape)
            for k in range(len(expanded_label11)):
                # optimizer1.zero_grad()
                # expanded_txt1[k] = deduplicate_sequences(expanded_txt1[k].to(device), pad_vector)
                out = context_model(eval_emb[k].detach().unsqueeze(0), expanded_txt1[k].unsqueeze(0), expanded_ratio1[k].unsqueeze(0),expanded_sum1[k].unsqueeze(0),expanded_txt1_2[k].unsqueeze(0),expanded_txt1_sc[k].unsqueeze(0).to(device),expanded_sc1_pad[k].unsqueeze(0).to(device))
                
                label = expanded_label11[k].to(device)
                # loss = criterion(out[0], label)
                # _, predicted = torch.max(out.squeeze(0), 1)
                predicted = (torch.sigmoid(out[0][(expanded_label11[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)]) > 0.5).int()
                # loss.backward()
                # optimizer1.step()
                # print(label.shape,expanded_label11.shape)
                corre += (label[(expanded_label11[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)].squeeze(1).to(device) == predicted.squeeze(1)).sum().item()
                total += label[(expanded_label11[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)].shape[0]
                pred.append(out[0][(expanded_label11[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
            # print("1601",pred)
            print("pre_val",corre/total,corre,total)
            pre_val = corre/total
            eval_emb1 = eval_emb.reshape(-1, eval_emb.size(2))  # [251 * 24, 128]
            indices1 = torch.arange(eval_emb1.size(1)).repeat(eval_emb1.size(0))  # [251 * 24]
            predicted1 = (torch.sigmoid(data_eval1) > 0.5).int()
            # predicted1 = torch.argmax(data_eval1,dim=1)
            correct3 = (label_eval1.to(device) == predicted1.to(device).squeeze(1)).sum().item()
            data_test = []
            label_test = []
            pred, test_emb = dtml(expanded_tensor2.to(device))
            
           
            
            for i in range(len(expanded_label2)):
                data_test.append(pred[i][(expanded_label2[i]!=torch.tensor(2)).squeeze(1).to(device)])
                label_test.append(expanded_label2[i].to(device)[(expanded_label2[i]!=torch.tensor(2)).squeeze(1).to(device)])
            data_test2 = torch.cat(data_test, dim=0)
            label_test2 = torch.cat(label_test, dim=0).squeeze(-1)
            predicted = (torch.sigmoid(data_test2) > 0.5).int()
            # predicted = torch.argmax(data_test2,dim=1)
            # print("1622",data_test2,data_test2.shape)
            correct2 = (label_test2.to(device) == predicted.to(device).squeeze(1)).sum().item()
            test_emb1 = test_emb.reshape(-1, test_emb.size(2))  # [251 * 24, 128]

            test_emb = test_emb.permute(1,0,2)
            


            corre = 0
            total = 0
            pred = []
            print("input1617",test_emb.shape,expanded_txt2.shape)
            for k in range(len(expanded_label22)):
                # optimizer1.zero_grad()
                # expanded_txt2[k] = deduplicate_sequences(expanded_txt2[k].to(device), pad_vector)
                out = context_model(test_emb[k].detach().unsqueeze(0), expanded_txt2[k].unsqueeze(0),expanded_ratio2[k].unsqueeze(0),expanded_sum2[k].unsqueeze(0),expanded_txt2_2[k].unsqueeze(0),expanded_txt2_sc[k].unsqueeze(0).to(device),expanded_sc2_pad[k].unsqueeze(0).to(device))
                label = expanded_label22[k].to(device)
                # loss = criterion(out[0], label)
                # _, predicted12 = torch.max(out.squeeze(0), 1)
                predicted12 = (torch.sigmoid(out[0][(expanded_label22[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)]) > 0.5).int()
                # loss.backward()
                # optimizer1.step()
                corre += (label[(expanded_label22[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)].squeeze(1).to(device) == predicted12.squeeze(1)).sum().item()
                total += label[(expanded_label22[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)].shape[0]
                label_all.extend(label[(expanded_label22[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)].squeeze(1).tolist())
                pred_all.extend(predicted12.squeeze(1).tolist())
                pred.append(out[0][(expanded_label22[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
            # print("1647",pred)
            print(corre/total,corre,total)
            if corre/total > best_eval  and epoch >2 and abs(corre/total-pre_val)<0.2:
                best_eval = corre/total
                print("test_val",pre_val, corre/total,corre,total,calculate_mcc(pred_all,label_all))
                # torch.save(context_model,"context_model427.pth")


            if correct3/len(predicted1)>best:
                best = correct3/len(predicted1)
                print(predicted1.shape,predicted.shape)
                # torch.save(dtml,"/home/zhaokx/method2/ckpt/"+dir1+"_".join([str(b) for b in [layers,heads,dims,pos_weis,learning_rate,gl_wei,mamba_ly]])+"_model_time_t.pth")
                print("***",correct2/len(predicted),calculate_mcc(label_test2.detach().cpu().numpy(),predicted.detach().cpu().numpy()),correct3/len(predicted1),calculate_mcc(label_eval1.detach().cpu().numpy(),predicted1.detach().cpu().numpy()))
                best_test_acc = correct2/len(predicted)
            # loss_last = loss1
if __name__ == "__main__":
    # dataset_train = StockDataset("/home/zhaokx/method2/dataset/acl18/tweet/preprocessed","/home/zhaokx/method2/dataset/acl18/price/preprocessed",mode='train')
    # dataset_val = StockDataset("/home/zhaokx/method2/dataset/acl18/tweet/preprocessed","/home/zhaokx/method2/dataset/acl18/price/preprocessed",mode='test')
    # dataset_train = StockDataset("/home/zhaokx/CMIN-Dataset-main/CMIN-US/news/preprocessed","/home/zhaokx/CMIN-Dataset-main/CMIN-US/price/processed",mode='train')
    # dataset_val = StockDataset("/home/zhaokx/CMIN-Dataset-main/CMIN-US/news/preprocessed","/home/zhaokx/CMIN-Dataset-main/CMIN-US/price/processed",mode='test')
    # dataset_train = StockDataset("/home/zhaokx/method2/sn2-main/tweet/preprocessed","/home/zhaokx/method2/data_sm2/price/preprocessed",mode='train')
    # dataset_val = StockDataset("/home/zhaokx/method2/sn2-main/tweet/preprocessed","/home/zhaokx/method2/data_sm2/price/preprocessed",mode='test')
    batch_size = 1
    num_epochs = 200
    # print(dataset_val.word_table_init.shape)
    # train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    # val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    stock_feature_dim = 11
    hidden_dim = 64
    window_size = 5
    class_weights = torch.tensor([1.0,1], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # 
    # criterion1 = nn.BCEWithLogitsLoss()
    best_eval = 0
    dataset = ['kdd17','ftse100','ni225','stocknet']
    from imblearn.over_sampling import SMOTE
    dir1 = "ni225"
    # model = TimeTextModel(stock_feature_dim, hidden_dim).to(device)
    # dtml = torch.load("/home/zhaokx/method2/pth_files/"+str(dir1)+"_model_time.pth",weights_only=False)
    # print("dir:",dir1)
    loaded = np.load('./../datasets/'+dir1+'_data.npz')
    train_data = torch.tensor(loaded['tra_pv'])
    train_label = torch.tensor(loaded['tra_gt'])
    train_wd =  torch.tensor(loaded['tra_wd'])
    val_data = torch.tensor(loaded['val_pv'])
    val_label = torch.tensor(loaded['val_gt'])
    val_wd =  torch.tensor(loaded['val_wd'])
    test_data = torch.tensor(loaded['tes_pv'])
    test_label = torch.tensor(loaded['tes_gt'])
    test_wd = torch.tensor(loaded['tes_wd'])
    tra_dt = torch.tensor(loaded['tra_dt'])
    val_dt = torch.tensor(loaded['val_dt'])
    test_dt = torch.tensor(loaded['tes_dt'])
    tra_tn = torch.tensor(loaded['tra_tn'])
    val_tn = torch.tensor(loaded['val_tn'])
    tes_tn = torch.tensor(loaded['tes_tn'])
    tra_lb = torch.tensor(loaded['tra_lb'])
    val_lb = torch.tensor(loaded['val_lb'])
    tes_lb = torch.tensor(loaded['tes_lb'])
    # tra_txt = torch.tensor(loaded['tra_txt'])
    # val_txt = torch.tensor(loaded['val_txt'])
    # tes_txt = torch.tensor(loaded['tes_txt'])
    # tra_txt2 = torch.tensor(loaded['tra_txt2'])
    # val_txt2 = torch.tensor(loaded['val_txt2'])
    # tes_txt2 = torch.tensor(loaded['tes_txt2'])
    # tra_ratio = torch.tensor(loaded['tra_ratio'])
    # val_ratio = torch.tensor(loaded['val_ratio'])
    # tes_ratio = torch.tensor(loaded['tes_ratio'])
    # tra_txt_sc = torch.tensor(loaded['tra_txt_sc'])
    # val_txt_sc = torch.tensor(loaded['val_txt_sc'])
    # tes_txt_sc = torch.tensor(loaded['tes_txt_sc'])
    # tra_sum = torch.tensor(loaded['tra_sum'])
    # val_sum = torch.tensor(loaded['val_sum'])
    # tes_sum = torch.tensor(loaded['tes_sum'])

    # tra_scores_all = torch.tensor(loaded['tra_scores_pad'])
    # val_scores_all = torch.tensor(loaded['val_scores_pad'])
    # tes_scores_all = torch.tensor(loaded['tes_scores_pad'])

    train_data = np.concatenate((train_data,train_wd),axis=-1)
    test_data = np.concatenate((test_data,test_wd),axis=-1)
    val_data = np.concatenate((val_data,val_wd),axis=-1)
    batch_size, seq_len, feature_dim = train_data.shape
    best = 0 
    acc = 0
    idx = 1000
    dis = 1000
    train_data = torch.tensor(train_data)
    grouped_data = defaultdict(list)
    grouped_label = defaultdict(list)
    grouped_sid = defaultdict(list)
    grouped_lb = defaultdict(list)
    # grouped_txt = defaultdict(list)
    # grouped_ratio = defaultdict(list)
    # grouped_sum = defaultdict(list)
    # grouped_txt_2 = defaultdict(list)
    # grouped_txt_sc = defaultdict(list)
    # grouped_tra_sc = defaultdict(list)
    max1 = 0
    for i in range(train_data.shape[0]):
        date = tra_dt[i].item()
        grouped_data[date].append(train_data[i])
        grouped_label[date].append(train_label[i])
        grouped_sid[date].append(tra_tn[i][0].long())
        grouped_lb[date].append(tra_lb[i][0].long())
        # grouped_txt[date].append(tra_txt[i].long())
        # grouped_ratio[date].append(tra_ratio[i])
        # grouped_sum[date].append(tra_sum[i])
        # grouped_txt_2[date].append(tra_txt2[i])
        # grouped_txt_sc[date].append(tra_txt_sc[i])
        # grouped_tra_sc[date].append(tra_scores_all[i])
        if tra_tn[i][0]>max1:
            max1 = tra_tn[i][0].int()
    print(len(grouped_data))
    grouped_data = {k: torch.stack(v) for k, v in grouped_data.items()}
    grouped_label = {k: torch.stack(v) for k, v in grouped_label.items()}
    grouped_sid = {k: torch.stack(v) for k, v in grouped_sid.items()}
    grouped_lb = {k: torch.stack(v) for k, v in grouped_lb.items()}
    # grouped_txt = {k: torch.stack(v) for k, v in grouped_txt.items()}
    # grouped_ratio = {k: torch.stack(v) for k, v in grouped_ratio.items()}
    # grouped_sum = {k: torch.stack(v) for k, v in grouped_sum.items()}
    # grouped_txt_2 = {k: torch.stack(v) for k, v in grouped_txt_2.items()}
    # grouped_txt_sc = {k: torch.stack(v) for k, v in grouped_txt_sc.items()}
    # grouped_tra_sc = {k: torch.stack(v) for k, v in grouped_tra_sc.items()}
    # 初始化张量（用 padding 统一维度）
    expanded_tensor = torch.zeros(len(grouped_data), max1+1, 5, 16)
    expanded_label = torch.zeros(len(grouped_data), max1+1,1 )
    # expanded_txt = torch.zeros(len(grouped_data), max1+1,20, 40).long()
    # expanded_ratio = torch.zeros(len(grouped_data), max1+1,20, 40)
    # expanded_sum = torch.zeros(len(grouped_data), max1+1,20, 40)
    # expanded_txt_2 = torch.zeros(len(grouped_data), max1+1, 500)
    # expanded_txt_sc = torch.zeros(len(grouped_data),max1+1, 10)
    # expanded_sc_pad = torch.zeros(len(grouped_data),max1+1, 20, 10)
    # 将数据填充到 expanded_tensor
    numbers0 = {}
    numbers01 = {}
    for i, (date, samples) in enumerate(grouped_data.items()):
        expanded_tensor[i, grouped_sid[date]] = samples.float()
        expanded_label[i,grouped_sid[date]] = grouped_label[date].float()
        # expanded_txt[i, grouped_sid[date]] = grouped_txt[date]
        # expanded_ratio[i, grouped_sid[date]] = grouped_ratio[date].float()
        # expanded_sum[i,grouped_sid[date]] = grouped_sum[date].float()
        # expanded_txt_2[i,grouped_sid[date]] = grouped_txt_2[date].float()
        # # print(grouped_txt_sc[date].float(),"2011",grouped_txt_sc[date].float().shape,grouped_tra_sc[date].shape)
        # expanded_txt_sc[i, grouped_sid[date]] = grouped_txt_sc[date].float()
        # expanded_sc_pad[i, grouped_sid[date]] = grouped_tra_sc[date].float()
        numbers0[i] = grouped_sid[date]
        numbers01[i] = (grouped_lb[date] == torch.tensor(1))
        numbers0[i] = numbers0[i][numbers01[i]]
        
    expanded_tensor = expanded_tensor.permute(0,2,1,3)
    # print("2082",expanded_tensor.shape,expanded_label.shape,expanded_txt.shape,expanded_txt_2.shape,expanded_txt_sc.shape)
    val_data = torch.tensor(val_data)
    grouped_data1 = defaultdict(list)
    grouped_label1 = defaultdict(list)
    grouped_sid1 = defaultdict(list)
    grouped_lb1 = defaultdict(list)
    # grouped_txt1 = defaultdict(list)
    # grouped_ratio1 = defaultdict(list)
    # grouped_sum1 = defaultdict(list)
    # grouped_txt1_2 = defaultdict(list)
    # grouped_txt1_sc = defaultdict(list)
    # grouped_tra1_sc = defaultdict(list)
    
    for i in range(val_data.shape[0]):
        date = val_dt[i].item()
        grouped_data1[date].append(val_data[i])
        grouped_label1[date].append(val_label[i])
        grouped_sid1[date].append(val_tn[i][0].long())
        grouped_lb1[date].append(val_lb[i][0].long())
        # grouped_txt1[date].append(val_txt[i].long())
        # grouped_ratio1[date].append(val_ratio[i])
        # grouped_sum1[date].append(val_sum[i])
        # grouped_txt1_2[date].append(val_txt2[i])
        # grouped_txt1_sc[date].append(val_txt_sc[i])
        # grouped_tra1_sc[date].append(val_scores_all[i])

    grouped_data1 = {k: torch.stack(v) for k, v in grouped_data1.items()}
    grouped_label1 = {k: torch.stack(v) for k, v in grouped_label1.items()}
    grouped_sid1 = {k: torch.stack(v) for k, v in grouped_sid1.items()}
    grouped_lb1 = {k: torch.stack(v) for k, v in grouped_lb1.items()}
    # grouped_txt1 = {k: torch.stack(v) for k, v in grouped_txt1.items()}
    # grouped_ratio1 = {k: torch.stack(v) for k, v in grouped_ratio1.items()}
    # grouped_sum1 = {k: torch.stack(v) for k, v in grouped_sum1.items()}
    # grouped_txt1_2 = {k: torch.stack(v) for k, v in grouped_txt1_2.items()}
    # grouped_txt1_sc = {k: torch.stack(v) for k, v in grouped_txt1_sc.items()}
    # grouped_tra1_sc = {k: torch.stack(v) for k, v in grouped_tra1_sc.items()}
    # max_samples_per_day = max(len(samples) for samples in grouped_data1.values())

    # 初始化张量（用 padding 统一维度）
    expanded_tensor1 = torch.zeros(len(grouped_data1), max1+1, 5, 16)
    expanded_label1 = torch.zeros(len(grouped_data1), max1+1,1 )
    # expanded_txt1 = torch.zeros(len(grouped_data1), max1+1,20, 40).long()
    # expanded_ratio1 = torch.zeros(len(grouped_data1), max1+1,20, 40)
    # expanded_sum1 = torch.zeros(len(grouped_data1), max1+1,20, 40)
    # expanded_txt1_2 = torch.zeros(len(grouped_data1), max1+1, 500)
    # expanded_txt1_sc = torch.zeros(len(grouped_data1),max1+1, 10)
    # expanded_sc1_pad = torch.zeros(len(grouped_data1),max1+1, 20, 10)
    # 将数据填充到 expanded_tensor
    numbers1 = {}
    numbers11 = {}
    
    for i, (date, samples) in enumerate(grouped_data1.items()):
        expanded_tensor1[i, grouped_sid1[date]] = samples.float()
        expanded_label1[i, grouped_sid1[date]] = grouped_label1[date].float()
        # expanded_txt1[i, grouped_sid1[date]] = grouped_txt1[date]
        # expanded_ratio1[i, grouped_sid1[date]] = grouped_ratio1[date].float()
        # expanded_sum1[i,grouped_sid1[date]] = grouped_sum1[date].float()
        # expanded_txt1_2[i, grouped_sid1[date]] = grouped_txt1_2[date].float()
        # expanded_txt1_sc[i, grouped_sid1[date]] = grouped_txt1_sc[date].float()
        # expanded_sc1_pad[i, grouped_sid1[date]] = grouped_tra1_sc[date].float()
        numbers1[i] = grouped_sid1[date]
        numbers11[i] = grouped_lb1[date]
        numbers11[i] = (grouped_lb1[date] == torch.tensor(1))
        numbers1[i] = numbers1[i][numbers11[i]]
    expanded_tensor1 = expanded_tensor1.permute(0,2,1,3)

    test_data = torch.tensor(test_data)
    grouped_data2 = defaultdict(list)
    grouped_label2 = defaultdict(list)
    grouped_sid2 = defaultdict(list)
    grouped_lb2 = defaultdict(list)
    # grouped_txt2 = defaultdict(list)
    # grouped_ratio2 = defaultdict(list)
    # grouped_sum2 = defaultdict(list)
    # grouped_txt2_2 = defaultdict(list)
    # grouped_txt2_sc = defaultdict(list)
    # grouped_tra2_sc = defaultdict(list)

    for i in range(test_data.shape[0]):
        date = test_dt[i].item()
        grouped_data2[date].append(test_data[i])
        grouped_label2[date].append(test_label[i])
        grouped_sid2[date].append(tes_tn[i][0].long())
        grouped_lb2[date].append(tes_lb[i][0].long())
        # grouped_txt2[date].append(tes_txt[i].long())
        # grouped_ratio2[date].append(tes_ratio[i])
        # grouped_sum2[date].append(tes_sum[i])
        # grouped_txt2_2[date].append(tes_txt2[i])
        # grouped_txt2_sc[date].append(tes_txt_sc[i])
        # grouped_tra2_sc[date].append(tes_scores_all[i])
    
    grouped_data2 = {k: torch.stack(v) for k, v in grouped_data2.items()}
    grouped_label2 = {k: torch.stack(v) for k, v in grouped_label2.items()}
    grouped_sid2 = {k: torch.stack(v) for k, v in grouped_sid2.items()}
    grouped_lb2 = {k: torch.stack(v) for k, v in grouped_lb2.items()}
    # grouped_txt2 = {k: torch.stack(v) for k, v in grouped_txt2.items()}
    # grouped_ratio2 = {k: torch.stack(v) for k, v in grouped_ratio2.items()}
    # grouped_sum2 = {k: torch.stack(v) for k, v in grouped_sum2.items()}
    # grouped_txt2_2 = {k: torch.stack(v) for k, v in grouped_txt2_2.items()}
    # grouped_txt2_sc = {k: torch.stack(v) for k, v in grouped_txt2_sc.items()}
    # grouped_tra2_sc = {k: torch.stack(v) for k, v in grouped_tra2_sc.items()}
    # 初始化张量（用 padding 统一维度）
    expanded_tensor2 = torch.zeros(len(grouped_data2), max1+1, 5, 16)
    expanded_label2 = torch.zeros(len(grouped_data2), max1+1, 1)
    # expanded_txt2 = torch.zeros(len(grouped_data2), max1+1, 20, 40).long()
    # expanded_ratio2 = torch.zeros(len(grouped_data2), max1+1,20, 40)
    # expanded_sum2 = torch.zeros(len(grouped_data2), max1+1,20, 40)
    # expanded_txt2_2 = torch.zeros(len(grouped_data2), max1+1, 500)
    # expanded_txt2_sc = torch.zeros(len(grouped_data2), max1+1, 10)
    # expanded_sc2_pad = torch.zeros(len(grouped_data2),max1+1, 20, 10)
    # 将数据填充到 expanded_tensor
    # print(expanded_tensor2.shape,"1487")
    numbers2 = {}
    numbers21 = {}
    for i, (date, samples) in enumerate(grouped_data2.items()):
        expanded_tensor2[i, grouped_sid2[date]] = samples.float()
        expanded_label2[i, grouped_sid2[date]] = grouped_label2[date].float()
        # expanded_txt2[i, grouped_sid2[date]] = grouped_txt2[date]
        # expanded_ratio2[i, grouped_sid2[date]] = grouped_ratio2[date].float()
        # expanded_sum2[i, grouped_sid2[date]] = grouped_sum2[date].float()
        # expanded_txt2_2[i, grouped_sid2[date]] = grouped_txt2_2[date].float()
        # expanded_txt2_sc[i, grouped_sid2[date]] = grouped_txt2_sc[date].float()
        # expanded_sc2_pad[i, grouped_sid2[date]] = grouped_tra2_sc[date].float()
        numbers2[i] = grouped_sid2[date]
        numbers21[i] = grouped_lb2[date]
        numbers21[i] = (grouped_lb2[date] == torch.tensor(1))
        numbers2[i] = numbers2[i][numbers21[i]]
        # print(len(numbers21[i]))
    expanded_tensor2 = expanded_tensor2.permute(0,2,1,3)
    # test(expanded_label,expanded_label1,expanded_label2,expanded_txt1, expanded_txt2,expanded_tensor1,expanded_tensor2, expanded_ratio1,expanded_ratio2,expanded_sum1,expanded_sum2,expanded_txt_2,expanded_txt1_2,expanded_txt2_2, expanded_txt_sc, expanded_txt1_sc, expanded_txt2_sc, expanded_sc_pad,expanded_sc1_pad,expanded_sc2_pad)
    sampler = optuna.samplers.TPESampler(seed=2)
    study = optuna.create_study(direction="maximize", sampler=sampler)  # 最大化目标值
    study.optimize(objective, n_trials=50)  # 运行 50 次试验