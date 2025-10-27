import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from mamba import Mamba

class MEFI(nn.Module):
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
        # self.inference_params = InferenceParams()
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
            # Compute short convolution
            if conv_state is not None:
                x_padded = F.pad(x, (self.d_conv - x.shape[-1], 0))  # 填充x的最后一个维度
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
        dtype = hidden_states.dtype
        # 不再有时间步限制，允许多个时间步
        # 原本代码会在这里assert hidden_states.shape[1] == 1, 被去掉
        xz = self.in_proj(hidden_states)  # (B T D)
        x, z = xz.chunk(2, dim=-1)  # (B T D)
        
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
        # self.inference_params = InferenceParams()
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

class Encoder(nn.Module):
    def __init__(self, mamba_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.norm = norm_layer
        # self.inference_params = InferenceParams()
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        mamba_layers = []
        for _ in range(num_layers):
            mamba_forward = MEFI(
                d_model=input_dim,
                d_state=16,
                d_conv=4,
                expand=2,
                layer_idx=0
            )
            mamba_backward = MEFI(
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
        # 输出层：使用全局平均池化（兼容不同长度）
        self.output_layer = nn.Linear(input_dim, 2)

    def forward(self, x):
        # print("439**",x.shape)
        x = self.encoder(x)
        # print("441**",x.shape)
        output = self.output_layer(x)  # [B, 1]
        # print("443**",output.shape)
        return output, x