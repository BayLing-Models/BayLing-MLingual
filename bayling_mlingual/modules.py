import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class MLP(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out, hidden_act, rms_norm_eps):
        super(MLP, self).__init__()
        self.input_layernorm = LlamaRMSNorm(dim_in, eps=rms_norm_eps)
        self.linear1 = nn.Linear(dim_in, dim_mid)
        self.act_fn = ACT2FN[hidden_act]
        self.linear2 = nn.Linear(dim_mid, dim_out)

    def forward(self, x):
        x = self.input_layernorm(x)
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


class Mapping(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out, layer_num, hidden_act="silu", rms_norm_eps=1e-6):
        super(Mapping, self).__init__()
        self.end_boundary = nn.Parameter(
            torch.zeros(1, 1, dim_out), requires_grad=True
        )

        assert layer_num >= 1, "layer_num should be >= 1"
        layers = []
        if layer_num == 1:
            layers = [MLP(dim_in, dim_mid, dim_out, hidden_act, rms_norm_eps)]
        else:
            layers.append(MLP(dim_in, dim_mid, dim_mid, hidden_act, rms_norm_eps))
            for _ in range(layer_num - 2):
                layers.append(MLP(dim_mid, dim_mid, dim_mid, hidden_act, rms_norm_eps))
            layers.append(MLP(dim_mid, dim_mid, dim_out, hidden_act, rms_norm_eps))

        self.layers = nn.Sequential(*layers)

    def forward(self, hidden_states):
        return self.layers(hidden_states)

    def get_embed(self):
        return self.end_boundary
