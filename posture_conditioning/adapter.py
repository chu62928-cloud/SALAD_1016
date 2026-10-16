"""Grouped posture encoding and gated, motion-query cross attention."""
import torch
from torch import nn

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
GROUPS = [[3, 6, 9], [0, 1, 2], [1, 4, 7, 10], [2, 5, 8, 11]]


class PostureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        adjacency = torch.eye(12)
        for child, parent in enumerate(PARENTS):
            if parent >= 0:
                adjacency[child, parent] = adjacency[parent, child] = 1
        self.register_buffer('adjacency', adjacency / adjacency.sum(1, keepdim=True))
        self.input = nn.Linear(12, 128)
        self.joint_id = nn.Parameter(torch.randn(12, 128) * .02)
        self.messages = nn.ModuleList([nn.Linear(256, 128) for _ in range(2)])
        self.output = nn.Linear(128, 256)
        self.shoulder = nn.Linear(1, 256)
        self.group_id = nn.Parameter(torch.randn(6, 256) * .02)

    def forward(self, core, shoulder):
        h = torch.nn.functional.gelu(self.input(core) + self.joint_id)
        for layer in self.messages:
            neighbors = torch.einsum('ij,bjd->bid', self.adjacency, h)
            h = h + torch.nn.functional.gelu(layer(torch.cat([h, neighbors], -1)))
        groups = [self.output(h[:, indices].mean(1)) for indices in GROUPS]
        # These two tokens share one measurement; group identities distinguish sides.
        left = self.shoulder(shoulder)
        right = self.shoulder(-shoulder)
        return torch.stack([left, right] + groups, 1) + self.group_id


class PostureAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(256)
        self.query = nn.Linear(256, 128)
        self.key = nn.Linear(256, 128)
        self.value = nn.Linear(256, 128)
        self.output = nn.Linear(128, 256, bias=False)
        nn.init.zeros_(self.output.weight)
        self.gate = nn.Sequential(nn.Linear(384, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.constant_(self.gate[-1].bias, -2.)
        self.tokens = None
        self.scale = 1.
        self.capture = False
        self.diagnostics = None

    def forward(self, h):
        if self.tokens is None or self.scale == 0:
            return h
        b, t, j, d = h.shape
        q = self.query(self.norm(h)).reshape(b, t*j, 4, 32).transpose(1, 2)
        k = self.key(self.tokens).reshape(b, 6, 4, 32).transpose(1, 2)
        v = self.value(self.tokens).reshape(b, 6, 4, 32).transpose(1, 2)
        alpha = (q @ k.transpose(-1, -2) / (32**.5)).softmax(-1)
        u = (alpha @ v).transpose(1, 2).reshape(b, t, j, 128)
        gate = self.gate(torch.cat([h, u], -1)).sigmoid()
        residual = gate * self.output(u) * self.scale
        if self.capture:
            self.diagnostics = dict(alpha=alpha.detach().cpu(), gate=gate.detach().cpu(),
                                    residual_norm=residual.norm(dim=-1).detach().cpu())
        return h + residual


class PostureModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.encoder = PostureEncoder()
        transformer = backbone.transformer
        self.blocks = list(transformer.input_blocks) + [transformer.middle_block] + list(transformer.output_blocks)
        for block in self.blocks:
            block.posture_adapter = PostureAdapter()
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.encoder.train(mode)
        for block in self.blocks:
            block.posture_adapter.train(mode)
        return self

    def adapter_state(self):
        return {name: value for name, value in self.state_dict().items()
                if name.startswith('encoder.') or '.posture_adapter.' in name}

    def load_adapter(self, state):
        expected = self.adapter_state()
        if set(state) != set(expected):
            raise ValueError('Adapter checkpoint keys differ')
        self.load_state_dict(state, strict=False)

    def forward(self, z, time, text, core=None, shoulder=None, mask=None, scale=1., capture=False):
        tokens = self.encoder(core, shoulder) if core is not None and scale != 0 else None
        for block in self.blocks:
            adapter = block.posture_adapter
            adapter.tokens, adapter.scale, adapter.capture = tokens, scale, capture
        try:
            return self.backbone(z, time, text, len_mask=mask)[0]
        finally:
            for block in self.blocks:
                block.posture_adapter.tokens = None
