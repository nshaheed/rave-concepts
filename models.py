import torch
import torch.nn as nn


class EmbedBlock(nn.Module):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        act=nn.GELU(),
        resid=True,
        use_bn=False,
        requires_grad=True,
        **kwargs,
    ) -> None:
        "generic little block for embedding stuff.  note residual-or-not doesn't seem to make a huge difference for a-a"
        super().__init__()
        self.in_dims, self.out_dims, self.act, self.resid = (
            in_dims,
            out_dims,
            act,
            resid,
        )
        self.lin = nn.Linear(in_dims, out_dims, **kwargs)
        self.bn = (
            nn.BatchNorm1d(out_dims) if use_bn else None
        )  # even though points in 2d, only one non-batch dim in data
        self.dropout = nn.Dropout(0.001)

        if requires_grad == False:
            self.lin.weight.requires_grad = False
            self.lin.bias.requires_grad = False

    def forward(self, xin):
        x = self.lin(xin)
        x = self.dropout(x) if self.dropout is not None else x
        if self.bn is not None:
            x = self.bn(x)  # vicreg paper uses BN before activation
        if self.act is not None:
            x = self.act(x)
        # if self.bn  is not None: x = self.bn(x)   # re. "BN before or after Activation? cf. https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md"
        return xin + x if (self.resid and self.in_dims == self.out_dims) else x


class Effect(nn.Module):
    """
    effect model (just using as gain for now
    """

    def __init__(
        self,
        in_dims=128,
        out_dims=128,
        hidden_dims_scale=6,
        num_inner_layers=6,
        act=nn.GELU(),
        use_bn=False,  # bn is bad for regression model
        resid=True,
        block=EmbedBlock,  # Linear layer with optional activation & optional BatchNorm
        trivial=False,  # ignore everything and make this an identity mapping
    ):
        super().__init__()
        self.resid, self.trivial = resid, trivial
        hidden_dims = hidden_dims_scale * in_dims
        # resid=False # turn it off for inner layers, just leave outer resid
        self.blocks = nn.Sequential(
            block(in_dims, hidden_dims, act=act, use_bn=use_bn, resid=resid),
            *[
                block(hidden_dims, hidden_dims, act=act, use_bn=use_bn, resid=resid)
                for _ in range(num_inner_layers)
            ],
            block(
                hidden_dims, out_dims, act=None, use_bn=use_bn, resid=resid, bias=False
            ),  # bias=False from VICReg paper
        )

    def forward(self, y):
        y = y.to(dtype=torch.float)

        if self.trivial:
            return y  # "trivial" no-op  flag for quick testing
        z = self.blocks(
            y
        )  # transpose is just so embeddings dim goes last for matrix mult
        return z + y if self.resid else z
