import torch
import torch.nn as nn
import torch.nn.functional as F


class _LSKAOneBranch(nn.Module):
    def __init__(self, dim: int, k_size: int):
        super().__init__()
        self.dim = dim
        k = int(k_size)
        assert k in (7, 11, 23, 35, 41, 53), f"Unsupported kernel size: {k}"

        if k in (7, 11):
            if k == 7:
                k0, ks, d = 3, 3, 2
                pad_s = 2
            else:
                k0, ks, d = 3, 5, 2
                pad_s = 4
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, k0), padding=(0, (k0 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(k0, 1), padding=((k0 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, ks), padding=(0, pad_s), groups=dim, dilation=d)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(ks, 1), padding=(pad_s, 0), groups=dim, dilation=d)
        else:
            k0 = 5
            ks_map = {23: 7, 35: 11, 41: 13, 53: 17}
            ks = ks_map[k]
            pad_s = {7: 9, 11: 15, 13: 18, 17: 24}[ks]
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, k0), padding=(0, (k0 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(k0, 1), padding=((k0 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, ks), padding=(0, pad_s), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(ks, 1), padding=(pad_s, 0), groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return attn


class MLSKAP_K7_35_53(nn.Module):
    def __init__(
            self,
            dim: int,
            k_sizes: list = [7, 35, 53]
    ):
        super().__init__()
        self.dim = dim
        self.k_sizes = k_sizes
        self.num_branches = len(k_sizes)

        self.branches = nn.ModuleList()
        for k_size in k_sizes:
            self.branches.append(_LSKAOneBranch(dim, k_size))

        self.weights = nn.Parameter(torch.ones(self.num_branches) / self.num_branches)

        self.conv_final = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        # self.bn = nn.BatchNorm2d(dim)  # 添加BN层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()

        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))

        attn = torch.zeros_like(branch_outputs[0])
        for i, output in enumerate(branch_outputs):
            attn += output * self.weights[i]

        attn = torch.sigmoid(attn)
        output = u * attn
        return output


if __name__ == "__main__":
    model = MLSKAP_K7_35_53(dim=64, k_sizes=[7, 35, 53])
    x = torch.randn(1, 64, 32, 32)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"使用的卷积核尺寸: {model.k_sizes}")
    print(f"分支数量: {model.num_branches}")
    print(f"各分支权重: {model.weights}")