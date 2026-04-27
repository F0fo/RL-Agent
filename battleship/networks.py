"""CNN backbones with policy and value heads for shooter and placer."""
import torch
import torch.nn as nn


class ConvBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShooterNet(nn.Module):
    """Policy logits over actions plus a scalar value.

    With `num_action_channels=1`, output is H*W logits (one per cell): fire-only.
    With `num_action_channels=2`, output is 2*H*W logits: first H*W are fire
    targets, next H*W are sonar centers (matches BattleshipGame.decode_action)."""

    def __init__(self, board_size: int, in_channels: int = 4,
                 num_action_channels: int = 1, hidden: int = 64):
        super().__init__()
        self.board_size = board_size
        self.in_channels = in_channels
        self.num_action_channels = num_action_channels
        self.backbone = ConvBackbone(in_channels, hidden)
        self.policy_head = nn.Conv2d(hidden, num_action_channels, 1)
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * board_size * board_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        # flatten(1) on (B, C, H, W) gives row-major (channel, row, col), so:
        # fire (channel 0) occupies indices [0, H*W); sonar (channel 1) occupies [H*W, 2*H*W).
        logits = self.policy_head(h).flatten(1)
        value = self.value_head(h).squeeze(-1)
        return logits, value


class PlacerNet(nn.Module):
    """Policy logits over (r, c, orient) flattened as r*W*2 + c*2 + orient."""

    def __init__(self, board_size: int, in_channels: int = 2, hidden: int = 64):
        super().__init__()
        self.board_size = board_size
        self.backbone = ConvBackbone(in_channels, hidden)
        self.policy_head = nn.Conv2d(hidden, 2, 1)         # 2 orientations
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * board_size * board_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        # (B, 2, H, W) -> (B, H, W, 2) -> (B, H*W*2) so action index = r*W*2 + c*2 + o
        logits = self.policy_head(h).permute(0, 2, 3, 1).reshape(obs.shape[0], -1)
        value = self.value_head(h).squeeze(-1)
        return logits, value
