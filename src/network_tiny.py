from typing import List
from copy import deepcopy

import torch
import torch.nn as nn

from catalyst import utils


def _get_linear_net(
    in_features: int,
    history_len: int = 1,
    features: List = None,
    use_bias: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    activation: str = "ReLU"
) -> nn.Module:

    features = features or [16, 32, 16]
    activation_fn = torch.nn.__dict__[activation]

    def _get_block(**linear_params):
        layers = [nn.Linear(**linear_params)]
        if use_normalization:
            layers.append(nn.LayerNorm(linear_params["out_features"]))
        if use_dropout:
            layers.append(nn.Dropout(p=0.1))
        layers.append(activation_fn(inplace=True))
        return layers

    features.insert(0, history_len * in_features)
    params = []
    for i, (in_features, out_features) in enumerate(utils.pairwise(features)):
        params.append(
            {
                "in_features": in_features,
                "out_features": out_features,
                "bias": use_bias,
            }
        )

    layers = []
    for block_params in params:
        layers.extend(_get_block(**block_params))

    net = nn.Sequential(*layers)
    net.apply(utils.create_optimal_inner_init(activation_fn))

    return net


class StateNet(nn.Module):
    def __init__(
        self,
        main_net: nn.Module,
        features_net: nn.Module = None,
        vector_field_net: nn.Module = None,
    ):
        super().__init__()
        self.main_net = main_net
        self.features_net = features_net
        self.vector_field_net = vector_field_net

    def forward(self, state):
        batch_size, _, _ = state.shape
        x = state.contiguous().view(batch_size, -1)
        x = self.main_net(x)
        return x

    @classmethod
    def get_from_params(
        cls,
        features_net_params=None,
        vector_field_net_params=None,
        # aggregation_net_params=None,
        main_net_params=None,
    ) -> "StateNet":
        main_net_params = deepcopy(main_net_params)
        main_net = _get_linear_net(**main_net_params)

        net = cls(
            main_net=main_net
        )

        return net


class StateActionNet(nn.Module):
    def __init__(
        self,
        main_net: nn.Module,
        features_net: nn.Module = None,
        vector_field_net: nn.Module = None,
        action_net: nn.Module = None
    ):
        super().__init__()
        self.main_net = main_net
        self.features_net = features_net
        self.vector_field_net = vector_field_net
        self.action_net = action_net

    def forward(self, state, action):
        batch_size, _, _ = state.shape
        tiny = state.contiguous().view(batch_size, -1)

        action = self.action_net(action)
        x = torch.cat([tiny, action], dim=1)
        x = self.main_net(x)

        return x

    @classmethod
    def get_from_params(
        cls,
        features_net_params=None,
        vector_field_net_params=None,
        action_net_params=None,
        main_net_params=None,
    ) -> "StateActionNet":

        action_net_params = deepcopy(action_net_params)
        main_net_params = deepcopy(main_net_params)

        action_net = _get_linear_net(**action_net_params)
        action_net_out_features = action_net_params["features"][-1]

        main_net_params["in_features"] += action_net_out_features
        main_net = _get_linear_net(**main_net_params)

        net = cls(
            action_net=action_net,
            main_net=main_net
        )

        return net
