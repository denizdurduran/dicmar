from typing import List
from copy import deepcopy

import torch
import torch.nn as nn

from catalyst import utils

def get_network(params):
    params = deepcopy(params)
    if(params["type"] == "lstm"):
        return _get_lstm_net(**params)
    else:
        return _get_linear_net(**params)


# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, seq_len,
                    num_layers=2,activation_fn = nn.ReLU):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len # sequation dim
        self.num_layers = num_layers

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)


    def init_hidden(self, batch_size):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, input):
        try:
            if(len(input.shape) == 2 ):
                if(input.shape[0] == 1):
                    input = input.reshape(5,-1).unsqueeze(0) 
                else:
                    input = input.reshape(input.shape[0],self.seq_len,-1)
            batch_size, _, _ = input.shape
            (h0,c0) = self.init_hidden(batch_size)
            # Forward pass through LSTM layer
            # shape of lstm_out: [input_size, batch_size, hidden_dim]
            # shape of self.hidden: (a, b), where a and b both 
            # have shape (num_layers, batch_size, hidden_dim).
            lstm_out, self.hidden = self.lstm(input,(h0.detach(), c0.detach()))
            
            # Only take the output from the final timetep and passit through fc
            return self.fc(lstm_out[:,-1,:])
        except:
            import ipdb; ipdb.set_trace()

def _get_lstm_net(
    type: "lstm",
    in_features: int,
    hidden_dim: int,
    num_layers: int,
    history_len: int = 1,
    features: List = None,
    #use_bias: bool = False,             # Not needed
    #use_normalization: bool = False,    # Not needed
    #use_dropout: bool = False,
    activation: str = "ReLU"
) -> nn.Module:

    activation_fn = torch.nn.__dict__[activation]
    net = LSTM(in_features, hidden_dim, history_len, num_layers, activation_fn)
    return net



def _get_linear_net(
    type: "linear",
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
    ):
        super().__init__()
        self.main_net = main_net

    def forward(self, state):
        batch_size, _, _ = state.shape
        x = state.contiguous().view(batch_size, -1)
        x = self.main_net(x)
        return x

    @classmethod
    def get_from_params(
        cls,
        # aggregation_net_params=None,
        main_net_params=None,
    ) -> "StateNet":
        main_net_params = deepcopy(main_net_params)
        main_net = _get_lstm_net(**main_net_params)

        net = cls(
            main_net=main_net
        )

        return net


class StateActionNet(nn.Module):
    def __init__(
        self,
        state_net: nn.Module = None,
        action_net: nn.Module = None,
        main_net: nn.Module = None
    ):
        super().__init__()
        self.main_net = main_net
        self.action_net = action_net
        self.state_net = state_net

    def forward(self, state, action):
        state = self.state_net(state)
        action = self.action_net(action)
        x = torch.cat([state, action], dim=1)
        x = self.main_net(x)
        return x

    @classmethod
    def get_from_params(
        cls,
        state_net_params=None,
        action_net_params=None,
        main_net_params=None,
    ) -> "StateActionNet":

        state_net = get_network(state_net_params)
        main_net = get_network(main_net_params)
        action_net = get_network(action_net_params)

        net = cls(
            state_net=state_net,
            action_net=action_net,
            main_net=main_net
        )

        return net
