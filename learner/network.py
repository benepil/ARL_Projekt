import torch
import torch.nn as nn


class SimpleNN(nn.Module):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 trainable_parameter: (int,),
                 activation: nn.Module,
                 ):

        super(SimpleNN, self).__init__()

        layers = []
        previous_layer_input_size: int = input_size
        for params in trainable_parameter:
            layers.append(nn.Linear(previous_layer_input_size, params))
            layers.append(nn.Dropout(p=0.2))
            layers.append(activation())
            previous_layer_input_size = params
        layers.append(nn.Linear(previous_layer_input_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        output = self.network(x)
        return output
