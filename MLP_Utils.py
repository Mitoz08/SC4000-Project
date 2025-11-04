from torch import nn
import torch

class MLP_1(nn.Module):
    def __init__(self, features=1059, hidden_layers = [128, 256, 64]):
        super().__init__()

        layers = []
        input_dim = features

        for output in hidden_layers:
            layers.append(nn.Linear(input_dim, output))
            layers.append(nn.ReLU())
            input_dim = output

        layers.append(nn.Linear(input_dim, 1)) # Binary classification default or not
        self.seq = nn.Sequential(*layers)
    def forward(self, x, logits=False):
        one_hot, s_2_count, cat, num = x
        s_2_count = s_2_count.unsqueeze(1)
        x = torch.cat((s_2_count,  one_hot, cat, num), dim=1)
        x = self.seq(x)
        if logits:
            return x
        else:
            return torch.sigmoid(x)


class MLP_1(nn.Module):
    def __init__(self, features=1059, h_layers_OH =[128, 256, 64], h_layers_Cat = [128, 256, 64], h_layers_Num = [128, 256, 64]):
        super().__init__()

        def make_nn(input_dim, layer_sizes):
            layers = []
            for output in layer_sizes:
                layers.append(nn.Linear(input_dim, output))
                layers.append(nn.ReLU())
                input_dim = output
            return nn.Sequential(*layers), input_dim # Return the size of the final layer for combine_dim

        self.seq_OH, OH_dim = make_nn(features, h_layers_OH)
        self.seq_Cat, CAT_dim = make_nn(features, h_layers_Cat)
        self.seq_Num, NUM_dim = make_nn(features, h_layers_Num)

        combine_dim = OH_dim + CAT_dim + NUM_dim + 1 # +1 for the S_2_count

        self.final_layer = nn.Linear(combine_dim, 1)


    def forward(self, x, logits=False):
        one_hot, s_2_count, cat, num = x
        s_2_count = s_2_count.unsqueeze(1)
        x_one_hot = self.seq_OH(one_hot)
        x_cat = self.seq_Cat(cat)
        x_num = self.seq_Num(num)
        x = torch.cat((s_2_count, x_one_hot, x_cat, x_num), dim=1)
        if logits:
            return x
        else:
            return self.final_layer(x)