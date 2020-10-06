import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class Gating(nn.Module):
    #  gating network, whose operation is denoted by Ω(·), is a three layer neural network
    #  that computes the blending coefficients ω given the input data x:
    #
    #       Ω(xˆ;μ) = σ(W2′ ELU(W1′ ELU(W0′ xˆ+b0′)+b1′)+b2′),
    #
    # σ (·) is a softmax operator that normalizes the inputs such that they sum up to 1,
    # which is required for the further linear blending.

    def __init__(self, input_size, output_size, hidden_size=32, keep_prob=0.7, init_weights=True):
        super(Gating, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(p=keep_prob)

        if init_weights:
            self.apply(weights_init)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dropout(self.l1(x))
        x = self.dropout(self.l2(x))
        x = self.l3(x)

        return x
