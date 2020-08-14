import numpy as np
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class MPN(nn.Module):
    # simple three layer neural network that receives the previous state of the character x
    # outputs the data in the format of y as follows:
    #
    #       Θ(x;α) = W2 ELU( W1 ELU( W0 x + b0) + b1) + b2,

    def __init__(self, state_dim, action_dim, hidden_layer=512, keep_prob=0.7, init_weights=False):
        super(MPN, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(state_dim, hidden_layer),
            nn.ELU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer),
            nn.ELU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_layer, action_dim),
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

    def compute_weights(self, expert_weights, blending_coefficients):
        # we're assuming that the given expert weights also have a 3 layer NN
        alpha = np.zeros(3)

        # K is a meta-parameter that can be adjusted according to the complexity and size of the training data
        K = len(expert_weights)

        # neural network weights α is computed by blending K expert weights β = {α1, ..., αK }
        # with the blending coefficients ω = {ω1, ..., ωK } computed by gating network
        for i in range(K):
            # compute NN weights by α = 􏰛∑ ω_i * α_i
            alpha += blending_coefficients[i] * expert_weights[i]

        # update NN weights
        self.l1[0].weight, self.l2[0].weight, self.l3[0].weight = alpha

