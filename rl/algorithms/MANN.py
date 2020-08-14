# Implementation of Mode Adaptive Neural Network (MANN)
# Paper: http://homepages.inf.ed.ac.uk/tkomura/dog.pdf
import torch
import torch.nn as nn

from rl.networks.Gating import Gating
from rl.networks.ModelPrediction import MPN


class MANN:
    def __init__(self, state_dim, action_dim, motion_feat_dim, num_experts,
                 lr=1e-4, weight_decay=2.5e-3, epochs=150, Te=10, Tmult=2, batch_size=32, dropout=0.7):
        # K (number of experts) is a meta-parameter that can be adjusted
        # according to the complexity and size of the training data
        # K = 4 is enough to generate high-quality motions
        # K = 8 is able to produce even better results with sharper movements

        self.gating = Gating(motion_feat_dim, num_experts)
        self.model_prediction = MPN(state_dim, action_dim)

        # initialize loss function
        self.loss_fn = nn.MSELoss()

        # We use the stochastic gradient descent algorithm with the warm restart technique of AdamWR
        self.optimizer = torch.optim.SGD(self.gating.parameters(), lr=lr, weight_decay=weight_decay)

    def update_gating_network(self):
        pass

    def update_model_prediction_network(self):
        pass
