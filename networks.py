import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedNet(nn.Module): 
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared_net = nn.Linear(state_dim, 128)

        ## policy_head
        self.policy_head = nn.Linear(128, action_dim)

        ## value_head 
        self.value_head = nn.Linear(128, 1)



    def forward(self, state):
        x = F.relu(self.shared_net(state))
        logits = self.policy_head(x) ## this is for the policy_head
        value = self.value_head(x)  ## this is for the value_head
        return logits, value
