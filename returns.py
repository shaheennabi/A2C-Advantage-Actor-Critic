from config import GAMMA
import torch


def calculate_returns(rewards, next_value, gamma=GAMMA):
    """Compute discounted returns.

    Args:
        rewards (list[Tensor]): list of reward scalars (torch.Tensor)
        next_value (Tensor or float): bootstrap value for the step after the last reward
        gamma (float): discount factor

    Returns:
        Tensor: stacked returns with same length as rewards
    """
    returns = []
    R = next_value
    for reward in reversed(rewards):
        R = reward + gamma * R
        returns.append(R)

    returns.reverse()
    return torch.stack(returns)

