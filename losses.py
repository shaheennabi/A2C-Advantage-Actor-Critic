from config import VALUE_COEF, COEF_ENTROPY


def compute_loss(log_probs, advantage, values, entropies):
    """Compute total loss for A2C.

    Args:
        log_probs (Tensor): log probabilities for actions (N,)
        advantage (Tensor): advantage estimates (N,)
        values (Tensor): value estimates (N,)
        entropies (Tensor): entropies per step (N,)

    Returns:
        Tensor: scalar loss
    """
    policy_loss = -(log_probs * advantage).mean()
    value_loss = VALUE_COEF * advantage.pow(2).mean()
    entropy_loss = - (COEF_ENTROPY * entropies).mean()

    return policy_loss + value_loss + entropy_loss