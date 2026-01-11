import gymnasium as gym
import torch.optim as optim 
from networks import SharedNet
from config import N_STEPS
import torch
from torch.distributions import Categorical
from returns import calculate_returns
from losses import compute_loss

def training_loop(episodes):

    ## this is the training file..

    ## first we will setup the enviornment 
    env = gym.make("CartPole-v1")

    ## this is the state_dim and the action_dim
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = SharedNet(state_dim=state_dim, action_dim=action_dim)

    ## this is the optimizer 
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Collect per-episode metrics for optional return
    metrics = []

    for episode in range(episodes): 
        ## first initialize the must have things, before starting the training 
        ## here we will append the things,,,
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        entropies = []
        last_done = False

        state, _ = env.reset()

        ## this will run, for n_steps, 
        for _ in range(N_STEPS): 
            state = torch.tensor(state, dtype=torch.float32)
            logits, value = model(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probability = dist.log_prob(action)

            ## now the env will step, and it will return the new things like new observation or new state 
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            last_done = done

            ## append the must need things
            states.append(state)
            actions.append(action)
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            values.append(value.squeeze())
            log_probs.append(log_probability)
            entropies.append(dist.entropy())


            ## here if the current state is terminal, then the next_state will be env.reset means---> it will start from initial state again ok...
            if done: 
                next_state, _ = env.reset() 
            
            state = next_state 

        ## let's build the logic for the next_value  --> to later calculate the target 
        if last_done:
            next_value = torch.tensor(0.0)
        else: 
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                _, next_value = model(state_tensor)
                next_value = next_value.squeeze(-1)



        ## now compute the returns here, 
        returns = calculate_returns(rewards, next_value)  ## returns is a torch Tensor


        ## calculate the advantage 
        ## first let's stack the values
        values = torch.stack(values)
        advantage = returns - values.detach()  ## this is the advantage 


        ## now let's calculate the loss here
        # Prepare tensors for loss computation
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)
        loss = compute_loss(log_probs_tensor, advantage, values, entropies_tensor)
        
        ## now the optimizer will do the rest
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum([r.item() for r in rewards])
        # print progress every 100 episodes (use 1-based episode numbering for readability)
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1} -- Return: {total_reward}  loss: {loss.item():.4f}")

        # append metrics for this episode so callers can examine training results
        metrics.append({"episode": episode, "total_reward": total_reward, "loss": round(loss.item(), 4)})

    # return collected metrics so callers (or tests) can inspect training progress
    return metrics




        


