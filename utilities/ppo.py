import sys             # SKIPPED
import numpy as np

import random

import torch
import torch.nn.functional as F
import utilities.constants as ct
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

seed = 64
SEED = int(seed)   
random.seed(SEED)
np.random.seed(SEED)

'''
    Calculates Generalized Advantage Estimate
'''
def compute_gae(next_value, rewards, masks, values, gamma=0.9, tau=0.99):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

'''
    Takes random steps/iterations of the episode
'''
def ppo_iter(states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // ct.MINI_BATCH_SIZE):
        rand_ids = np.random.randint(0, batch_size, ct.MINI_BATCH_SIZE)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

'''
    Updates current policy by using sampled steps/iterations from ppo_iter
'''
def ppo_update(model_musculo, optimizer_musculo, states, actions, log_probs, returns, advantages):
    actor_losses = []
    critic_losses = []
    entropies = []
    losses = []
    kl_divs = []
    stds = []
    for _ in range(ct.NUM_EPOCHS):
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
            dist, value, std = model_musculo(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - ct.CLIP_PARAM, 1.0 + ct.CLIP_PARAM) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            kl_divs.append(F.kl_div(new_log_probs, old_log_probs, reduction='batchmean'))

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropies.append(entropy)
            losses.append(loss)
            stds.append(std.mean())

            optimizer_musculo.zero_grad()
            loss.backward()
            optimizer_musculo.step()

    actor_losses = torch.stack(actor_losses)
    critic_losses = torch.stack(critic_losses)
    entropies = torch.stack(entropies)
    losses = torch.stack(losses)
    kl_divs = torch.stack(kl_divs)
    stds = torch.stack(stds)
    return actor_losses, critic_losses, entropies, losses, kl_divs, stds