# train_ppo_double_qstyle_fixed.py

import os
import numpy as np
import torch
import torch.nn.functional as F
import sumo_rl
from agents.ppo_agent import PPO
import matplotlib.pyplot as plt

def build_neighbours(env):
    """Init traci first via env.reset(), then map each TL to its 2‑char neighbours."""
    ts = env.ts_ids
    neigh = {tl: [] for tl in ts}
    for eid in env.sumo.edge.getIDList():
        if len(eid) == 2 and eid[0] in ts and eid[1] in ts:
            a, b = eid[0], eid[1]
            neigh[a].append(b)
            neigh[b].append(a)
    return {tl: np.array(v) for tl, v in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=165):
    """Concatenate own obs + 0.2×neighbours, pad/truncate to pad_len."""
    x = obs[tl].copy()
    for n in neighbours[tl]:
        x += 0.2 * obs[n]
    if len(x) < pad_len:
        x = np.concatenate([x, np.full(pad_len - len(x), -1.0)])
    else:
        x = x[:pad_len]
    return x

def train():
    # 1) create SUMO env
    env = sumo_rl.SumoEnvironment(
        net_file    = "nets/double/network.net.xml",
        route_file  = "nets/double/doubleRoutes.rou.xml",
        use_gui     = False,
        num_seconds = 20000,
        single_agent=False
    )
    # init traci
    obs0       = env.reset()
    tls        = env.ts_ids
    neighbours = build_neighbours(env)

    # 2) one PPO agent per TL
    agents = {}
    for tl in tls:
        n_phases = len(env.traffic_signals[tl].all_phases)
        agents[tl] = PPO(
            state_dim   = 165,
            action_dim  = n_phases,
            hidden_size = 64,
            lr          = 3e-4,
            gamma       = 0.99,
            clip_ratio  = 0.2,
            K_epoch     = 4
        )

    # 3) step‑by‑step train loop
    state       = {tl: prepare_obs(obs0, tl, neighbours) for tl in tls}
    done        = {"__all__": False}
    avg_rewards = []
    step        = 0
    MAX_STEPS   = 10000

    while not done["__all__"] and step < MAX_STEPS:
        step += 1
        actions   = {}
        old_probs = {}

        # 3a) select a masked legal action
        for tl, agent in agents.items():
            s_tensor = torch.FloatTensor(state[tl]).unsqueeze(0)
            logits, _ = agent.actor_critic(s_tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0)

            # clean any NaN/Inf before masking
            probs = torch.nan_to_num(probs, nan=1e-8, posinf=1e-8, neginf=1e-8)
            probs = torch.clamp(probs, min=0.0)

            # build legal set
            curr = env.traffic_signals[tl].green_phase
            valid = [
                p for p in range(probs.size(0))
                if (curr, p) in env.traffic_signals[tl].yellow_dict
            ]

            # **FALLBACK**: if no legal transitions, stay in current phase
            if not valid:
                valid = [curr]

            # mask & renormalize
            mask   = torch.zeros_like(probs)
            mask[valid] = 1.0
            masked = probs * mask

            # zero out any remaining NaN/Inf, clamp negatives
            masked = torch.nan_to_num(masked, nan=0.0, posinf=0.0, neginf=0.0)
            masked = torch.clamp(masked, min=0.0)

            total = masked.sum().item()
            if total == 0.0:
                # fallback to uniform over valid
                masked[valid] = 1.0 / len(valid)
            else:
                masked /= total

            a = int(torch.multinomial(masked, 1))
            actions[tl]   = a
            old_probs[tl] = probs[a].item()

        # 3b) step SUMO
        obs2, rew_dict, done, _ = env.step(actions)

        # 3c) online update for each TL
        for tl, agent in agents.items():
            r  = rew_dict[tl]
            s  = state[tl]
            a  = actions[tl]
            op = old_probs[tl]
            agent.update(
                states    = [s],
                actions   = [a],
                old_probs = [op],
                rewards   = [r]
            )

        # 3d) rotate state & record
        state = {tl: prepare_obs(obs2, tl, neighbours) for tl in tls}
        avg_rewards.append(np.mean(list(rew_dict.values())))

        if step % 500 == 0:
            print(f"Step {step:5d}  AvgRwd {avg_rewards[-1]:.3f}")

    # 4) save all agents’ models
    os.makedirs("trained_models", exist_ok=True)
    for tl, agent in agents.items():
        fname = f"trained_models/ppo_double_{tl}.pth"
        torch.save(agent.actor_critic.state_dict(), fname)
        print(f"→ Saved model for TL {tl} → {fname}")

    # 5) plot learning curve
    plt.figure(figsize=(8,4))
    plt.plot(avg_rewards)
    plt.xlabel("Step")
    plt.ylabel("Avg Reward")
    plt.title("Double Intersection PPO Training")
    plt.grid(True)
    plt.show()

    # 6) close env
    env.close()

if __name__ == "__main__":
    train()
