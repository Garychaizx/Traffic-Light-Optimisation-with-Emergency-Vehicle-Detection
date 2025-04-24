# train_ppo_double_qstyle_fixed.py

import os
import numpy as np
import torch
import torch.nn.functional as F
import sumo_rl
from agents.ppo_agent import PPO
import matplotlib.pyplot as plt

# pad length for raw + neighbor observations
PAD_LEN = 165

def build_neighbours(env):
    """Init traci first via env.reset(), then map each TL to its 2-char neighbours."""
    ts = env.ts_ids
    neigh = {tl: [] for tl in ts}
    for eid in env.sumo.edge.getIDList():
        if len(eid) == 2 and eid[0] in ts and eid[1] in ts:
            a, b = eid[0], eid[1]
            neigh[a].append(b)
            neigh[b].append(a)
    return {tl: np.array(v) for tl, v in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=PAD_LEN):
    """
    Concatenate own obs + 0.2×neighbours, pad/truncate to pad_len.
    Returns an array of length pad_len.
    """
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

    # collect per‐phase lane lists for queue measurement
    phases        = {tl: env.traffic_signals[tl].all_phases for tl in tls}
    ctrl_lanes    = {tl: env.sumo.trafficlight.getControlledLanes(tl) for tl in tls}
    lanes_by_phase= {}
    for tl in tls:
        lanes_by_phase[tl] = []
        for ph in phases[tl]:
            serve = [
                lane for lane, sig in zip(ctrl_lanes[tl], ph.state)
                if sig.upper()=='G'
            ]
            lanes_by_phase[tl].append(serve)

    # 2) one PPO agent per TL, state_dim = PAD_LEN + num_phases
    agents = {}
    for tl in tls:
        n_phases = len(phases[tl])
        state_dim= PAD_LEN + n_phases
        agents[tl] = PPO(
            state_dim   = state_dim,
            action_dim  = n_phases,
            hidden_size = 64,
            lr          = 3e-4,
            gamma       = 0.99,
            clip_ratio  = 0.2,
            K_epoch     = 4
        )

    # 3) initial state dicts, compute initial queue features
    def get_queue_feat(tl):
        q = np.array([
            sum(env.sumo.lane.getLastStepHaltingNumber(l)
                for l in lanes_by_phase[tl][p])
            for p in range(len(phases[tl]))
        ], dtype=float)
        return q / (q.max() + 1e-3)

    state = {}
    for tl in tls:
        raw   = prepare_obs(obs0, tl, neighbours)
        qfeat = get_queue_feat(tl)
        state[tl] = np.concatenate([raw, qfeat])

    done        = {"__all__": False}
    avg_rewards = []
    step        = 0
    MAX_STEPS   = 10000

    while not done["__all__"] and step < MAX_STEPS:
        step += 1
        actions   = {}
        old_probs = {}

        # 3a) select a masked legal action for each TL (using augmented state)
        for tl, agent in agents.items():
            s_tensor = torch.FloatTensor(state[tl]).unsqueeze(0)
            logits, _= agent.actor_critic(s_tensor)
            probs     = F.softmax(logits, dim=-1).squeeze(0)

            # mask out illegal transitions
            curr  = env.traffic_signals[tl].green_phase
            valid = [p for p in range(probs.size(0))
                     if (curr,p) in env.traffic_signals[tl].yellow_dict]
            if not valid: valid = [curr]

            mask   = torch.zeros_like(probs)
            mask[valid] = 1.0
            masked = probs * mask
            masked = torch.nan_to_num(masked, nan=0.0, posinf=0.0, neginf=0.0)
            if masked.sum().item() == 0.0:
                masked[valid] = 1.0/len(valid)
            else:
                masked /= masked.sum()

            a = int(torch.multinomial(masked,1))
            actions[tl]   = a
            old_probs[tl] = probs[a].item()

        # 3b) step SUMO
        obs2, rew_dict, done, _ = env.step(actions)

        # compute global queue and global reward
        total_queue_all = 0.0
        for tl in tls:
            # measure each TL's queue by summing halting vehicles on all controlled lanes
            for lane in ctrl_lanes[tl]:
                total_queue_all += env.sumo.lane.getLastStepHaltingNumber(lane)
        global_reward = - total_queue_all

        # 3c) online update for each TL with the global_reward
        for tl, agent in agents.items():
            agent.update(
                states    = [ state[tl] ],
                actions   = [ actions[tl] ],
                old_probs = [ old_probs[tl] ],
                rewards   = [ global_reward ]
            )

        # 3d) build next state & record avg reward
        next_state = {}
        for tl in tls:
            raw   = prepare_obs(obs2, tl, neighbours)
            qfeat = get_queue_feat(tl)
            next_state[tl] = np.concatenate([raw, qfeat])
        state = next_state

        avg_rewards.append(global_reward)
        if step % 500 == 0:
            print(f"Step {step:5d}  GlobalQueue {total_queue_all:.1f}  AvgRwd {global_reward:.3f}")

    # 4) save all agents’ models
    os.makedirs("trained_models", exist_ok=True)
    for tl, agent in agents.items():
        fname = f"trained_models/ppo_double_{tl}.pth"
        torch.save(agent.actor_critic.state_dict(), fname)
        print(f"→ Saved model for TL {tl} → {fname}")

    # 5) plot learning curve
    plt.figure(figsize=(8,4))
    plt.plot(avg_rewards, label='GlobalReward')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Double Intersection PPO Training (Global Reward)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6) close env
    env.close()

if __name__ == "__main__":
    train()
