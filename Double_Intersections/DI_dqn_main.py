# test_dqn_double.py

import os
import numpy as np
import torch
import sumo_rl
import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent

def build_neighbours(env):
    """
    After env.reset(), build a map from each TL id to the
    list of its two‑char neighbours (edges like "AB", "BC", etc.).
    """
    ts = env.ts_ids
    neigh = {tl: [] for tl in ts}
    for eid in env.sumo.edge.getIDList():
        if len(eid) == 2 and eid[0] in ts and eid[1] in ts:
            a, b = eid[0], eid[1]
            neigh[a].append(b)
            neigh[b].append(a)
    return {tl: list(set(lst)) for tl, lst in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=165):
    """
    Stack own observation + 0.2× each neighbour’s,
    then pad (with -1) or truncate to pad_len.
    Returns a flat numpy array.
    """
    x = obs[tl].copy()
    for n in neighbours.get(tl, []):
        x += 0.2 * obs[n]
    if len(x) < pad_len:
        x = np.concatenate([x, np.full(pad_len - len(x), -1.0)])
    else:
        x = x[:pad_len]
    return x

def main():
    # 1) Create the SUMO environment with GUI for visualization
    env = sumo_rl.SumoEnvironment(
        net_file    = "nets/double/network.net.xml",
        route_file  = "nets/double/doubleRoutes_high.rou.xml",
        use_gui     = True,
        num_seconds = 20000,
        single_agent=False
    )

    # 2) Reset and build neighbour map
    obs0 = env.reset()
    tls  = env.ts_ids
    neighbours = build_neighbours(env)

    # Initialize DQN agents for training (not loading pre-trained)
    agents = {}
    for tl in tls:
        n_phases = len(env.traffic_signals[tl].all_phases)
        agent = DQNAgent(
            state_dim   = 165,
            action_dim  = n_phases,
            gamma      = 0.99,
            lr         = 1e-3,
            epsilon    = 1.0,      # Start with full exploration
            min_epsilon = 0.05,    # Minimum exploration rate
            epsilon_decay = 1e-5,  # Gradual exploration decay
            batch_size = 64,
            memory_size = 20000    # Replay buffer size
        )
        agents[tl] = agent

    # Define shaping weights
    QUEUE_PENALTY = 0.3          # Penalize the largest queue
    CLEAR_BONUS = 0.2            # Bonus for clearing vehicles
    PHASE_CHANGE_PENALTY = 0.05  # Penalize phase changes
    GAMMA = 0.99                 # Discount factor for potential-based shaping

    # Initialize variables for shaping
    prev_total_queues = {tl: 0 for tl in tls}

    # Initialize variables
    state = {tl: prepare_obs(obs0, tl, neighbours) for tl in tls}
    done = {"__all__": False}
    rewards = []
    queues = []
    phase_counts = {tl: {p:0 for p in range(len(env.traffic_signals[tl].all_phases))} for tl in tls}
    step = 0
    MAX_STEPS = 5000

    # Main training loop
    while not done["__all__"] and step < MAX_STEPS:
        step += 1
        actions = {}

        # Select actions with exploration
        for tl, agent in agents.items():
            curr = env.traffic_signals[tl].green_phase
            valid = [p for p in range(agent.action_dim)
                     if (curr, p) in env.traffic_signals[tl].yellow_dict]
            if not valid:
                valid = [curr]
            a = agent.choose_action(state[tl], valid)
            phase_counts[tl][a] += 1
            actions[tl] = a

        # Measure queue before the action
        before_queues = {}
        for tl in tls:
            lanes = env.sumo.trafficlight.getControlledLanes(tl)
            before_queues[tl] = sum(env.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)

        # Step the environment
        obs2, reward_dict, done, _ = env.step(actions)

        # Measure queue after the action
        after_queues = {}
        cleared_vehicles = {}
        for tl in tls:
            lanes = env.sumo.trafficlight.getControlledLanes(tl)
            after_queues[tl] = sum(env.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
            cleared_vehicles[tl] = max(0, before_queues[tl] - after_queues[tl])

        # === Shaped Reward Calculation ===
        shaped_rewards = {}
        for tl in tls:
            # 1. Queue penalty
            queue_penalty = -QUEUE_PENALTY * before_queues[tl]

            # 2. Bonus for clearing vehicles
            clear_bonus = CLEAR_BONUS * cleared_vehicles[tl]

            # 3. Phase change penalty
            phase_change_penalty = -PHASE_CHANGE_PENALTY if actions[tl] != env.traffic_signals[tl].green_phase else 0.0

            # 4. Potential-based shaping
            potential_diff = GAMMA * (-after_queues[tl]) - (-prev_total_queues[tl])

            # Combine all components into the shaped reward
            shaped_rewards[tl] = (
                reward_dict[tl]
                + queue_penalty
                + clear_bonus
                + phase_change_penalty
                + potential_diff
            )

            # Update previous total queue
            prev_total_queues[tl] = after_queues[tl]

        # Record avg shaped reward
        avg_shaped_reward = np.mean(list(shaped_rewards.values()))
        rewards.append(avg_shaped_reward)

        # Record avg queue length
        total_q = sum(after_queues.values())
        queues.append(total_q / len(tls))

        print(f"Step {step:4d} | Avg Shaped Reward {avg_shaped_reward: .3f} | Avg Queue {queues[-1]: .1f}")

        # === Store experience and train ===
        for tl in tls:
            next_state = prepare_obs(obs2, tl, neighbours)
            agent = agents[tl]
            
            # Store experience in replay buffer
            agent.store_experience(
                state[tl], 
                actions[tl], 
                shaped_rewards[tl], 
                next_state, 
                float(done["__all__"])
            )
            
            # Update DQN model
            agent.update_model()

            # Update state
            state[tl] = next_state

        # Print training progress
        if step % 100 == 0:
            avg_reward = np.mean(list(shaped_rewards.values()))
            avg_queue = np.mean(list(after_queues.values()))
            print(
                f"Step {step:4d} | "
                f"Avg Reward {avg_reward:.3f} | "
                f"Avg Queue {avg_queue:.1f} | "
                f"ε {agents[tls[0]].epsilon:.3f}"
            )

    # Save trained models
    print("\n✅ Training complete. Saving models...")
    os.makedirs("trained_models", exist_ok=True)
    for tl in tls:
        model_path = f"trained_models/dqn_double_{tl}.pth"
        agents[tl].save(model_path)
        print(f"Saved {model_path}")

    # === Summary ===
    print("\n✅ Evaluation complete.")
    print(f"Mean shaped reward over episode: {np.mean(rewards):.3f}")
    print(f"Mean queue length over episode: {np.mean(queues):.1f}")
    for tl in tls:
        print(f"\nPhase counts for {tl}:")
        for p, c in phase_counts[tl].items():
            print(f"  Phase {p}: {c}")

    # === Plotting ===
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label="Reward")
    plt.xlabel("Step"); plt.ylabel("Reward"); plt.title("DQN Training: Reward Over Time")
    plt.grid(True); plt.legend()

    plt.figure(figsize=(10, 4))
    plt.plot(queues, color="orange", label="Avg Queue")
    plt.xlabel("Step"); plt.ylabel("Queue Length"); plt.title("DQN Double Intersection: Queue")
    plt.grid(True); plt.legend()

    plt.show()
    env.close()

if __name__ == "__main__":
    main()