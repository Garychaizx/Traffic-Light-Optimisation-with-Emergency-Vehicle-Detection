# train_ppo_intersection_with_shaping.py

import os
import random
import numpy as np
import torch
import sumo_rl
import traci
from agents.ppo_agent import PPO
from tensorflow.keras.models import load_model
import librosa
import matplotlib.pyplot as plt
from generator import TrafficGenerator

# === Siren Detection ===
def extract_features(audio_file, max_pad_len=862):
    try:
        audio, sr = librosa.load(audio_file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=80)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return np.mean(mfccs, axis=1).reshape(1,1,80)
    except:
        return None

def detect_siren():
    path = "dynamic_sounds/ambulance.wav"
    if not os.path.exists(path) or siren_model is None:
        return False
    feats = extract_features(path)
    if feats is None:
        return False
    return float(siren_model.predict(feats)[0][0]) > 0.5

def main():
    # === load siren model ===
    try:
        global siren_model
        siren_model = load_model('siren_model/best_model.keras')
        print("✅ Siren model loaded")
    except:
        siren_model = None
        print("⚠️ No siren model, skipping override")

    # === generate routes ===
    gen = TrafficGenerator(max_steps=5000, n_cars_generated=1000)
    gen.generate_routefile(seed=42)

    # === build SUMO env ===
    env = sumo_rl.SumoEnvironment(
        net_file    = 'nets/intersection/environment.net.xml',
        route_file  = 'nets/intersection/episode_routes.rou.xml',
        use_gui     = True,
        num_seconds = 5000,
        single_agent=False
    )
    tl_id = env.ts_ids[0]
    num_phases = len(env.traffic_signals[tl_id].all_phases)

    # === grab initial observation & controlled‐lanes info ===
    obs0 = env.reset()
    ctrl_lanes = traci.trafficlight.getControlledLanes(tl_id)
    lanes_by_phase = [
        [lane for lane, sig in zip(ctrl_lanes, ph.state) if sig == 'G']
        for ph in env.traffic_signals[tl_id].all_phases
    ]

    # === PPO agent ===
    state_dim = len(obs0[tl_id]) + num_phases
    agent = PPO(
        state_dim   = state_dim,
        action_dim  = num_phases,
        hidden_size = 64,
        lr          = 3e-4,
        gamma       = 0.99,
        clip_ratio  = 0.2,
        K_epoch     = 10
    )

    # === shaping hyper-parameters ===
    QUEUE_PEN        = 0.3   # w1: penalize total queue
    SERVE_BONUS      = 0.2   # w2: cleared on chosen phase
    THROUGHPUT_BONUS = 0.2   # w3: total throughput
    PHASE_CHANGE_PEN = 0.05  # w4: cost for switching phases
    EMERGENCY_BONUS  = 10.0  # w6: when siren detected & override taken

    episode_rewards = []
    epsilon         = 1.0
    min_epsilon     = 0.1
    decay           = 1/10000

    # === one training episode ===
    observations = obs0
    # build initial state vector
    raw0 = observations[tl_id]
    q0   = np.array([
        sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
        for p in range(num_phases)
    ], dtype=float)
    total_q0 = q0.sum()
    feat0    = q0/(total_q0+1e-3)
    state    = np.concatenate([raw0, feat0])
    prev_phase = env.traffic_signals[tl_id].green_phase

    done = {"__all__": False}
    step = 0
    states, actions, old_probs, rewards = [], [], [], []
    episode_reward = 0.0

    while not done["__all__"]:
        step += 1
        # valid next phases
        curr = env.traffic_signals[tl_id].green_phase
        valid = [p for p in range(num_phases)
                 if (curr,p) in env.traffic_signals[tl_id].yellow_dict]
        if not valid:
            valid = [curr]

        # siren override?
        is_siren = detect_siren()
        if is_siren:
            override = 0 if curr in (0,1) else curr
            action = override
        else:
            override = None
            # exploration or greedy PPO
            if random.random() < epsilon:
                action = random.choice(valid)
            else:
                action = agent.choose_action(state)
                if action not in valid:
                    action = random.choice(valid)

        # compute shaping components **before** stepping
        queue_before = np.array([
            sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
            for p in range(num_phases)
        ], dtype=float)
        total_q_before = queue_before.sum()
        before_sel     = queue_before[action]
        change_pen     = PHASE_CHANGE_PEN if action!=prev_phase else 0.0

        # step env
        next_obs, reward_dict, done, _ = env.step({tl_id: action})

        # recompute **after** queues
        queue_after    = np.array([
            sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
            for p in range(num_phases)
        ], dtype=float)
        total_q_after = queue_after.sum()
        cleared_sel   = max(0.0, before_sel - queue_after[action])
        cleared_tot   = max(0.0, total_q_before - total_q_after)

        # potential-based shaping
        pot_diff = agent.gamma * (-total_q_after) - (-total_q_before)

        # emergency flag
        em_flag = 1 if (is_siren and action==override) else 0

        # final shaped reward
        shaped_r = (
            - QUEUE_PEN*total_q_before
            + SERVE_BONUS*cleared_sel
            + THROUGHPUT_BONUS*cleared_tot
            + pot_diff
            - change_pen
            + EMERGENCY_BONUS*em_flag
        )

        # store for PPO
        states.append(state)
        actions.append(action)
        old_probs.append(agent.actor_critic.actor(
            torch.FloatTensor(state).unsqueeze(0)
        ).detach().squeeze()[action].item())
        rewards.append(shaped_r)
        episode_reward += shaped_r

        # decay ε
        epsilon = max(min_epsilon, epsilon - decay)

        # prepare next state
        raw2 = next_obs[tl_id]
        feat2= queue_after/(total_q_after+1e-3)
        state = np.concatenate([raw2, feat2])
        prev_phase = action
        observations = next_obs

        print(f"Step {step:4d} | shaped_r {shaped_r: .2f} | ε {epsilon:.3f}")

    # update PPO
    agent.update(states, actions, old_probs, rewards)
    episode_rewards.append(episode_reward)
    print(f"✅ Episode done. Total shaped return: {episode_reward:.2f}")

    # save
    os.makedirs("trained_models", exist_ok=True)
    torch.save(agent.actor_critic.state_dict(), 'trained_models/model_ppo.pth')

    # plot episode‐level shaped return
    plt.figure(figsize=(9,5))
    plt.plot(episode_rewards, marker='o')
    plt.title("PPO Shaped Episode Returns")
    plt.xlabel("Episode")
    plt.ylabel("Sum of Rewards")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    env.close()


if __name__=="__main__":
    main()
