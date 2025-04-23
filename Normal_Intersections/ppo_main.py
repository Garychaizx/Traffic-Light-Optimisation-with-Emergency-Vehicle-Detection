import sumo_rl
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
from agents.ppo_agent import PPO  # Assuming you saved your PPO class in agents/ppo.py
from tensorflow.keras.models import load_model
import librosa
import os
from generator import TrafficGenerator

# === Siren Detection ===
def extract_features(audio_file, max_pad_len=862):
    try:
        audio, sr = librosa.load(audio_file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=80)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        features = np.mean(mfccs, axis=1)
        return features.reshape(1, 1, 80)
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

def detect_siren():
    audio_path = "dynamic_sounds/ambulance.wav"
    if not os.path.exists(audio_path):
        return False
    features = extract_features(audio_path)
    if features is not None:
        prediction = siren_model.predict(features)
        return prediction[0][0] > 0.5
    return False

# === Load Siren Model ===
try:
    siren_model = load_model('siren_model/best_model.keras')
    print("✅ Siren model loaded!")
except Exception as e:
    print(f"❌ Failed to load siren model: {e}")
    siren_model = None

# === Generate Routes ===
gen = TrafficGenerator(max_steps=5000, n_cars_generated=1000)
gen.generate_routefile(seed=42)

# === SUMO Env Setup ===
env = sumo_rl.SumoEnvironment(
    net_file='nets/intersection/environment.net.xml',
    route_file='nets/intersection/episode_routes.rou.xml',
    use_gui=True,
    num_seconds=5000,
    single_agent=False
)

tl_id = env.ts_ids[0]
num_phases = len(env.traffic_signals[tl_id].all_phases)
state_dim = len(env.reset()[tl_id])
agent = PPO(state_dim, num_phases, hidden_size=64, lr=3e-4, gamma=0.99, clip_ratio=0.2, K_epoch=10)

# === PPO Training ===
episode_rewards = []
phase_visit_count = {p: 0 for p in range(num_phases)}
max_steps = 5000
epsilon = 1.0
min_epsilon = 0.1
decay = 1 / 10000

for episode in range(1):
    observations = env.reset()
    state = observations[tl_id]
    episode_reward = 0

    states, actions, old_probs, rewards = [], [], [], []

    done = {"__all__": False}
    i = 0
    while not done["__all__"]:
        i += 1

        current_phase = env.traffic_signals[tl_id].green_phase
        valid_phases = [p for p in range(num_phases) if (current_phase, p) in env.traffic_signals[tl_id].yellow_dict]
        if not valid_phases:
            valid_phases = [current_phase]

        # === Emergency Override ===
        emergency_override = False
        if siren_model is not None and detect_siren():
            print("🚨 Emergency detected — overriding PPO decision!")
            for edge, phase in {"N2TL": 0, "S2TL": 0, "E2TL": 2, "W2TL": 2}.items():
                for veh_id in env.sumo.vehicle.getIDList():
                    if env.sumo.vehicle.getRoadID(veh_id).startswith(edge) and env.sumo.vehicle.getTypeID(veh_id) == "emergency_veh":
                        action = phase
                        emergency_override = True
                        break
                if emergency_override:
                    break

        if not emergency_override:
            # Encourage exploration of new phases
            unexplored = [p for p in valid_phases if phase_visit_count[p] == 0]
            if unexplored:
                action = random.choice(unexplored)
            elif random.random() < epsilon:
                action = random.choice(valid_phases)
            else:
                raw_action = agent.choose_action(state)
                action = raw_action if raw_action in valid_phases else random.choice(valid_phases)

        phase_visit_count[action] += 1
        actions_dict = {tl_id: action}

        next_obs, reward_dict, done, _ = env.step(actions_dict)
        reward = reward_dict[tl_id]
        next_state = next_obs[tl_id]

        # Track data for PPO
        states.append(state)
        actions.append(action)
        old_probs.append(agent.actor_critic.actor(torch.FloatTensor(state)).squeeze()[action].item())
        rewards.append(reward)

        state = next_state
        episode_reward += reward
        if epsilon > min_epsilon:
            epsilon -= decay

        print(f"Step {i}: Reward = {reward:.2f}, Action (phase): {action}")

    agent.update(states, actions, old_probs, rewards)
    episode_rewards.append(episode_reward)
    print(f"✅ Episode finished. Total Reward: {episode_reward:.2f}")

# === Save model ===
torch.save(agent.actor_critic.state_dict(), 'trained_models/model_ppo.pth')

# === Plot Reward ===
plt.figure(figsize=(9, 5))
plt.plot(episode_rewards)
plt.title("PPO Episode Rewards with Emergency Override")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.show()
