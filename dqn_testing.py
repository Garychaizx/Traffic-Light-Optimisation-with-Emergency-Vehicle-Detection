import sumo_rl
import torch
import numpy as np
import random
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import librosa
import os
from agents.dqn_agent import DQNAgent

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
    except:
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
    print("✅ Siren detection model loaded!")
except:
    siren_model = None
    print("❌ Siren model not found.")

# === SUMO Environment ===
env = sumo_rl.SumoEnvironment(
    net_file='nets/intersection/environment.net.xml',
    route_file='nets/intersection/episode_routes.rou.xml',
    use_gui=True,
    num_seconds=5000,
    single_agent=False
)

tl_id = env.ts_ids[0]
num_phases = len(env.traffic_signals[tl_id].all_phases)
observations = env.reset()
state_dim = len(observations[tl_id])

agent = DQNAgent(state_dim, num_phases)
agent.load('trained_models/model_dqn.pth')
agent.epsilon = 0.0  # Greedy evaluation

state = observations[tl_id]
rewards = []
queue_lengths = []
phase_counts = {p: 0 for p in range(num_phases)}
done = {"__all__": False}
i = 0

while not done["__all__"]:
    i += 1

    current_phase = env.traffic_signals[tl_id].green_phase
    valid_phases = [p for p in range(num_phases) if (current_phase, p) in env.traffic_signals[tl_id].yellow_dict]
    if not valid_phases:
        valid_phases = [current_phase]

    emergency_override = False
    emergency_dir = None
    for veh_id in env.sumo.vehicle.getIDList():
        if env.sumo.vehicle.getTypeID(veh_id) == "emergency_veh":
            road_id = env.sumo.vehicle.getRoadID(veh_id)
            dist = 750 - env.sumo.vehicle.getLanePosition(veh_id)
            if dist < 100 and detect_siren():
                emergency_override = True
                emergency_dir = road_id
                break

    if emergency_override:
        if emergency_dir.startswith("N2TL") or emergency_dir.startswith("S2TL"):
            action = 0
        elif emergency_dir.startswith("E2TL") or emergency_dir.startswith("W2TL"):
            action = 2
    else:
        action = agent.choose_action(state, valid_phases)

    phase_counts[action] += 1
    actions = {tl_id: action}
    observations, reward_dict, done, _ = env.step(actions)

    state = observations[tl_id]
    reward = reward_dict[tl_id]
    rewards.append(reward)

    incoming_lanes = env.sumo.trafficlight.getControlledLanes(tl_id)
    total_queue = sum(env.sumo.lane.getLastStepHaltingNumber(lane) for lane in incoming_lanes)
    queue_lengths.append(total_queue)

    print(f"Step {i}: Reward = {reward:.2f}, Phase = {action}, Queue Length = {total_queue}")

# === Summary ===
print("\n✅ Evaluation complete.")
print(f"Average reward: {np.mean(rewards):.2f}")
for p, count in phase_counts.items():
    print(f"Phase {p}: {count} times")

# === Plotting ===
plt.figure(figsize=(10, 4))
plt.plot(rewards, label='Reward')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('DQN Testing Rewards Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(queue_lengths, color='orange', label='Queue Length')
plt.xlabel('Step')
plt.ylabel('Queue Length')
plt.title('Queue Length Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
