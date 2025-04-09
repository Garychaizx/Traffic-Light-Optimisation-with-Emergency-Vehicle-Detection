import sumo_rl
import numpy as np
import torch
import random
import os
import librosa
from matplotlib import pyplot as plt
from agents.sac_agent import SACAgent
from tensorflow.keras.models import load_model

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
        print(f"Error extracting features: {e}")
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
except Exception as e:
    print(f"❌ Failed to load siren model: {e}")
    siren_model = None

# === SUMO Environment ===
env = sumo_rl.SumoEnvironment(
    net_file='nets/intersection/environment.net.xml',
    route_file='nets/intersection/episode_routes.rou.xml',
    use_gui=True,
    num_seconds=5000,
    single_agent=False
)

tl_id = env.ts_ids[0]
print(f"Testing SAC agent on traffic light: {tl_id}")

num_phases = len(env.traffic_signals[tl_id].all_phases)
observations = env.reset()
state_dim = len(observations[tl_id])

# === Load SAC Agent ===
agent = SACAgent(state_dim=state_dim, action_dim=num_phases)
agent.actor.load_state_dict(torch.load('trained_models/sac_actor.pth'))
agent.actor.eval()

# === Testing Loop ===
done = {"__all__": False}
rewards = []
queue_lengths = []
phase_counts = {p: 0 for p in range(num_phases)}
i = 0
state = observations[tl_id]

while not done["__all__"]:
    i += 1

    current_phase = env.traffic_signals[tl_id].green_phase
    valid_phases = [p for p in range(num_phases) if (current_phase, p) in env.traffic_signals[tl_id].yellow_dict]
    if not valid_phases:
        valid_phases = [current_phase]

    # === Emergency Vehicle Override ===
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
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = agent.actor(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()
        if action not in valid_phases:
            action = random.choice(valid_phases)

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
print("\n✅ SAC Evaluation completed.")
print(f"Average reward: {np.mean(rewards):.2f}")
print("\nPhase usage:")
for p, count in phase_counts.items():
    print(f"Phase {p}: {count} times")

# === Plotting ===
plt.figure(figsize=(10, 4))
plt.plot(rewards, label='Reward')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('SAC Testing Rewards over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(queue_lengths, color='orange', label='Queue Length')
plt.xlabel('Step')
plt.ylabel('Queue Length')
plt.title('SAC Queue Length over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
