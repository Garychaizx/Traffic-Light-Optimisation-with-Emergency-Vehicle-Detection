import sumo_rl
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
from agents.ql_agent import QlAgent
from tensorflow.keras.models import load_model
import librosa
import os
from generator import TrafficGenerator
import traci

gen = TrafficGenerator(max_steps=5000, n_cars_generated=1000)
gen.generate_routefile(seed=42)

# === Siren Detection Utilities ===
def extract_features(audio_file, max_pad_len=862):
    try:
        audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
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
    audio_path = "dynamic_sounds/ambulance.wav"  # Replace with real-time audio in deployment
    if not os.path.exists(audio_path):
        return False
    features = extract_features(audio_path)
    if features is not None:
        prediction = siren_model.predict(features)
        return prediction[0][0] > 0.5
    return False

# === Load Siren Detection Model ===
try:
    siren_model = load_model('siren_model/best_model.keras')
    print("✅ Siren model loaded!")
except Exception as e:
    print(f"❌ Failed to load siren model: {e}")
    siren_model = None

# === SUMO Environment ===
env = sumo_rl.SumoEnvironment(
    net_file='nets/intersection/environment.net.xml',
    route_file='nets/intersection/episode_routes_high.rou.xml',
    use_gui=True,
    num_seconds=5000,
    single_agent=False
)

# === Get traffic light ID & phase info ===
tl_id = env.ts_ids[0]
print(f"Controlling traffic light: {tl_id}")

num_phases = len(env.traffic_signals[tl_id].all_phases)
print(f"Number of valid phases: {num_phases}")

# === Agent Setup ===
observations = env.reset()
state_dim = len(observations[tl_id])
agent = QlAgent(input_shape=state_dim, output_shape=num_phases)

# === Training Params ===
gamma = 0.9
epsilon = 1.0
min_epsilon = 0.1
decay = 1 / 10000
avg_rewards = []

# === Phase coverage control ===
phase_visit_count = {p: 0 for p in range(num_phases)}

# Define shaping weights
QUEUE_PENALTY = 0.3          # Penalize the largest queue
CLEAR_BONUS = 0.2            # Bonus for clearing vehicles
PHASE_CHANGE_PENALTY = 0.05  # Penalize phase changes
EMERGENCY_BONUS = 1.0        # Bonus for emergency vehicle handling
GAMMA = 0.9                  # Discount factor for potential-based shaping

# Initialize variables for shaping
prev_total_queue = 0

# === Training Loop ===
done = {"__all__": False}
i = 0
input_tensor = torch.FloatTensor(observations[tl_id])

while not done["__all__"]:
    i += 1

    pred_rewards = agent.predict_rewards(input_tensor)
    current_phase = env.traffic_signals[tl_id].green_phase
    valid_phases = [p for p in range(num_phases) if (current_phase, p) in env.traffic_signals[tl_id].yellow_dict]
    if not valid_phases:
        valid_phases = [current_phase]

    # === Emergency Override ===
    emergency_override = False
    if siren_model is not None and detect_siren():
        print("🚨 Emergency detected — overriding Q-learning decision!")
        for edge, phase in {"N2TL": 0, "S2TL": 0, "E2TL": 2, "W2TL": 2}.items():
            for veh_id in env.sumo.vehicle.getIDList():
                if env.sumo.vehicle.getRoadID(veh_id).startswith(edge) and env.sumo.vehicle.getTypeID(veh_id) == "emergency_veh":
                    action = phase
                    emergency_override = True
                    break
            if emergency_override:
                break

    if not emergency_override:
        unexplored = [p for p in valid_phases if phase_visit_count[p] == 0]
        if unexplored:
            action = random.choice(unexplored)
        elif random.random() < epsilon:
            action = random.choice(valid_phases)
        else:
            pred = pred_rewards[valid_phases]
            action = valid_phases[torch.argmax(pred).item()]

    phase_visit_count[action] += 1

    # Measure queue before the action
    before_queue = sum(
        traci.lane.getLastStepHaltingNumber(l)
        for l in traci.trafficlight.getControlledLanes(tl_id)
    )

    # Step the environment
    actions = {tl_id: action}
    observations, rewards, done, infos = env.step(actions)
    next_input_tensor = torch.FloatTensor(observations[tl_id])
    base_reward = rewards[tl_id]

    # Measure queue after the action
    after_queue = sum(
        traci.lane.getLastStepHaltingNumber(l)
        for l in traci.trafficlight.getControlledLanes(tl_id)
    )
    cleared_vehicles = max(0, before_queue - after_queue)

    # === Shaped Reward Calculation ===
    # 1. Queue penalty
    queue_penalty = -QUEUE_PENALTY * before_queue

    # 2. Bonus for clearing vehicles
    clear_bonus = CLEAR_BONUS * cleared_vehicles

    # 3. Phase change penalty
    phase_change_penalty = -PHASE_CHANGE_PENALTY if action != current_phase else 0.0

    # 4. Emergency vehicle bonus
    emergency_bonus = EMERGENCY_BONUS if emergency_override else 0.0

    # 5. Potential-based shaping
    potential_diff = GAMMA * (-after_queue) - (-prev_total_queue)

    # Combine all components into the shaped reward
    shaped_reward = (
        base_reward
        + queue_penalty
        + clear_bonus
        + phase_change_penalty
        + emergency_bonus
        + potential_diff
    )

    # Update previous total queue
    prev_total_queue = after_queue

    # Q-learning update
    with torch.no_grad():
        q_target = shaped_reward + GAMMA * torch.max(agent.predict_rewards(next_input_tensor)[valid_phases])
    agent.learn(pred_rewards[action].unsqueeze(0), torch.tensor([q_target]))

    # Update state
    input_tensor = next_input_tensor

    # Decay epsilon
    if epsilon > min_epsilon:
        epsilon -= decay

    avg_rewards.append(shaped_reward)
    print(f"Step {i}: Shaped Reward = {shaped_reward:.2f}, Base Reward = {base_reward:.2f}, Action (phase): {action}")

# === Save model ===
torch.save(agent.model.state_dict(), 'trained_models/model_ql.pth')

# === Plot Rewards ===
plt.figure(figsize=(9, 5))
plt.plot(avg_rewards)
plt.xlabel("Steps")
plt.ylabel("Shaped Reward")
plt.title("Q-Learning with Shaped Rewards on Single Intersection")
plt.grid(True)
plt.tight_layout()
plt.show()
