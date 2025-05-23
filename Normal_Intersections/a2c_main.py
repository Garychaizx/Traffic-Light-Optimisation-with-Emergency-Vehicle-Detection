import sumo_rl
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import os
from agents.a2c_agent import A2CAgent
from generator import TrafficGenerator
import librosa
import traci
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

# === Load Siren Detection Model ===
try:
    siren_model = load_model('siren_model/best_model.keras')
    print("✅ Siren model loaded!")
except Exception as e:
    print(f"❌ Failed to load siren model: {e}")
    siren_model = None

# === Generate Traffic Routes ===
gen = TrafficGenerator(max_steps=5000, n_cars_generated=1000)
gen.generate_routefile(seed=42)

# === SUMO Environment Setup ===
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

# Initialize the A2C Agent
agent = A2CAgent(state_dim, num_phases)
state = observations[tl_id]

# Initialize reward log
rewards_log = []
done = {"__all__": False}
i = 0

# Define shaping weights
QUEUE_PENALTY = 0.3  # Penalize large queues
CLEAR_BONUS = 0.2    # Bonus for clearing vehicles
PHASE_CHANGE_PENALTY = 0.05  # Penalize phase changes
EMERGENCY_BONUS = 1.0  # Bonus for emergency vehicle handling

# Initialize variables for shaping
prev_total_queue = 0
gamma = 0.99  # Discount factor for potential-based shaping

while not done["__all__"]:
    i += 1
    current_phase = env.traffic_signals[tl_id].green_phase
    valid_phases = [p for p in range(num_phases) if (current_phase, p) in env.traffic_signals[tl_id].yellow_dict]
    if not valid_phases:
        valid_phases = [current_phase]

    # === Emergency Siren Override ===
    emergency_override = False
    emergency_dir = None
    for car_id in env.sumo.vehicle.getIDList():
        if env.sumo.vehicle.getTypeID(car_id) == "emergency_veh":
            road_id = env.sumo.vehicle.getRoadID(car_id)
            dist = 750 - env.sumo.vehicle.getLanePosition(car_id)
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

    actions = {tl_id: action}
    observations, reward_dict, done, _ = env.step(actions)
    next_state = observations[tl_id]
    base_reward = reward_dict[tl_id]
    done_flag = float(done["__all__"])

    # === Shaped Reward Calculation ===
    # 1. Queue penalty
    total_queue = sum(
        traci.lane.getLastStepHaltingNumber(l)
        for l in traci.trafficlight.getControlledLanes(tl_id)
    )
    queue_penalty = -QUEUE_PENALTY * total_queue

    # 2. Bonus for clearing vehicles
    cleared_vehicles = max(0, prev_total_queue - total_queue)
    clear_bonus = CLEAR_BONUS * cleared_vehicles

    # 3. Phase change penalty
    phase_change_penalty = -PHASE_CHANGE_PENALTY if action != current_phase else 0.0

    # 4. Emergency vehicle bonus
    emergency_bonus = EMERGENCY_BONUS if emergency_override else 0.0

    # 5. Potential-based shaping
    potential_diff = gamma * (-total_queue) - (-prev_total_queue)

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
    prev_total_queue = total_queue

    # Store experience and update the A2C model
    agent.update(state, action, shaped_reward, next_state, done_flag)

    # Set next state
    state = next_state
    rewards_log.append(shaped_reward)

    print(f"Step {i}: Shaped Reward = {shaped_reward:.2f}, Base Reward = {base_reward:.2f}, Phase = {action}")

# === Save the A2C model ===
torch.save(agent.actor.state_dict(), 'trained_models/a2c_actor.pth')
torch.save(agent.critic.state_dict(), 'trained_models/a2c_critic.pth')

# === Plot Rewards ===
plt.figure(figsize=(10, 4))
plt.plot(rewards_log)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('A2C Training Rewards Over Time')
plt.grid(True)
plt.tight_layout()
plt.show()
