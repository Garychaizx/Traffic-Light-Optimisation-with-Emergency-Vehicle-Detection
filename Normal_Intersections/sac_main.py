import sumo_rl
import torch
import numpy as np
import random
import os
import librosa
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from agents.sac_agent import SACAgent  # Make sure this is implemented properly
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
        return np.mean(mfccs, axis=1).reshape(1, 1, 80)
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
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

# === Traffic Generator ===
gen = TrafficGenerator(max_steps=5000, n_cars_generated=1000)
gen.generate_routefile(seed=42)

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
state_dim = len(env.reset()[tl_id])

# === SAC Agent ===
agent = SACAgent(state_dim=state_dim, action_dim=num_phases)

# === Training ===
# === Training Loop (Episodic) ===
num_episodes = 1
max_steps_per_episode = 5000
epsilon = 1.0
min_epsilon = 0.1
decay = 1 / (num_episodes * max_steps_per_episode)
episode_rewards = []
step_rewards = []
steps = []  # this will track x-axis labels

for episode in range(num_episodes):
    observations = env.reset()
    state = observations[tl_id]
    episode_reward = 0
    done = {"__all__": False}
    step = 0

    while not done["__all__"] and step < max_steps_per_episode:
        step += 1

        # Emergency Handling
        current_phase = env.traffic_signals[tl_id].green_phase
        valid_phases = [p for p in range(num_phases) if (current_phase, p) in env.traffic_signals[tl_id].yellow_dict]
        if not valid_phases:
            valid_phases = [current_phase]

        emergency_override = False
        if siren_model is not None and detect_siren():
            for edge, phase in {"N2TL": 0, "S2TL": 0, "E2TL": 2, "W2TL": 2}.items():
                for veh_id in env.sumo.vehicle.getIDList():
                    if env.sumo.vehicle.getRoadID(veh_id).startswith(edge) and env.sumo.vehicle.getTypeID(veh_id) == "emergency_veh":
                        action = phase
                        emergency_override = True
                        break
                if emergency_override:
                    break

        if not emergency_override:
            if random.random() < epsilon:
                action = random.choice(valid_phases)
            else:
                action = agent.choose_action(state)
                if action not in valid_phases:
                    action = random.choice(valid_phases)

        next_obs, reward_dict, done, _ = env.step({tl_id: action})
        next_state = next_obs[tl_id]
        reward = reward_dict[tl_id]

        agent.store_experience(state, action, reward, next_state, float(done["__all__"]))
        agent.update(batch_size=64)

        state = next_state
        episode_reward += reward
        step_rewards.append(reward)
        steps.append(step)

        if epsilon > min_epsilon:
            epsilon -= decay

        print(f"Episode {episode + 1}, Step {step}: Reward = {reward:.2f}, Action = {action}")

    episode_rewards.append(episode_reward)
    print(f"✅ Episode {episode + 1} finished. Total Reward: {episode_reward:.2f}")


# === Save Models ===
os.makedirs('trained_models', exist_ok=True)
torch.save(agent.actor.state_dict(), 'trained_models/sac_actor.pth')
torch.save(agent.critic_1.state_dict(), 'trained_models/sac_critic1.pth')
torch.save(agent.critic_2.state_dict(), 'trained_models/sac_critic2.pth')

# === Plot Rewards ===
plt.figure(figsize=(10, 4))
plt.plot(steps, step_rewards, label='Step-wise Reward')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('SAC Step-wise Rewards')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

