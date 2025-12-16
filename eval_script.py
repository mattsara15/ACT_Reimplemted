# type: ignore[all]

import torch
import gymnasium as gym
from gymnasium.utils.save_video import save_video
from model.act import ACTModel
from model.temporal_ensemble_policy import TemporalEnsemblePolicy
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np

import gym_pusht  # Important: This registers the namespace

def main():
    checkpoint_path = (
        "checkpoints/checkpoint_epoch24_step8616.pth"  # Update with the actual path
    )
    env_name = "gym_pusht/PushT-v0"  # Update with the actual environment name
    device = torch.device("mps")

    env = gym.make(env_name, obs_type="pixels_agent_pos", render_mode="rgb_array")
    act = ACTModel(
        input_image_size=[3,84,84],
        input_state_size=2,
        action_space=[2],
        K=14,
        device=device
    )
    model = ACTModelWrapper(act).to(device)
    model.load_from_checkpoint(checkpoint_path)

    policy = TemporalEnsemblePolicy(
        model,
        model._device,
        chunk_size=14,
        num_samples=1,
        temporal_ensemble=True,
    )
    
    NUM_EVAL_EPISODES = 5

    env = RecordVideo(
        env,
        video_folder="videos",    # Folder to save videos
        name_prefix="eval",               # Prefix for video filenames
        episode_trigger=lambda x: True    # Record every episode
    )

    # Add episode statistics tracking
    env = RecordEpisodeStatistics(env, buffer_length=NUM_EVAL_EPISODES)

    for ep_num in range(NUM_EVAL_EPISODES):
        observation, _ = env.reset(seed=42)
        done = False
        per_episode_reward = 0
        out_frames = []
        while not done:
            state = torch.from_numpy(observation["agent_pos"])
            image = torch.from_numpy(observation["pixels"])

            # Convert to float32 with image from channel first in [0,255]
            # to channel last in [0,1]
            state = state.to(torch.float32)
            image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)

            state = state.to(model._device, non_blocking=True)
            image = image.to(model._device, non_blocking=True)

            state = state.unsqueeze(0)
            image = image.unsqueeze(0)

            predicted_action = policy.get_action(image, state)

            if np.isnan(predicted_action[0]) or np.isnan(predicted_action[1]):
                continue

            print(f"selected action: {predicted_action}")

            observation, reward, terminated, truncated, _ = env.step(predicted_action)

            per_episode_reward += reward

            done = terminated or truncated
            out_frames.append(env.render())

        save_video(out_frames, "videos", fps=20)
        print(f"Episode # {ep_num} reward == {per_episode_reward}")
    
    env.close()


if __name__ == "__main__":
    main()