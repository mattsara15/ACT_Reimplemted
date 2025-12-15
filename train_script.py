# type: ignore[all]

import argparse
import os
import torch
import time
import numpy as np
import gymnasium as gym
import gym_pusht  # Important: This registers the namespace

from tqdm import tqdm
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features

from torch.utils.tensorboard import SummaryWriter

# local imports
from dataset import ACTDataLoader
from model.act import ACTModel


def parse_args():
    p = argparse.ArgumentParser(
        description="Train script for LeRobot dataset (minimal)"
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default="lerobot/pusht",
        help="Name of the dataset to load",
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default="tensorboard",
        help="Name of the directory to output logs to",
    )
    p.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for DataLoader"
    )
    p.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=30,
        help="Number of training epochs batches to run",
    )
    p.add_argument(
        "--device",
        "-d",
        type=str,
        default="mps",
        help="Device to use, e.g. 'cpu' or 'mps'",
    )
    p.add_argument("-k", type=int, default=14, help="The actions K to predict over")
    p.add_argument(
        "--grayscale",
        action="store_true",
        default=False,
        help="Convert observation.image to grayscale",
    )
    p.add_argument(
        "--render_eval",
        action="store_true",
        default=False,
        help="Render the eval in human viewable mode",
    )
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    return p.parse_args()


def evaluate_model(model, env_name, render_eval: bool, num_episodes=10, max_steps=1000):
    """Evaluate the model in the specified gym environment."""
    env = None
    print(f"Making eval with render mode: {render_eval}")
    if render_eval:
        
        env = gym.make(env_name, obs_type="pixels_agent_pos", render_mode="rgb_array")
    else:
        env = gym.make(env_name, obs_type="pixels_agent_pos")
    episode_steps = []
    episode_rewards = []
    for _ in range(num_episodes):
        observation, _ = env.reset(seed=42)
        done = False
        steps_count = 0
        rewards = 0
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

            # Add extra (empty) batch dimension, required to forward the policy
            state = state.unsqueeze(0)
            image = image.unsqueeze(0)

            action = model.select_action(image, state).cpu().detach().numpy()[0]
            print(f"Action {action}")
            if (action[0] > 0 and action[0] < 512) and (action[1] > 0 and action[1] < 512):
                observation, reward, terminated, truncated, _ = env.step(action)    
                # TODO(mattsara) implement a temporal ensemble
            else:
                print("Act fallback")
                observation, reward, terminated, truncated, _ = env.step(np.array([0,0]))
            done = terminated or truncated
            env.render()

            max_steps += 1
            rewards += reward
            if steps_count > max_steps:
                break
        episode_rewards.append(rewards)
        episode_steps.append(steps_count)

    env.close()

    return {
        "mean_episode_reward": np.mean(episode_rewards),
        "mean_episode_steps": np.mean(episode_steps),
    }


def main():
    args = parse_args()

    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logger = SummaryWriter(
        f"{args.log_dir}/{time.strftime('%d-%m-%Y_%H-%M-%S')}",
        flush_secs=1,
        max_queue=1,
    )

    # Print the chosen configuration
    print("Training Action Chunking Transformer")
    print(
        f"dataset_name={args.dataset_name}, batch_size={args.batch_size}, epochs={args.epochs}, device={args.device}"
    )

    # Configure the model from the dataset metadata
    device = torch.device(args.device)
    dataset_metadata = LeRobotDatasetMetadata(args.dataset_name)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    image_features_shape = input_features["observation.image"].shape
    if args.grayscale:
        # Modify input feature shape to be grayscale
        input_features["observation.image"].shape = (
            1,
            image_features_shape[1],
            image_features_shape[2],
        )

    model = ACTModel(
        device=device,
        input_image_size=[input_features["observation.image"].shape],
        input_state_size=input_features["observation.state"].shape[0],
        action_space=output_features["action"].shape,
        K=args.k,
    )

    act_dataset = ACTDataLoader(
        k=args.k,
        use_grayscale=args.grayscale,
        dataset_name=args.dataset_name,
        device=args.device,
    )
    dataloader = torch.utils.data.DataLoader(
        act_dataset,
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    # Run training loop.
    step_num = 0
    for epoch in range(args.epochs):
        # train for one epoch
        for batch in tqdm(dataloader):
            image = batch["image"]
            state = batch["state"].unsqueeze(1)
            action = batch["action"]
            result = model.train(image, state, action)
            for key, value in result.items():
                logger.add_scalar(f"train/{key}", value, step_num)

            step_num += 1

        # include any evaluation logic here
        eval_result = evaluate_model(model, "gym_pusht/PushT-v0", args.render_eval)
        for key, value in eval_result.items():
            logger.add_scalar(f"eval/{key}", value, step_num)

        # evaluate
        print(
            f"Completed epoch {epoch+1}/{args.epochs} with average score: {eval_result['mean_episode_reward']} and average steps {eval_result['mean_episode_steps']}"
        )

        # Save checkpoint periodically
        checkpoint_path = os.path.join(
            args.checkpoint_dir, f"checkpoint_epoch{epoch+1}_step{step_num}.pth"
        )
        model.save_checkpoint(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


if __name__ == "__main__":
    main()
