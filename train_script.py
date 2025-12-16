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
from dataset import ACTDataLoader, ACTValDataLoader
from model.act import ACTModel, ACTModelWrapper
from model.temporal_ensemble_policy import TemporalEnsemblePolicy


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


def evaluate_model_in_gym(
    model, env_name, render_eval: bool, logger, num_episodes=10, max_steps=1000
):
    """Evaluate the model in the specified gym environment."""
    env = None
    print(f"Making eval with render mode: {render_eval}")
    if render_eval:
        env = gym.make(env_name, obs_type="pixels_agent_pos", render_mode="rgb_array")
    else:
        env = gym.make(env_name, obs_type="pixels_agent_pos")
    episode_steps = []
    episode_rewards = []
    actions_x = []
    actions_y = []
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

            state = state.unsqueeze(0)
            image = image.unsqueeze(0)

            # TODO(mattsara) implement temporal ensemble policy
            action = model.select_action(image, state).cpu().detach().numpy()[0]

            if np.isnan(action[0]) or np.isnan(action[1]):
                steps_count += 1
                continue
            actions_x.append(action[0])
            actions_y.append(action[1])
            observation, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            env.render()

            steps_count += 1
            rewards += reward

            if steps_count > max_steps:
                break

        episode_rewards.append(rewards)
        episode_steps.append(steps_count)

    env.close()

    return {
        "mean_episode_reward": np.mean(episode_rewards),
        "mean_episode_steps": np.mean(episode_steps),
        "action_histogram_x": np.asarray(actions_x),
        "action_histogram_y": np.asarray(actions_y),
    }


def evaluate_model_on_demonstrations(
    model,
    val_dataloader,
    args
):
    action_errors = []
    for episode in val_dataloader:
        with torch.no_grad():
            policy = TemporalEnsemblePolicy(
                model,
                model._device,
                chunk_size=args.k,
                num_samples=1,
                temporal_ensemble=True,
            )
            images = episode["image"].squeeze(0)
            states = episode["state"].squeeze(0)
            actions = episode["action"].squeeze(0)

            for step in zip(images, actions, states):
                image = step[0].unsqueeze(0)
                state = step[1].unsqueeze(0)
                action = step[2].unsqueeze(0)
                predicted_action = policy.get_action(image, state)
                action_error = np.linalg.norm(predicted_action - action.cpu().numpy(), ord=1)
                action_errors.append(action_error)

    return {
        "min_action_error": np.min(action_errors),
        "mean_action_error": np.mean(action_errors),
        "max_action_error": np.max(action_errors),
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

    act = ACTModel(
        device=device,
        input_image_size=[input_features["observation.image"].shape],
        input_state_size=input_features["observation.state"].shape[0],
        action_space=output_features["action"].shape,
        K=args.k,
    )
    model = ACTModelWrapper(act).to(device)

    act_dataset = ACTDataLoader(
        k=args.k,
        use_grayscale=args.grayscale,
        dataset_name=args.dataset_name,
        device=args.device,
        train_samples=195,
    )
    dataloader = torch.utils.data.DataLoader(
        act_dataset,
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=True,
    )
    act_val_dataset = ACTValDataLoader(
        use_grayscale=args.grayscale,
        dataset_name=args.dataset_name,
        device=args.device,
        test_samples=10,
    )
    val_dataloader = torch.utils.data.DataLoader(
        act_val_dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
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
                if "histogram" in key:
                    logger.add_histogram(f"train/{key}", value, step_num)
                else:
                    logger.add_scalar(f"train/{key}", value, step_num)

            step_num += 1

        # Evaluate the model
        eval_result = evaluate_model_on_demonstrations(model, val_dataloader, args)

        #eval_result = evaluate_model_in_gym(
        #    model, "gym_pusht/PushT-v0", args.render_eval, logger
        #)
        for key, value in eval_result.items():
            if "histogram" in key:
                logger.add_histogram(f"eval/{key}", value, step_num)
            else:
                logger.add_scalar(f"eval/{key}", value, step_num)
        print(
            f"Completed epoch {epoch+1}/{args.epochs} with eval error: {eval_result['mean_action_error']}"
        )

        # Save checkpoint periodically
        checkpoint_path = os.path.join(
            args.checkpoint_dir, f"checkpoint_epoch{epoch+1}_step{step_num}.pth"
        )
        model.save_checkpoint(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


if __name__ == "__main__":
    main()
