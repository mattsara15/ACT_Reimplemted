# type: ignore[all]

import argparse
import os
import torch
import time

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
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    return p.parse_args()


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

        # evaluate
        print(f"Completed epoch {epoch+1}/{args.epochs}")
        # include any evaluation logic here

        # Save checkpoint periodically
        checkpoint_path = os.path.join(
            args.checkpoint_dir, f"checkpoint_epoch{epoch+1}_step{step_num}.pth"
        )
    
        torch.save(
            {
                "epoch": epoch + 1,
                "step": step_num,
                "model_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved at {checkpoint_path}")
            

if __name__ == "__main__":
    main()
