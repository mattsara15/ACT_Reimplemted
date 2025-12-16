# type: ignore[all]

import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class ACTDataLoader(Dataset):
    def __init__(self, k, use_grayscale, dataset_name, device, train_samples:int):
        self._K = k
        self._use_grayscale = use_grayscale
        self._dataset_name = dataset_name
        self._device = device
        self._train_samples = train_samples

        # Build the chunks
        self._chunks = self._create_chunks()

    def _create_chunks(self):
        """Group data into chunks of size k with the same episode_index."""
        chunks = []
        current_episode = []
        current_episode_index = 0

        # download the dataset
        data = LeRobotDataset(self._dataset_name)

        max_episode_idx = 0
        for entry in data:
            image = entry["observation.image"]
            state = entry["observation.state"]
            action = entry["action"]

            current_episode.append({"image": image, "state": state, "action": action})

            # we've got new episode
            if entry["episode_index"] != current_episode_index:
                # Process the previous episode
                chunks.extend(self._chunk_episode(current_episode))

                # Start a new episode
                current_episode = []
                current_episode_index = entry["episode_index"]

            if current_episode_index > max_episode_idx:
                max_episode_idx = current_episode_index


        # Process the last episode
        chunks.extend(self._chunk_episode(current_episode))
        return chunks

    def _chunk_episode(self, episode):
        """Create chunks of size k from a single episode."""
        result =  [episode[i : i + self._K] for i in range(len(episode) - self._K + 1)]
        return result

    def __len__(self):
        return len(self._chunks)

    def __getitem__(self, idx):
        chunk = self._chunks[idx]
        batch = {
            key: (
                torch.stack([torch.tensor(entry[key]).to(self._device) for entry in chunk])
                if isinstance(chunk[0][key], (list, torch.Tensor)) and key == "action"
                else torch.tensor(chunk[0][key]).to(self._device)
            )
            for key in chunk[0]
        }
        return batch


class ACTValDataLoader(Dataset):
    def __init__(self, use_grayscale, dataset_name, device, test_samples:int):
        self._use_grayscale = use_grayscale
        self._dataset_name = dataset_name
        self._device = device
        self._test_samples = test_samples

        self._episodes = self._create_episodes()
    
    def _create_episodes(self):
        episodes = []
        current_episode_index = 0
        current_episode = []

        # download the dataset
        data = LeRobotDataset(self._dataset_name)

        max_episode_idx = 0
        for entry in data:
            image = entry["observation.image"]
            state = entry["observation.state"]
            action = entry["action"]

            current_episode.append({"image": image, "state": state, "action": action})

            # we've got new episode
            if entry["episode_index"] != current_episode_index:
                # Process the previous episode
                episodes.append(current_episode)

                # Start a new episode
                current_episode = []
                current_episode_index = entry["episode_index"]

            if current_episode_index > max_episode_idx:
                max_episode_idx = current_episode_index

        episodes.append(current_episode)
        return episodes[-self._test_samples:]


    def __len__(self):
        return len(self._episodes)

    def __getitem__(self, idx):
        episode = self._episodes[idx]
        batch = {
            key: (
                torch.stack([torch.tensor(entry[key]).to(self._device) for entry in episode])
            )
            for key in episode[0]
        }
        return batch