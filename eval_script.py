# type: ignore[all]

import torch
import gymnasium as gym
from model.act import ACTModel

def load_model(checkpoint_path, device):
    """Load the ACTModel from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ACTModel(
        input_image_size=checkpoint['input_image_size'],
        input_state_size=checkpoint['input_state_size'],
        action_space=checkpoint['action_space'],
        K=checkpoint['K'],
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate_model(model, env_name, num_episodes=10):
    """Evaluate the model in the specified gym environment."""
    env = gym.make(env_name)
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # Assuming the model takes state as input and outputs actions
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(model._device)
            action = model.select_action(state_tensor).cpu().detach().numpy()
            print(f"Action shape {action.shape}")
            state, reward, done, info = env.step(action)
            env.render()
    env.close()

if __name__ == "__main__":
    checkpoint_path = "path/to/checkpoint.pth"  # Update with the actual path
    env_name = "pusht-v0"  # Update with the actual environment name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, device)
    evaluate_model(model, env_name)