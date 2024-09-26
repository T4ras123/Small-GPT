import torch
from gpt import SmallGPT, decode # Import the model from model.py

# Instantiate the model
m = SmallGPT()

# Path to the saved model
model_path = "model.pth"

# Load the model state dictionary
m.load_state_dict(torch.load(model_path))

# Set model to evaluation mode if needed (for inference)
m.eval()


context = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=700)[0].tolist()))