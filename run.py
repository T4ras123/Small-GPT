import torch
from gpt import SmallGPT, decode # Import the model from model.py

m = SmallGPT()

model_path = "model.pth"

# Load the model state dictionary
m.load_state_dict(torch.load(model_path))

m.eval()


context = torch.zeros((1, 1), dtype=torch.long)

max_new_tokens=700
print(decode(m.generate(context, max_new_tokens)[0].tolist()))