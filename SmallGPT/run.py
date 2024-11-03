import torch
from gpt import SmallGPT # Import the model from model.py
from tokenizer import GPT4Tokenizer

m = SmallGPT()

model_path = "../model.pth"

# Load the model state dictionary
m.load_state_dict(torch.load(model_path))

m.eval()

tokenizer = GPT4Tokenizer()
tokenizer.load_vocab('vocab.json')


context = torch.zeros((1, 1), dtype=torch.long)

max_new_tokens=700
print(tokenizer.decode(m.generate(context, max_new_tokens)[0].tolist()))
