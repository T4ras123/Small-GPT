import torch 
import torch.nn as nn 
from torch.nn import functional as F
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(163173)
print(torch.cuda.is_available())

#----------------------------------------------------------

n_embed = 384
n_head = 6 
n_layers = 6
learning_rate = 3e-4
dropout = 0.2
batch_size = 64
block_size = 256
eval_iters = 200
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 10000


#-----------------------------------------------------------

with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    
chars = sorted(list(set(text)))
vocab_size = len(chars)

itos = {i: char for i, char in enumerate(chars)}
stoi = {char: i for i, char in enumerate(chars)}
encode = lambda x: [stoi[char] for char in list(x)]
decode = lambda x: ''.join([itos[num] for num in list(x)])

data = torch.tensor(encode(text), dtype=torch.long).to(device)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([data[i:block_size + i] for i in ix])
    y = torch.stack([data[i + 1:block_size + i + 1] for i in ix])
    return x, y 

#-------------------------------------------------------------


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)

        out = wei @ v
        
        return out 
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), 
            nn.ReLU(inplace=False),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Regular addition
        x = x + self.ffwd(self.ln2(x))  # Regular addition
        return x

class SmallGPT(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.l_head = nn.Linear(n_embed, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layers)])
        
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.l_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__=='__main__':
    
    model = SmallGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        xb, yb = get_batch('train')

        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=700)[0].tolist()))

    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)

    print(f"Model saved at {model_path}")