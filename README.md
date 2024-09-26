# Small GPT

Reproduced a transformer architecture with some modifications following [Vaswani et al. 2017](https://arxiv.org/pdf/1706.03762)


## Training reslts 

- #### Validation loss - 1.5
- #### Example text:
```
He cannot have him as kind; we all Edward,
But I will any moe to a husband
In here, making instruments on their harm;
And therefore it off; for 'tis but by cold,
He is engrace, and designs, as the sea
As doth finger grave in Sicilia childishment;
It shall be so, to retire thee day,
For if exy were seen the proof, with a mind,
As I have ever shall be regal as one.

ESCALUS:
Say you shall I be the
common secucord and by the like a figure?
Farewell exhale ashame.

ESCALUS:
I am, we say!

Justice:
It beseech you? weary true.

CANUS:
He says within the tribunes? Our order means
Fled to make duty?

FROTH:
Marry, that who comes highness chances to bay;
Unless his force and that encompass he
Of vict
```

## My model

### 1. Implemented multihead attention

![alt text](Media/image.png)

```python
class MultiHeadAttention(nn.Module):
    
        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range (num_heads)])
            self.proj = nn.Linear(n_embed, n_embed)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
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
        
        wei = wei.masked_fill(self.tril[:T, :T]==0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)

        out = wei @ v
        
        return out 
```

### 2. Feed forward is a simple: Linear -> ReLU -> Linear -> Dropout

```python
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed), 
            nn.ReLU(),
            nn.Linear(4*  n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
```

- Diviation from the paper is here in a Dropout layer that prevents overfitting the data

### 3. Add & Norm Block

![alt text](Media/image-4.png)

```python
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        
        head_size = n_embed//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

        
    def forward(self, x):
        x += self.sa(self.ln1(x)) # Layer norm is applied before attention
        x += self.ffwd(self.ln2(x)) # And before MLP
        return x
```

- In the original paper Add & Norm layer is applied after Attention or MLP, but I applied it before following the latest practices  
