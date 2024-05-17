import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# hyperparameters
# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train data, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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

class HEAD(nn.Module):
    """ one head of self attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, device=device, bias=False)
        self.query = nn.Linear(n_embd, head_size, device=device, bias=False)
        self.value = nn.Linear(n_embd, head_size, device=device, bias=False)
        # to save a tensor (or something not in nn.module) in the state dict when saving models
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape
        # this will be applied to all batches [0, ... , 63]
        key = self.key(x)
        query = self.query(x)
        # compute attention scores or called "affinities"
        # weights = query @ key.transpose(-2, -1) * C**-(1/2)
        weights = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # masking the future letters to the tensors that are not allowed
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)  # B, T, T
        weights = self.dropout(weights)
        # perform weighted aggregation
        out = weights @ self.value(x)

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, number_of_heads, block_size):

        super().__init__()
        self.heads = nn.ModuleList([HEAD(block_size) for _ in range(number_of_heads)])
        self.proj = nn.Linear(block_size * number_of_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concat all heads outputs in one array : 4 heads of 8 dimensions = 32 channels
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ simple feed forward layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embeddings, number_of_heads):
        super().__init__()
        head_size = n_embeddings // number_of_heads
        self.sa = MultiHeadAttention(number_of_heads, head_size)
        self.ffwd = FeedForward(n_embeddings)
        self.layer1 = nn.LayerNorm(n_embeddings)
        self.layer2 = nn.LayerNorm(n_embeddings)

    def forward(self, x):
        # skip connections : x original data
        x = x + self.sa(self.layer1(x))
        x = x + self.ffwd(self.layer2(x))
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        self.position_embedding_table = nn.Embedding(block_size,  embedding_dim=n_embd)
        """ we only call block now since we define what's below in this class """
        # lets for this the number of embeddings is 32. so for multi-attention head we need 4 heads of 8 dimensions
        # self.sa_heads = MultiHeadAttention(4, n_embd // 4)
        # seeing the paper of attention is all you need when we apply the MultiHeadAttention only, we are not letting
        # the network enough time to use the embedding information "nn think on that data" collected from key query and
        # value exchange between embeddings, so we add another feed forward layer
        # self.ffwd = FeedForward(n_embd)
        """
        self.blocks = nn.Sequential(
            Block(n_embd, number_of_heads=n_head),
            Block(n_embd, number_of_heads=n_head),
            Block(n_embd, number_of_heads=n_head),
            nn.LayerNorm(n_embd)
        )
        """
        self.blocks = nn.Sequential(*[Block(n_embd, number_of_heads=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        # we can call this a decoder
        self.lm_head = nn.Linear(n_embd, vocab_size)  # (C, vocab_size)
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

        batch, time = idx.shape
        # idx and targets are both (B,T) tensor of integers this will find the right index in the emedding table to
        # get the corresponding logit representation for each character : 1 char -> 32 tensor
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        # embed the position of each character in the text (to not lose Time dimension)
        pos_emb = self.position_embedding_table(torch.arange(time, device=device))  # (T, C)
        # add the two embeddings into each other
        x = tok_emb + pos_emb  # (B,T,C) + (T, C) --> (B, T, C)
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cropped = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cropped)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


"""
xb, yb = get_batch('train')
model = BigramLanguageModel(vocab_size).to(device=device)
model(xb)

### self attention implementation
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

# single head performs attention
head_size = 16
# so this basically it launches key in every single node in each position emmits two vectors :
# query vector roughly speaking is saying what i'm looking for
# key vector roughly speaking saying what do i contain
# to get affinities between each token in a sequence is to do a dot product between them
# to see which embedding relating to another
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

# we multiply by square root so the softmax won't bias towards very high or low values to control the variance
# specially during initialization : try and notice how values of softmax gets higher around high values
# example_tensor = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])
# F.softmax(example_tensor * 8, dim=-1)

weights = query(x) @ key(x).transpose(-2, -1) * head_size**-(1/2)  # B, T, T

# and then we use this to mask out the history (so tokens don't look onto the future as a natural speaking language)
# PS : in pytorch you make multiplication directly without seeing the batch because it's implicit, and he knows how
# to handle it like the creating a mask of T x T and applying it to a B x T x T
tril = torch.tril(torch.ones(T, T))
weights = weights.masked_fill(tril == 0, float('-inf'))
weights = F.softmax(weights, dim=-1)  # B, T, T
x_bag_of_words_3 = weights @ value(x).shape

# x_bag_of_words_3[0] --> is a T, C : time , channels(embedding). so in this one sequence of 8 letters and each
# letter has 32 vector embedding representing the word which has attention about the past.
"""

####### """ training model """ ###########
# define model
model = BigramLanguageModel(vocab_size)
model = model.to(device=device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iteration in tqdm(range(max_iters)):  # increase number of steps for good results...

    # every once in a while evaluate the loss on train and val sets
    if iteration % eval_interval == 0 or iteration == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# example of text generation
print("Text after training")
print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=500)[0].tolist()))
