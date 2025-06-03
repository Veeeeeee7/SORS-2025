import torch
import numpy as np
import os
from datetime import datetime

from model import TimeSeriesTransformer

def log(message):
    with open('/Users/victorli/Documents/GitHub/SORS-2025/TST/log.txt', 'a') as f:
        f.write(message + '\n')
        print(message)

if os.path.exists('/Users/victorli/Documents/GitHub/SORS-2025/TST/log.txt'):
    os.remove('/Users/victorli/Documents/GitHub/SORS-2025/TST/log.txt')

with open('/Users/victorli/Documents/GitHub/SORS-2025/TST/log.txt', 'w') as f:
    f.write(f'Log started at {datetime.now()}\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f'Torch device: {device}')
torch.set_default_dtype(torch.float32)

data = np.load('/Users/victorli/Documents/GitHub/SORS-2025/TST/data/data_windowed.npy')
data = data[:10]
log(f"Data shape: {data.shape}")

static = np.load('/Users/victorli/Documents/GitHub/SORS-2025/TST/data/static.npy')
log(f"Static shape: {static.shape}")

null_stations = np.load('/Users/victorli/Documents/GitHub/SORS-2025/TST/data/null_stations.npy')
log(f"Null stations shape: {null_stations.shape}")
null_stations_set = set(map(tuple, null_stations))

split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]
log(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

d_model = 128
num_heads = 8
num_variables = 6
num_static = 2
seq_len = 24
num_encoder_layers = 6
num_decoder_layers = 6
d_ff = 4*d_model
budget = 100
top_k = 50
dropout = 0.1

num_epochs = 2
batch_size = 8
learning_rate = 0.001

model = TimeSeriesTransformer(
    d_model=d_model,
    num_variables=num_variables,
    num_static=num_static,
    seq_len=seq_len,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    top_k=top_k,
    dropout=dropout
)

log(f"Model parameters: d_model={d_model}, num_heads={num_heads}, num_variables={num_variables}, "
    f"num_static={num_static}, seq_len={seq_len}, num_encoder_layers={num_encoder_layers}, "
    f"num_decoder_layers={num_decoder_layers}, d_ff={d_ff}, top_k={top_k}, dropout={dropout}")

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    log(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
model.to(device)

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    permute = torch.randperm(data.shape[1])
    sensed_indices = permute[:budget]
    unsensed_indices = permute[budget:]

    for i in range(0, len(train_data)):
        X = torch.tensor(train_data[i, :, :, :])
        X = X[sensed_indices, :, :]

        X_static = torch.tensor(static)
        X_static = X_static[sensed_indices, :]

        X_unsensed_static = torch.tensor(static)
        X_unsensed_static = X_unsensed_static[unsensed_indices, :]

        optimizer.zero_grad()
        output, indices = model(X.to(device), X_static.to(device), X_unsensed_static.to(device))

        target = torch.tensor(train_data[i, :, :, :])[unsensed_indices, :, 0]

        loss = criterion(output, target)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

        sensed_indices = set()
        unsensed_indices = set(range(len(static)))
        for idx in indices:
            if (i, idx) not in null_stations_set:
                sensed_indices.add(idx)
                unsensed_indices.discard(idx)

        for idx in list(unsensed_indices):
            if (i, idx) in null_stations_set:
                unsensed_indices.discard(idx)

        if len(unsensed_indices) < budget - len(sensed_indices):
            raise Exception(f"Error: fewer than {budget} sensed indices, found {len(unsensed_indices)} indices remaining and need {budget-len(sensed_indices)} indices.")
        
        while len(sensed_indices) < budget:
            idx = np.random.choice(list(unsensed_indices))
            sensed_indices.add(idx)
            unsensed_indices.discard(idx)

        sensed_indices = list(sensed_indices)
        unsensed_indices = list(unsensed_indices)

    log(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_data):.8f}')