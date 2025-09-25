import torch
import numpy as np
import os
from datetime import datetime

from random_selector_model import TimeSeriesTransformer

root = '/users/vmli3/SORS-2025/'
log_file = root + 'random_selector_log1.txt'
model_path = '/scratch/vmli3/SORS-2025/random_selector_model1.pth'

# root = '/Users/victorli/Documents/GitHub/SORS-2025/TST/'
# log_file = root + 'log.txt'
# model_path = '/Users/victorli/Documents/GitHub/SORS-2025/TST/model.pth'

data_path = root + 'data/stbernard_data_filled.npy'
static_path = root + 'data/stbernard_static.npy'
null_stations_path = root + 'data/stbernard_null_set.npy'

def log(message):
    with open(log_file, 'a') as f:
        f.write(message + '\n')
        print(message)

if os.path.exists(log_file):
    os.remove(log_file)

with open(log_file, 'w') as f:
    f.write(f'Log started at {datetime.now()}\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f'Torch device: {device}')
torch.set_default_dtype(torch.float32)

data = np.load(data_path)
log(f"Data shape: {data.shape}")

static = np.load(static_path)
log(f"Static shape: {static.shape}")

null_stations = np.load(null_stations_path)
log(f"Null stations shape: {null_stations.shape}")
null_stations_set = set((int(a), int(b)) for a, b in null_stations)

split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]
log(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

train_tensor  = torch.from_numpy(train_data).float().to(device)   # (N_train, S, L, F)
test_tensor   = torch.from_numpy(test_data).float().to(device)    # (N_test,  S, L, F)
static_tensor = torch.from_numpy(static).float().to(device)  # (S, static_feats)

train_tensor = torch.nan_to_num(train_tensor, nan=0.0, posinf=0.0, neginf=0.0)
test_tensor = torch.nan_to_num(test_tensor, nan=0.0, posinf=0.0, neginf=0.0)
static_tensor = torch.nan_to_num(static_tensor, nan=0.0, posinf=0.0, neginf=0.0)


d_model = 128
num_heads = 8
num_variables = 4
num_static = 2
seq_len = 24
num_encoder_layers = 6
num_decoder_layers = 6
d_ff = 4*d_model
budget = 12
top_k = 4
dropout = 0.1
eps = torch.finfo(torch.float32).eps

num_epochs = 400
learning_rate = 1e-3

model = TimeSeriesTransformer(
    d_model=d_model,
    num_variables=num_variables,
    num_static=num_static,
    seq_len=seq_len,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    num_sensors=budget,
    top_k=top_k,
    dropout=dropout,
    eps=eps
)

log(f"Model parameters: d_model={d_model}, num_heads={num_heads}, num_variables={num_variables}, "
    f"num_static={num_static}, seq_len={seq_len}, num_encoder_layers={num_encoder_layers}, "
    f"num_decoder_layers={num_decoder_layers}, d_ff={d_ff}, budget={budget}, top_k={top_k}, dropout={dropout}")

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    log(f"Using {torch.cuda.device_count()} {torch.cuda.get_device_name(0)} GPUs")
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
model.to(device)

# if trial != 0:
# model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

criterion = torch.nn.L1Loss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=eps)

log(f"Training parameters: num_epochs={num_epochs}, learning_rate={learning_rate}, criterion={criterion}, optimizer={optimizer}")
MIN_LOSS = 100
MIN_EPOCH = 0
for epoch in range(num_epochs):
    model.train()
    start_time = datetime.now()
    epoch_loss = 0.0

    permute = torch.randperm(static_tensor.size(0), device=device)
    selected_indices = permute[:budget].tolist()
    unsensed_indices = permute[budget:].tolist()

    for i in range(train_tensor.size(0)):
        X = train_tensor[i, selected_indices,   :, :]

        X_static = static_tensor[selected_indices, :]

        X_unsensed_static = static_tensor[unsensed_indices, :]

        optimizer.zero_grad()
        output, indices = model(X.to(device), X_static.to(device), X_unsensed_static.to(device))

        selected_indices_tensor = torch.tensor(selected_indices, device=device)
        indices = selected_indices_tensor[indices]

        target = train_tensor[i, unsensed_indices, :, 0]
        loss = criterion(output, target)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

        selected_indices_set = set()
        unsensed_indices_set = set(range(len(static)))

        for idx in indices:
            idx = idx.item()
            if (i, idx) not in null_stations_set:
                selected_indices_set.add(idx)
                unsensed_indices_set.discard(idx)

        for idx in list(unsensed_indices_set):
            if (i, idx) in null_stations_set:
                unsensed_indices_set.discard(idx)

        if len(unsensed_indices_set) < budget - len(selected_indices_set):
            raise Exception(f"Error: fewer than {budget} selected indices, found {len(unsensed_indices_set)} indices remaining and need {budget-len(selected_indices_set)} indices.")
        
        while len(selected_indices_set) < budget:
            idx = np.random.choice(list(unsensed_indices_set))
            selected_indices_set.add(idx)
            unsensed_indices_set.discard(idx)

        selected_indices = list(selected_indices_set)
        unsensed_indices = list(unsensed_indices_set)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        permute = torch.randperm(static_tensor.size(0), device=device)
        selected_indices = permute[:budget].tolist()
        unsensed_indices = permute[budget:].tolist()

        for i in range(test_tensor.size(0)):
            X = test_tensor[i, selected_indices,   :, :]

            X_static = static_tensor[selected_indices, :]

            X_unsensed_static = static_tensor[unsensed_indices, :]

            output, indices = model(X.to(device), X_static.to(device), X_unsensed_static.to(device))

            selected_indices_tensor = torch.tensor(selected_indices, device=device)
            indices = selected_indices_tensor[indices]

            target = test_tensor[i, unsensed_indices, :, 0]

            loss = criterion(output, target)
            test_loss += loss.item()

            selected_indices_set = set()
            unsensed_indices_set = set(range(len(static)))
            for idx in indices:
                idx = idx.item()
                if (i + split_idx, idx) not in null_stations_set:
                    selected_indices_set.add(idx)
                    unsensed_indices_set.discard(idx)

            for idx in list(unsensed_indices_set):
                if (i + split_idx, idx) in null_stations_set:
                    unsensed_indices_set.discard(idx)

            if len(unsensed_indices_set) < budget - len(selected_indices_set):
                raise Exception(f"Error: fewer than {budget} selected indices, found {len(unsensed_indices_set)} indices remaining and need {budget-len(selected_indices_set)} indices.")

            while len(selected_indices_set) < budget:
                idx = np.random.choice(list(unsensed_indices_set))
                selected_indices_set.add(idx)
                unsensed_indices_set.discard(idx)

            selected_indices = list(selected_indices_set)
            unsensed_indices = list(unsensed_indices_set)

    average_test_loss = test_loss / test_tensor.size(0)

    if average_test_loss < MIN_LOSS:
        MIN_LOSS = average_test_loss
        MIN_EPOCH = epoch+1

    log(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss/train_tensor.size(0):.8f}, Time: {datetime.now() - start_time}, Test Loss: {average_test_loss:.8f}')

log(f'Min Loss: {MIN_LOSS} at Epoch: {MIN_EPOCH}')
torch.save(model.state_dict(), model_path)


