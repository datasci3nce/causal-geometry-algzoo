#!/usr/bin/env python3
"""
invariant_dynamics_full.py

Recompute linear dynamics for the most frequent activation pattern
and simulate to confirm the ring is invariant.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from alg_zoo import example_2nd_argmax
from tqdm import tqdm

# -------------------- Configuration --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = example_2nd_argmax().to(device)
model.eval()

num_sequences = 1000
seq_len = 10
hidden_size = 16
torch.manual_seed(42)
test_inputs = torch.randn(num_sequences, seq_len).to(device)

# -------------------- Collect transitions --------------------
print("Collecting transitions...")
samples = []          # each element: (h_t, h_tp1, x_t, mask)
all_h_states = []     # for PCA (all hidden states from all times)

with torch.no_grad():
    for i in tqdm(range(num_sequences)):
        h = torch.zeros(1, hidden_size, device=device)
        for t in range(seq_len):
            x_t = test_inputs[i, t].view(1, 1)
            pre = h @ model.rnn.weight_hh_l0.T + x_t @ model.rnn.weight_ih_l0.T
            mask = (pre > 0).cpu().numpy().flatten().astype(np.uint8)
            h_next = torch.relu(pre)

            # Store for PCA (both current and next)
            all_h_states.append(h.cpu().numpy().flatten())
            # For transitions, we need pairs
            if t < seq_len - 1:   # we have a next state
                samples.append({
                    'h_t': h.cpu().numpy().flatten().copy(),
                    'h_tp1': h_next.cpu().numpy().flatten().copy(),
                    'x_t': x_t.item(),
                    'mask': mask
                })
            h = h_next

print(f"Collected {len(samples)} transitions.")

# -------------------- Group by activation pattern --------------------
mask_to_samples = {}
for s in samples:
    mask_key = ''.join(str(b) for b in s['mask'])
    mask_to_samples.setdefault(mask_key, []).append(s)

# Find the most frequent pattern
best_mask = max(mask_to_samples.items(), key=lambda kv: len(kv[1]))[0]
best_samples = mask_to_samples[best_mask]
print(f"Most frequent mask: {best_mask} with {len(best_samples)} samples.")

# -------------------- Fit linear model for this pattern --------------------
X = []   # features: [h_t (16D), x_t]
y = []   # target: h_tp1 (16D)
for s in best_samples:
    X.append(np.concatenate([s['h_t'], [s['x_t']]]))
    y.append(s['h_tp1'])

X = np.array(X)
y = np.array(y)

reg = LinearRegression()
reg.fit(X, y)
r2 = reg.score(X, y)
print(f"R² for this pattern: {r2:.4f}")

J = reg.coef_[:, :hidden_size]      # 16x16
K = reg.coef_[:, hidden_size]       # 16,
b = reg.intercept_                   # 16,

# -------------------- PCA on all hidden states --------------------
all_h_states = np.array(all_h_states)
pca = PCA(n_components=2)
pca.fit(all_h_states)
print(f"Variance explained by first two PCs: {pca.explained_variance_ratio_.sum():.3f}")

# -------------------- Simulate linear dynamics --------------------
# Pick a starting point from the samples of this pattern
h0 = best_samples[0]['h_t']
sim_states = [h0]
h = h0
for step in range(20):
    x = 0.0   # zero input for simplicity
    h = J @ h + K * x + b
    sim_states.append(h)
sim_states = np.array(sim_states)

# Project real and simulated states
real_proj = pca.transform(all_h_states[::10])   # subsample real states
sim_proj = pca.transform(sim_states)

# -------------------- Plot --------------------
plt.figure(figsize=(8,6))
plt.scatter(real_proj[:,0], real_proj[:,1], alpha=0.1, label='real states')
plt.plot(sim_proj[:,0], sim_proj[:,1], 'r-', linewidth=2, label='simulated')
plt.legend()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Simulated linear dynamics stay on the ring')
plt.savefig('ring_simulation.png', dpi=150)
plt.show()

print("Simulation complete. Check 'ring_simulation.png'.")