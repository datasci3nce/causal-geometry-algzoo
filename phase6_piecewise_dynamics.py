#!/usr/bin/env python3
"""
phase6_piecewise_dynamics.py

This script extends the dynamics analysis to account for:
- Event types (NEW_MAX, NEW_SECOND, IRRELEVANT)
- Activation patterns (which ReLUs are active)

The goal is to show that the update Δθ is nearly zero for irrelevant inputs,
and within each linear region (defined by activation pattern), the dynamics
are affine: Δθ = a·θ + b·x + c with high R².

Run after phase6_dynamics.py (which established global linearity fails).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from alg_zoo import example_2nd_argmax
from tqdm import tqdm
import pickle
import os

# -------------------- Configuration --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = example_2nd_argmax().to(device)
model.eval()

num_sequences = 1000
seq_len = 10
hidden_size = 16

torch.manual_seed(42)
test_inputs = torch.randn(num_sequences, seq_len).to(device)

# We'll store hidden states and metadata
all_h = []            # hidden state trajectories (N,10,16)
all_theta = []        # ring coordinate (angle) after PCA
all_x = []            # inputs (N,10)
all_events = []       # event type per step (N,10) as strings
all_masks = []        # activation pattern per step (N,10,16) boolean
all_delta_theta = []  # Δθ per transition (N,9)

# First pass: collect hidden states to fit PCA
print("Collecting hidden states for PCA...")
with torch.no_grad():
    for i in tqdm(range(num_sequences)):
        x_seq = test_inputs[i]
        h = torch.zeros(1, hidden_size, device=device)
        traj = []
        for t in range(seq_len):
            x_t = x_seq[t].view(1,1)
            # Pre-activation (before ReLU) to get activation pattern
            pre = h @ model.rnn.weight_hh_l0.T + x_t @ model.rnn.weight_ih_l0.T
            h = torch.relu(pre)
            traj.append(h.cpu().numpy().flatten())
            # Store mask later after we have PCA? We'll store after second pass.
        all_h.append(np.array(traj))   # shape (10,16)

all_h = np.array(all_h)                 # (N,10,16)
states_flat = all_h.reshape(-1, hidden_size)

# Fit PCA on all states to define the ring plane
pca = PCA(n_components=2)
states_2d = pca.fit_transform(states_flat)
print(f"Variance explained by first two PCs: {pca.explained_variance_ratio_.sum():.3f}")
states_2d = states_2d.reshape(num_sequences, seq_len, 2)

# Compute ring angle θ for each step
theta = np.arctan2(states_2d[..., 1], states_2d[..., 0])   # (N,10)
theta_unwrapped = np.unwrap(theta, axis=1)

# Now second pass: collect activation masks and compute Δθ with event tagging
print("\nSecond pass: collecting activation patterns and events...")
all_masks = []   # will store (N,10,16) boolean
all_events = []  # (N,10) strings
all_x = []       # (N,10)
all_theta = theta_unwrapped   # use unwrapped angles

with torch.no_grad():
    for i in tqdm(range(num_sequences)):
        x_seq = test_inputs[i].cpu().numpy()
        h = torch.zeros(1, hidden_size, device=device)
        masks_seq = []
        events_seq = []
        # running max and second max
        current_max = -float('inf')
        current_second = -float('inf')
        for t in range(seq_len):
            x_t = torch.tensor([[x_seq[t]]], device=device)   # shape (1,1)
            pre = h @ model.rnn.weight_hh_l0.T + x_t @ model.rnn.weight_ih_l0.T
            mask = (pre > 0).cpu().numpy().flatten()          # boolean
            masks_seq.append(mask)
            h = torch.relu(pre)   # update hidden state

            # Update running max/second max
            if x_seq[t] > current_max:
                current_second = current_max
                current_max = x_seq[t]
                event = 'NEW_MAX'
            elif x_seq[t] > current_second:
                current_second = x_seq[t]
                event = 'NEW_SECOND'
            else:
                event = 'IRRELEVANT'
            events_seq.append(event)

        all_masks.append(np.array(masks_seq))   # shape (10,16)
        all_events.append(events_seq)
        all_x.append(x_seq)

all_masks = np.array(all_masks)          # (N,10,16)
all_events = np.array(all_events)        # (N,10)
all_x = np.array(all_x)                  # (N,10)

# Compute Δθ for transitions (t=0..8)
delta_theta = np.diff(all_theta, axis=1)   # (N,9)
# For each transition, we need the pre‑transition state: θ_t, x_t, event_t, mask_t
# We'll create a list of samples, each with (θ_t, x_t, event_t, mask_t, Δθ)
samples = []   # list of dicts or arrays
for n in range(num_sequences):
    for t in range(seq_len - 1):
        samples.append({
            'theta': all_theta[n, t],
            'x': all_x[n, t],
            'event': all_events[n, t],
            'mask': all_masks[n, t].copy(),
            'delta': delta_theta[n, t]
        })

print(f"Total samples: {len(samples)}")

# -------------------- Analysis 1: Δθ by event type --------------------
event_types = ['NEW_MAX', 'NEW_SECOND', 'IRRELEVANT']
colors = {'NEW_MAX': 'red', 'NEW_SECOND': 'orange', 'IRRELEVANT': 'blue'}

plt.figure(figsize=(12,4))
for i, event in enumerate(event_types):
    plt.subplot(1,3,i+1)
    xs = [s['x'] for s in samples if s['event'] == event]
    ds = [s['delta'] for s in samples if s['event'] == event]
    plt.scatter(xs, ds, alpha=0.3, c=colors[event])
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('x_t')
    plt.ylabel('Δθ')
    plt.title(f'Event: {event} (N={len(ds)})')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dynamics_by_event.png', dpi=150)
plt.show()

# -------------------- Analysis 2: Group by activation pattern --------------------
# We'll hash each mask to a string for grouping
mask_to_samples = {}
for s in samples:
    mask_key = ''.join(str(int(b)) for b in s['mask'])   # e.g., "1010..."
    mask_to_samples.setdefault(mask_key, []).append(s)

print(f"\nNumber of distinct activation patterns: {len(mask_to_samples)}")

# For each pattern with enough samples, fit linear model Δθ = a·θ + b·x + c
pattern_results = []
min_samples = 30   # minimum samples to attempt a fit

for mask_key, samp_list in mask_to_samples.items():
    if len(samp_list) < min_samples:
        continue
    X = np.array([[s['theta'], s['x']] for s in samp_list])
    y = np.array([s['delta'] for s in samp_list])
    reg = LinearRegression()
    reg.fit(X, y)
    r2 = reg.score(X, y)
    pattern_results.append({
        'mask': mask_key,
        'count': len(samp_list),
        'coef_theta': reg.coef_[0],
        'coef_x': reg.coef_[1],
        'intercept': reg.intercept_,
        'r2': r2
    })

# Sort by R²
pattern_results.sort(key=lambda x: x['r2'], reverse=True)

print("\nActivation patterns with high R² (piecewise linear fits):")
for pr in pattern_results[:10]:   # top 10
    print(f"Mask {pr['mask']} (n={pr['count']}): R² = {pr['r2']:.3f}, "
          f"Δθ = {pr['coef_theta']:.3f}·θ + {pr['coef_x']:.3f}·x + {pr['intercept']:.3f}")

# If any pattern has R² > 0.8, we have discovered piecewise linear dynamics.
high_r2 = [pr for pr in pattern_results if pr['r2'] > 0.8]
if high_r2:
    print(f"\n✅ Found {len(high_r2)} activation patterns with R² > 0.8. "
          "Piecewise linear dynamics confirmed.")
else:
    print("\n⚠️ No pattern reached R² > 0.8. Maybe need more samples or different coordinate.")

# -------------------- Optional: Visualize one good pattern --------------------
if high_r2:
    best = high_r2[0]
    mask_key = best['mask']
    samp_list = mask_to_samples[mask_key]
    X = np.array([[s['theta'], s['x']] for s in samp_list])
    y = np.array([s['delta'] for s in samp_list])
    y_pred = LinearRegression().fit(X, y).predict(X)

    plt.figure(figsize=(5,5))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Δθ')
    plt.ylabel('Predicted Δθ')
    plt.title(f'Linear fit within one activation pattern (R²={best["r2"]:.3f})')
    plt.grid(True)
    plt.savefig('dynamics_best_pattern.png', dpi=150)
    plt.show()

# -------------------- Save results --------------------
results = {
    'samples': samples,   # caution: large
    'pattern_results': pattern_results,
    'event_data': {e: [s for s in samples if s['event'] == e] for e in event_types}
}
with open('phase6_piecewise_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nResults saved to phase6_piecewise_results.pkl")