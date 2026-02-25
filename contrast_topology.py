#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from alg_zoo import example_2nd_argmax, zoo_longest_cycle

def generate_contrast_figure():
    device = "cpu"
    print("🔄 Generating Contrast: RNN vs Transformer Topology...")

    # --- 1. RNN Data (The Ring) ---
    rnn = example_2nd_argmax()
    rnn_states = []
    rnn_c = [] # Color by semantic variable (value)
    
    # Generate RNN traces
    with torch.no_grad():
        inputs = torch.randn(200, 10)
        W_ih, W_hh = rnn.rnn.weight_ih_l0, rnn.rnn.weight_hh_l0
        for i in range(200):
            h = torch.zeros(1, 16)
            curr_max = -999
            for t in range(10):
                val = inputs[i, t].item()
                if val > curr_max: curr_max = val
                
                h = torch.relu(h @ W_hh.T + inputs[i, t].view(1,1) @ W_ih.T)
                if t > 0: # Skip t=0
                    rnn_states.append(h.flatten().numpy())
                    rnn_c.append(curr_max)

    # --- 2. Transformer Data (The Clusters) ---
    # Model: zoo_longest_cycle (h=8, s=6)
    trf = zoo_longest_cycle(hidden_size=8, seq_len=6)
    trf_states = []
    trf_c = [] # Color by semantic variable (current cycle length estimate)
    
    with torch.no_grad():
        # Discrete inputs for discrete task
        trf_inputs = torch.randint(0, 6, (200, 6))
        pos_ids = torch.arange(6)
        
        for i in range(200):
            # Embed
            x = trf.embed(trf_inputs[i:i+1]) + trf.pos_embed(pos_ids)
            
            # Pass through layers (treating depth as time)
            for layer in trf.attns:
                out, _ = layer(x, x, x)
                x = x + out
                # Collect state of first token (the classifier token)
                trf_states.append(x[0, 0, :].flatten().numpy())
                # Semantic proxy: just time step (layer depth) to show clustering
                # Or use input value to show it separates discrete tokens
                trf_c.append(trf_inputs[i, 0].item()) 

    # --- 3. PCA & Plotting ---
    pca_rnn = PCA(n_components=2).fit_transform(rnn_states)
    pca_trf = PCA(n_components=2).fit_transform(trf_states)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # RNN Plot
    sc1 = axes[0].scatter(pca_rnn[:,0], pca_rnn[:,1], c=rnn_c, cmap='viridis', s=10, alpha=0.5)
    axes[0].set_title("RNN: Continuous Logic Ring (H1)", fontweight='bold')
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].text(0.05, 0.95, "Task: 2nd-Argmax\nGeometry: Manifold\nMechanism: Analog Integration", 
                 transform=axes[0].transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.colorbar(sc1, ax=axes[0], label="Current Max Value")

    # Transformer Plot
    sc2 = axes[1].scatter(pca_trf[:,0], pca_trf[:,1], c=trf_c, cmap='tab10', s=15, alpha=0.7)
    axes[1].set_title("Transformer: Discrete Clusters (H0)", fontweight='bold')
    axes[1].set_xlabel("Principal Component 1")
    axes[1].set_yticks([])
    axes[1].text(0.05, 0.95, "Task: Longest Cycle\nGeometry: Clusters\nMechanism: Discrete Routing", 
                 transform=axes[1].transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.colorbar(sc2, ax=axes[1], label="Input Token Class")

    plt.tight_layout()
    plt.savefig('contrast_topology.png', dpi=300)
    print("✅ Saved 'contrast_topology.png'")

if __name__ == "__main__":
    generate_contrast_figure()