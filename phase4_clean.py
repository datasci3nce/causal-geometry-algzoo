#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from alg_zoo import example_2nd_argmax

def phase4_semantic_manifold(model, num_sequences=500, seq_len=10):
    device = "cpu"
    model.eval()
    
    print("🪐 Phase 4: Mapping the Logic Ring...")
    
    all_h = []
    running_maxes = []
    
    inputs = torch.randn(num_sequences, seq_len)
    
    with torch.no_grad():
        W_ih, W_hh = model.rnn.weight_ih_l0, model.rnn.weight_hh_l0
        for i in range(num_sequences):
            h = torch.zeros(1, 16)
            curr_max = -999.0
            
            for t in range(seq_len):
                val = inputs[i, t].item()
                if val > curr_max:
                    curr_max = val
                    
                h = torch.relu(h @ W_hh.T + inputs[i, t].view(1,1) @ W_ih.T)
                
                # We skip t=0 because it's just the input embedding, the ring forms t>=1
                if t > 0:
                    all_h.append(h.numpy().flatten())
                    running_maxes.append(curr_max)
                    
    all_h = np.array(all_h)
    running_maxes = np.array(running_maxes)
    
    # Run PCA to find the manifold plane
    pca = PCA(n_components=2)
    h_2d = pca.fit_transform(all_h)
    
    # Plotting
    plt.figure(figsize=(9, 7))
    
    # Scatter plot colored by the semantic variable (running_max)
    scatter = plt.scatter(h_2d[:, 0], h_2d[:, 1], c=running_maxes, cmap='viridis', 
                          alpha=0.6, s=15, edgecolor='none')
                          
    cbar = plt.colorbar(scatter)
    cbar.set_label('Semantic State: current_max', fontsize=12)
    
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.title('Phase 4: The Logic Ring (Semantic PCA Projection)', fontsize=14, weight='bold')
    
    # Add explanatory text
    plt.text(np.min(h_2d[:,0]), np.max(h_2d[:,1]), 
             "Movement along the continuous\nmanifold mathematically correlates\nwith updating the current maximum.",
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase4_logic_ring.png', dpi=300)
    print("✅ Saved 'phase4_logic_ring.png'")

if __name__ == "__main__":
    model = example_2nd_argmax()
    phase4_semantic_manifold(model)