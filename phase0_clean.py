#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax

def phase0_gradient_profile(model, num_samples=2000, seq_len=10):
    device = "cpu"
    model.eval()
    
    print("🔍 Phase 0: Computing Unsupervised Gradient Sensitivity...")
    
    avg_grads = np.zeros(seq_len)
    valid_samples = 0
    
    for _ in range(num_samples):
        # Generate random input
        x = torch.randn(1, seq_len, requires_grad=True)
        logits = model(x).squeeze()
        
        # Sort to find input ranks
        x_np = x.detach().numpy().flatten()
        sorted_indices = np.argsort(x_np) # from smallest (0) to largest (9)
        
        # Rank 8 is the 2nd-largest, Rank 9 is the maximum
        idx_max = sorted_indices[9]
        idx_sec = sorted_indices[8]
        
        # Decision Margin = logit(sec_max) - max(other logits)
        other_indices = [i for i in range(seq_len) if i != idx_sec]
        margin = logits[idx_sec] - torch.max(logits[other_indices])
        
        # Only use correct predictions to see what drives the *correct* logic
        if margin.item() > 0:
            model.zero_grad()
            margin.backward()
            
            # Extract gradients and align them by input RANK (not temporal position)
            grads = x.grad.numpy().flatten()
            aligned_grads = np.zeros(seq_len)
            for rank, orig_idx in enumerate(sorted_indices):
                aligned_grads[rank] = grads[orig_idx]
                
            avg_grads += aligned_grads
            valid_samples += 1
            
    avg_grads /= valid_samples

    # Plotting
    plt.figure(figsize=(10, 5))
    ranks = np.arange(seq_len)
    
    # Color coding: Gray for ignored, Red for Negative Boundary, Blue for Targets
    colors = ['#bdc3c7']*7 + ['#e74c3c', '#3498db', '#3498db']
    
    plt.bar(ranks, avg_grads, color=colors, edgecolor='black', zorder=3)
    plt.axhline(0, color='black', linewidth=1.2)
    
    plt.xticks(ranks, [f'Rank {i}\n(Smallest)' if i==0 else f'Rank {i}\n(Largest)' if i==9 else f'Rank {i}' for i in ranks])
    plt.ylabel('Gradient of Decision Margin', fontsize=12)
    plt.title('Phase 0: Unsupervised Feature Sensitivity Profile', fontsize=14, weight='bold')
    
    # Annotations
    plt.annotate('Negative Sensitivity\nBoundary', xy=(7, avg_grads[7]), xytext=(4, avg_grads[7] - 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=10, ha='center', bbox=dict(boxstyle="round", fc="white", ec="red", alpha=0.8))
                 
    plt.annotate('Target\nVariables', xy=(8.5, max(avg_grads[8], avg_grads[9])), xytext=(6, max(avg_grads[8], avg_grads[9]) + 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=10, ha='center', bbox=dict(boxstyle="round", fc="white", ec="blue", alpha=0.8))

    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig('phase0_gradient_profile.png', dpi=300)
    print("✅ Saved 'phase0_gradient_profile.png'")

if __name__ == "__main__":
    model = example_2nd_argmax()
    phase0_gradient_profile(model)