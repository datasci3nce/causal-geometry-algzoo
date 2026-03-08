#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from alg_zoo import example_2nd_argmax

def find_fixed_point(model, lr=0.05, steps=5000):
    print("🔍 Hunting for Fixed Points via Gradient Descent...")
    
    # 1. THE FIX: Create a random initialization to avoid the trivial zero-state.
    # We do the math FIRST, then clone, detach, and require grad to make it a perfect "leaf tensor".
    init_val = torch.randn(1, 16) * 0.5 
    h_star = init_val.clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([h_star], lr=lr)
    
    W_hh = model.rnn.weight_hh_l0
    zero_input = (torch.zeros(1, 1) @ model.rnn.weight_ih_l0.T).detach()

    for i in range(steps):
        optimizer.zero_grad()
        
        # What is the next state if input is neutral?
        h_next = torch.relu(h_star @ W_hh.T + zero_input)
        
        # 2. THE FIX: Use MSE instead of L2 Norm for smoother gradients near zero
        loss = torch.nn.functional.mse_loss(h_next, h_star)
        
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            with torch.no_grad():
                current_loss = torch.nn.functional.mse_loss(torch.relu(h_star @ W_hh.T + zero_input), h_star).item()
            print(f"  Step {i:04d}: Movement Loss = {current_loss:.8f}")
            
    with torch.no_grad():
        final_loss = torch.nn.functional.mse_loss(torch.relu(h_star @ W_hh.T + zero_input), h_star).item()
    print(f"✅ Found Fixed Point! Final Movement Loss: {final_loss:.8f}")
    
    return h_star.detach()

def analyze_jacobian_at_point(model, h_star):
    print("\n📐 Calculating Jacobian at Fixed Point...")
    W_hh = model.rnn.weight_hh_l0.detach().numpy()
    h_np = h_star.numpy().flatten()
    
    # Determine which ReLUs are active at h_star (for the next state)
    pre_activation = h_np @ W_hh.T
    active_neurons = (pre_activation > 0).astype(float)
    
    # Jacobian = diag(active) @ W_hh
    Jacobian = np.diag(active_neurons) @ W_hh
    
    eigenvalues, eigenvectors = np.linalg.eig(Jacobian)
    
    # Find Unstable/Neutral Directions (|lambda| >= 0.99)
    magnitudes = np.abs(eigenvalues)
    unstable_indices = np.where(magnitudes >= 0.99)[0]
    
    print(f"  Total Eigenvalues: {len(eigenvalues)}")
    print(f"  Unstable/Neutral Directions (|λ| >= 1): {len(unstable_indices)}")
    
    for idx in unstable_indices:
        print(f"    λ = {eigenvalues[idx]:.3f} (Magnitude: {magnitudes[idx]:.3f})")
        
    return eigenvectors[:, unstable_indices]

def plot_unstable_manifold(model, h_star, unstable_vecs):
    print("\n🎨 Plotting the Unstable Manifold over the PCA Ring...")
    # Collect hidden states for PCA mapping
    all_h =[]
    inputs = torch.randn(500, 10)
    with torch.no_grad():
        W_ih, W_hh = model.rnn.weight_ih_l0, model.rnn.weight_hh_l0
        for i in range(500):
            h = torch.zeros(1, 16)
            for t in range(10):
                h = torch.relu(h @ W_hh.T + inputs[i, t].view(1,1) @ W_ih.T)
                all_h.append(h.numpy().flatten())
    
    pca = PCA(n_components=2)
    h_2d = pca.fit_transform(np.array(all_h))
    
    # Project fixed point
    h_star_2d = pca.transform(h_star.numpy())
    
    plt.figure(figsize=(10, 8))
    plt.scatter(h_2d[:, 0], h_2d[:, 1], c='gray', s=10, alpha=0.2, label='Hidden States (The Ring)')
    plt.scatter(h_star_2d[0, 0], h_star_2d[0, 1], c='red', s=150, marker='X', zorder=5, label='Fixed Point $h^*$')
    
    # Plot unstable eigenvectors
    if unstable_vecs.shape[1] > 0:
        for i in range(unstable_vecs.shape[1]):
            vec = np.real(unstable_vecs[:, i])   # take real part
            
            # Scale the vector so it's visible on the plot
            scale_factor = 5.0
            point_on_vec = h_star.numpy().flatten() + vec * scale_factor
            
            # Project vector endpoint into PCA space to get direction
            vec_2d = pca.transform(point_on_vec.reshape(1, -1)) - h_star_2d
            
            plt.arrow(h_star_2d[0, 0], h_star_2d[0, 1], 
                      vec_2d[0, 0], vec_2d[0, 1],
                      color='blue', width=0.3, head_width=1.5, zorder=4,
                      label='Unstable Eigenvector' if i==0 else "")
    else:
        print("  ⚠️ No unstable vectors found to plot.")

    plt.title("Dynamical Systems Proof: The Ring as an Unstable Manifold", fontsize=15, weight='bold')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("phase7_fixed_point.png", dpi=300)
    print("✅ Saved 'phase7_fixed_point.png'")

if __name__ == "__main__":
    model = example_2nd_argmax()
    model.eval()
    
    # Run the full pipeline
    h_star = find_fixed_point(model)
    unstable_vecs = analyze_jacobian_at_point(model, h_star)
    plot_unstable_manifold(model, h_star, unstable_vecs)