import torch
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from sklearn.decomposition import PCA
from alg_zoo import example_2nd_argmax

def phase4_full_analysis(model, num_sequences=200):
    """
    Phase 4: Topological analysis + PCA visualization.
    Collects hidden states (all time steps) and metadata,
    computes persistence (H0, H1), and generates:
      - Persistence diagram with highlighted H1 point
      - 2D PCA scatter colored by gap size or true label
    """
    model.eval()
    device = next(model.parameters()).device
    hidden_size = model.rnn.hidden_size

    # Storage
    all_h = []                     # hidden states at all steps
    labels = []                     # true label for each step (repeated per sequence)
    gaps = []                       # gap for each step (repeated per sequence)

    print(f"🧬 Collecting {num_sequences} sequences × 10 steps = {num_sequences*10} points...")
    torch.manual_seed(42)
    inputs = torch.randn(num_sequences, 10, 1).to(device)

    with torch.no_grad():
        W_ih = model.rnn.weight_ih_l0
        W_hh = model.rnn.weight_hh_l0

        for i in range(num_sequences):
            # Compute true label and gap for this sequence
            seq_np = inputs[i, :, 0].cpu().numpy()  # shape (10,)
            sorted_vals = np.sort(seq_np)[::-1]
            gap = sorted_vals[0] - sorted_vals[1]
            true_label = np.where(seq_np == sorted_vals[1])[0][0]

            # Run RNN and collect states
            h = torch.zeros(1, hidden_size).to(device)
            for t in range(10):
                x_t = inputs[i, t]
                h = torch.relu(h @ W_hh.T + x_t @ W_ih.T)
                all_h.append(h.cpu().numpy().flatten())
                labels.append(true_label)
                gaps.append(gap)

    h_cloud = np.array(all_h)
    labels = np.array(labels)
    gaps = np.array(gaps)

    # ---- PCA for visualization (2D) ----
    print("📉 Running PCA (2D) for visualization...")
    pca_vis = PCA(n_components=2)
    h_2d = pca_vis.fit_transform(h_cloud)

    # ---- Persistent homology on a subsample (or full cloud with threshold) ----
    # To avoid memory issues, we can use the full cloud but with a distance threshold
    print("🕳️ Computing persistence (H0, H1) on full cloud (thresh=2.0)...")
    result = ripser(h_cloud, maxdim=1, thresh=2.0)
    dgms = result['dgms']

    # Find the most persistent H1 point
    if len(dgms) > 1 and len(dgms[1]) > 0:
        h1_lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
        max_h1_idx = np.argmax(h1_lifetimes)
        max_h1_point = dgms[1][max_h1_idx]
        max_h1_lifetime = h1_lifetimes[max_h1_idx]
    else:
        max_h1_point = None
        max_h1_lifetime = 0

    print(f"Max H1 persistence: {max_h1_lifetime:.4f}")

    # ---- Figure 4.1: Persistence diagram with highlighted H1 ----
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    plot_diagrams(dgms, show=False, ax=ax1)
    if max_h1_point is not None:
        # Highlight the most persistent H1 point with a red circle
        ax1.scatter(max_h1_point[0], max_h1_point[1], s=100,
                    facecolors='none', edgecolors='red', linewidths=2,
                    label=f'H₁ (persistence={max_h1_lifetime:.3f})')
    ax1.set_title("Persistence Diagram (H₀ black, H₁ red)")
    ax1.legend()
    plt.tight_layout()
    plt.savefig("figure_4_1_persistence.png", dpi=300)
    plt.show()

    # ---- Figure 4.2: PCA scatter colored by gap ----
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter colored by gap size
    sc = ax2a.scatter(h_2d[:, 0], h_2d[:, 1], c=gaps, cmap='viridis',
                      s=5, alpha=0.6)
    ax2a.set_title("Hidden States Colored by Gap Size")
    ax2a.set_xlabel("PC1")
    ax2a.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax2a, label='Gap')

    # Scatter colored by true label (second‑max index)
    sc2 = ax2b.scatter(h_2d[:, 0], h_2d[:, 1], c=labels, cmap='tab10',
                       s=5, alpha=0.6, vmin=0, vmax=9)
    ax2b.set_title("Hidden States Colored by True Label")
    ax2b.set_xlabel("PC1")
    ax2b.set_ylabel("PC2")
    plt.colorbar(sc2, ax=ax2b, label='True Label', ticks=range(10))

    plt.tight_layout()
    plt.savefig("figure_4_2_pca.png", dpi=300)
    plt.show()

    # Optional Figure 4.3: 3D PCA (first three components)
    print("📉 Running PCA (3D) for optional 3D plot...")
    pca_3d = PCA(n_components=3)
    h_3d = pca_3d.fit_transform(h_cloud)

    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection='3d')
    sc3 = ax3.scatter(h_3d[:, 0], h_3d[:, 1], h_3d[:, 2],
                      c=gaps, cmap='viridis', s=5, alpha=0.6)
    ax3.set_title("3D PCA (colored by gap)")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    plt.colorbar(sc3, ax=ax3, label='Gap', shrink=0.5)
    plt.tight_layout()
    plt.savefig("figure_4_3_pca_3d.png", dpi=300)
    plt.show()

    return dgms, h_cloud, h_2d, gaps, labels

if __name__ == "__main__":
    model = example_2nd_argmax()
    phase4_full_analysis(model, num_sequences=200)