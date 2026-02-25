# #!/usr/bin/env python3
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from alg_zoo import example_2nd_argmax

# def phase2_adaptive_heatmap(model, num_sequences=1000, neuron=2, n_bins=20):
#     device = "cpu"
#     model.eval()
    
#     # Generate data
#     inputs = torch.randn(num_sequences, 10).to(device)
#     gaps = []
#     acts = []
    
#     with torch.no_grad():
#         W_ih, W_hh = model.rnn.weight_ih_l0, model.rnn.weight_hh_l0
#         for i in range(num_sequences):
#             seq = inputs[i]
#             sorted_vals = np.sort(seq.numpy())[::-1]
#             gaps.append(sorted_vals[0] - sorted_vals[1])
            
#             h = torch.zeros(1, 16).to(device)
#             seq_acts = []
#             for t in range(10):
#                 h = torch.relu(h @ W_hh.T + seq[t].view(1,1) @ W_ih.T)
#                 seq_acts.append(h[0, neuron].item())
#             acts.append(seq_acts)
            
#     acts = np.array(acts)
#     gaps = np.array(gaps)
    
#     # Binning by gap
#     sort_idx = np.argsort(gaps)
#     sorted_acts = acts[sort_idx]
    
#     bin_edges = np.linspace(0, num_sequences, n_bins+1, dtype=int)
#     variance_map = np.zeros((n_bins, 10))
    
#     for b in range(n_bins):
#         start, end = bin_edges[b], bin_edges[b+1]
#         bin_data = sorted_acts[start:end]
#         variance_map[b] = np.var(bin_data, axis=0)

#     # Adaptive Threshold: 25th percentile of all valid variances (ignoring t=0)
#     valid_vars = variance_map[:, 1:].flatten()
#     adaptive_threshold = np.percentile(valid_vars, 25)
    
#     print(f"Adaptive Stabilization Threshold computed: {adaptive_threshold:.4f}")

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     im = plt.imshow(variance_map, aspect='auto', origin='lower', 
#                     extent=[0, 9, 0, 100], cmap='viridis_r')
    
#     cbar = plt.colorbar(im)
#     cbar.set_label('Activation Variance', fontsize=12)
    
#     # Calculate and plot Stabilization Front (t>=1)
#     stab_times = []
#     for b in range(n_bins):
#         row = variance_map[b, 1:]
#         stable_idx = np.where(row < adaptive_threshold)[0]
#         if len(stable_idx) > 0:
#             stab_times.append(stable_idx[0] + 1)
#         else:
#             stab_times.append(9)
            
#     y_centers = np.linspace(5, 95, n_bins)
#     plt.plot(stab_times, y_centers, color='cyan', linestyle='--', linewidth=3, label='Stabilization Front')
    
#     plt.xlabel('Time Step', fontsize=12)
#     plt.ylabel('Gap Percentile (0 = Hardest, 100 = Easiest)', fontsize=12)
#     plt.title(f'Difficulty Scale-Space for Neuron {neuron}', fontsize=14, weight='bold')
#     plt.legend(loc='lower right')
#     plt.grid(False)
#     plt.tight_layout()
#     plt.savefig('phase2_adaptive_heatmap.png', dpi=300)
#     print("Plot saved to 'phase2_adaptive_heatmap.png'")

# if __name__ == "__main__":
#     model = example_2nd_argmax()
#     phase2_adaptive_heatmap(model)

# #!/usr/bin/env python3
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from alg_zoo import example_2nd_argmax

# def phase2_adaptive_heatmap(model, num_sequences=3000, neuron=2, n_bins=20):
#     device = "cpu"
#     model.eval()
    
#     print(f"⏳ Phase 2: Generating Scale-Space Heatmap for Neuron {neuron}...")
    
#     # 1. Generate realistic Gaussian inputs
#     inputs = torch.randn(num_sequences, 10).to(device)
#     gaps = []
#     acts = []
    
#     with torch.no_grad():
#         W_ih, W_hh = model.rnn.weight_ih_l0, model.rnn.weight_hh_l0
#         for i in range(num_sequences):
#             seq = inputs[i]
            
#             # Calculate gap (v1 - v2)
#             sorted_vals = np.sort(seq.numpy())[::-1]
#             gaps.append(sorted_vals[0] - sorted_vals[1])
            
#             # Unroll RNN to collect activations at each time step
#             h = torch.zeros(1, 16).to(device)
#             seq_acts = []
#             for t in range(10):
#                 h = torch.relu(h @ W_hh.T + seq[t].view(1,1) @ W_ih.T)
#                 seq_acts.append(h[0, neuron].item())
#             acts.append(seq_acts)
            
#     acts = np.array(acts)
#     gaps = np.array(gaps)
    
#     # 2. Sort by Gap Size (Hardest to Easiest)
#     sort_idx = np.argsort(gaps)
#     sorted_acts = acts[sort_idx]
    
#     # 3. Bin into percentiles
#     bin_edges = np.linspace(0, num_sequences, n_bins+1, dtype=int)
#     variance_map = np.zeros((n_bins, 10))
    
#     for b in range(n_bins):
#         start, end = bin_edges[b], bin_edges[b+1]
#         bin_data = sorted_acts[start:end]
#         # Calculate variance of the neuron's activation across sequences in this difficulty bin
#         variance_map[b] = np.var(bin_data, axis=0)

#     # 4. Adaptive Threshold (25th percentile of all valid variances, ignoring t=0)
#     valid_vars = variance_map[:, 1:].flatten()
#     adaptive_threshold = np.percentile(valid_vars, 25)
#     print(f"   Adaptive Stabilization Threshold computed: {adaptive_threshold:.4f}")

#     # 5. Calculate Strict Monotonic Stabilization Front
#     stab_times = []
#     for b in range(n_bins):
#         # Ignore t=0 (index 0) because h is initialized to 0, so variance is artificially 0
#         row = variance_map[b, 1:] 
        
#         # Where is the variance safely below the threshold?
#         is_stable = row < adaptive_threshold
        
#         # Find the LAST time step it was unstable (False)
#         unstable_indices = np.where(~is_stable)[0]
        
#         if len(unstable_indices) == 0:
#             stab_times.append(1) # Stable immediately at t=1
#         elif unstable_indices[-1] == len(row) - 1:
#             stab_times.append(9) # Never stabilizes before the final step
#         else:
#             # Stabilizes permanently on the step AFTER its last unstable step
#             # (+1 for 0-indexing offset, +1 to move to the next time step)
#             stab_times.append(unstable_indices[-1] + 2)
            
#     # Calculate Y-coordinates for the line (centers of the bins)
#     y_centers = np.linspace(100/(2*n_bins), 100 - 100/(2*n_bins), n_bins)

#     # 6. Publication-Ready Plotting
#     plt.figure(figsize=(10, 6))
    
#     # Heatmap (viridis_r makes low variance dark/purple, high variance bright/yellow)
#     im = plt.imshow(variance_map, aspect='auto', origin='lower', 
#                     extent=[0, 9, 0, 100], cmap='viridis_r')
    
#     cbar = plt.colorbar(im)
#     cbar.set_label('Activation Variance', fontsize=12)
    
#     # Overlay the smooth Stabilization Front
#     plt.plot(stab_times, y_centers, color='cyan', linestyle='--', linewidth=3, label='Stabilization Front')
    
#     # Annotations
#     plt.text(0.5, 95, 'Easy (Large Gap)', ha='center', va='center', color='black', 
#              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
#     plt.text(0.5, 5, 'Hard (Small Gap)', ha='center', va='center', color='white', 
#              bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

#     plt.xlabel('Time Step', fontsize=12)
#     plt.ylabel('Gap Percentile (0 = Hardest, 100 = Easiest)', fontsize=12)
#     plt.title(f'Difficulty Scale-Space for Neuron {neuron}', fontsize=14, weight='bold')
    
#     # Format X-axis to show discrete time steps properly
#     plt.xticks(range(10))
#     plt.legend(loc='lower right', framealpha=0.9)
#     plt.grid(False)
    
#     plt.tight_layout()
#     plt.savefig('phase2_adaptive_heatmap.png', dpi=300)
#     print("✅ Plot successfully saved to 'phase2_adaptive_heatmap.png'")

# if __name__ == "__main__":
#     model = example_2nd_argmax()
#     # Feel free to change neuron=4 or neuron=7 if they look cleaner for your paper!
#     phase2_adaptive_heatmap(model, num_sequences=3000, neuron=2, n_bins=20)

#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from alg_zoo import example_2nd_argmax

def phase2_smoothed_heatmap(model, num_sequences=3000, neuron=2, n_bins=20):
    device = "cpu"
    model.eval()
    
    print(f"⏳ Phase 2: Generating Smoothed Scale-Space Heatmap for Neuron {neuron}...")
    
    inputs = torch.randn(num_sequences, 10).to(device)
    gaps = []
    acts = []
    
    with torch.no_grad():
        W_ih, W_hh = model.rnn.weight_ih_l0, model.rnn.weight_hh_l0
        for i in range(num_sequences):
            seq = inputs[i]
            sorted_vals = np.sort(seq.numpy())[::-1]
            gaps.append(sorted_vals[0] - sorted_vals[1])
            
            h = torch.zeros(1, 16).to(device)
            seq_acts = []
            for t in range(10):
                h = torch.relu(h @ W_hh.T + seq[t].view(1,1) @ W_ih.T)
                seq_acts.append(h[0, neuron].item())
            acts.append(seq_acts)
            
    acts = np.array(acts)
    gaps = np.array(gaps)
    
    sort_idx = np.argsort(gaps)
    sorted_acts = acts[sort_idx]
    
    bin_edges = np.linspace(0, num_sequences, n_bins+1, dtype=int)
    variance_map = np.zeros((n_bins, 10))
    
    for b in range(n_bins):
        start, end = bin_edges[b], bin_edges[b+1]
        bin_data = sorted_acts[start:end]
        variance_map[b] = np.var(bin_data, axis=0)

    # NEW: Apply Gaussian Smoothing across the difficulty bins (Y-axis)
    variance_map_smoothed = gaussian_filter1d(variance_map, sigma=1.5, axis=0)

    valid_vars = variance_map_smoothed[:, 1:].flatten()
    adaptive_threshold = np.percentile(valid_vars, 25)

    stab_times = []
    for b in range(n_bins):
        row = variance_map_smoothed[b, 1:] 
        is_stable = row < adaptive_threshold
        unstable_indices = np.where(~is_stable)[0]
        
        if len(unstable_indices) == 0:
            stab_times.append(1)
        elif unstable_indices[-1] == len(row) - 1:
            stab_times.append(9)
        else:
            stab_times.append(unstable_indices[-1] + 2)
            
    y_centers = np.linspace(100/(2*n_bins), 100 - 100/(2*n_bins), n_bins)

    plt.figure(figsize=(10, 6))
    im = plt.imshow(variance_map_smoothed, aspect='auto', origin='lower', 
                    extent=[0, 9, 0, 100], cmap='viridis_r')
    
    cbar = plt.colorbar(im)
    cbar.set_label('Smoothed Activation Variance', fontsize=12)
    
    plt.plot(stab_times, y_centers, color='cyan', linestyle='--', linewidth=3, label='Stabilization Front')
    
    plt.text(0.5, 95, 'Easy (Large Gap)', ha='center', va='center', color='black', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.text(0.5, 5, 'Hard (Small Gap)', ha='center', va='center', color='white', 
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Gap Percentile (0 = Hardest, 100 = Easiest)', fontsize=12)
    plt.title(f'Difficulty Scale-Space for Neuron {neuron}', fontsize=14, weight='bold')
    
    plt.xticks(range(10))
    plt.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig('phase2_smoothed_heatmap.png', dpi=300)
    print("✅ Plot successfully saved to 'phase2_smoothed_heatmap.png'")

if __name__ == "__main__":
    model = example_2nd_argmax()
    phase2_smoothed_heatmap(model, num_sequences=3000, neuron=2, n_bins=30) # Increased bins for smoother look