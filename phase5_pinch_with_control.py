#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from alg_zoo import example_2nd_argmax

def phase5_controlled_pinch(model, num_test=1000, intervention_step=5, strength=5.0):
    device = "cpu"
    model.eval()
    
    print("🚀 Extracting Manifold Geometry...")
    all_h = []
    inputs_pca = torch.randn(200, 10).to(device)
    with torch.no_grad():
        W_ih, W_hh = model.rnn.weight_ih_l0, model.rnn.weight_hh_l0
        for i in range(200):
            h = torch.zeros(1, 16).to(device)
            for t in range(10):
                h = torch.relu(h @ W_hh.T + inputs_pca[i, t].view(1,1) @ W_ih.T)
                all_h.append(h.cpu().numpy().flatten())
                
    pca = PCA(n_components=16).fit(np.array(all_h))
    
    ring_vec = torch.tensor(pca.components_[0], dtype=torch.float32).to(device)
    pinch_vec = torch.tensor(pca.components_[-1], dtype=torch.float32).to(device)
    
    # PROPER FIX: Create a random vector and orthogonalize it against BOTH the ring AND the pinch
    rand_vec = torch.randn(16).to(device)
    rand_vec = rand_vec - (torch.dot(rand_vec, ring_vec) * ring_vec)
    rand_vec = rand_vec - (torch.dot(rand_vec, pinch_vec) * pinch_vec)
    rand_vec = rand_vec / torch.norm(rand_vec) # Unit norm

    print(f"🚀 Running Causal Interventions (Strength = {strength})...")
    
    results = {'Baseline': 0, 'Tangential (Ring)': 0, 'Random OOD (Control)': 0, 'Orthogonal (Pinch)': 0}
    inputs_test = torch.randn(num_test, 10).to(device)
    truths = [np.argsort(inputs_test[i].numpy())[-2] for i in range(num_test)]

    for mode in results.keys():
        correct = 0
        with torch.no_grad():
            for i in range(num_test):
                h = torch.zeros(1, 16).to(device)
                for t in range(10):
                    h = torch.relu(h @ W_hh.T + inputs_test[i, t].view(1,1) @ W_ih.T)
                    
                    if t == intervention_step:
                        if mode == 'Tangential (Ring)': h += ring_vec * strength
                        elif mode == 'Orthogonal (Pinch)': h += pinch_vec * strength
                        elif mode == 'Random OOD (Control)': h += rand_vec * strength
                            
                logits = model.linear(h)
                if torch.argmax(logits).item() == truths[i]:
                    correct += 1
        results[mode] = correct / num_test

    print("\n" + "="*50)
    print("PHASE 5: CONTROLLED CAUSAL PINCH TEST")
    print("="*50)
    for k, v in results.items():
        print(f"{k:<25}: {v:.2%}")
    
    logic_gap = results['Tangential (Ring)'] - results['Orthogonal (Pinch)']
    ood_gap = results['Tangential (Ring)'] - results['Random OOD (Control)']
    
    print("-" * 50)
    print(f"Logic Gap (Ring vs Pinch): {logic_gap:.2%}")
    print(f"OOD Gap (Ring vs Random):  {ood_gap:.2%}")
    if logic_gap > ood_gap + 0.05:
        print("\n✅ SUCCESS: The orthogonal pinch is significantly more harmful than random OOD.")
    else:
        print("\n⚠️ WARNING: Random OOD is similarly harmful. Manifold effect is weak.")
    print("="*50)

    # Visualization
    plt.figure(figsize=(8, 6))
    labels = list(results.keys())
    values = [v*100 for v in results.values()]
    colors = ['#95a5a6', '#3498db', '#f39c12', '#e74c3c']
    
    bars = plt.bar(labels, values, color=colors, edgecolor='black', alpha=0.9)
    plt.axhline(values[0], color='gray', linestyle='--', alpha=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Manifold Pinch Test with Random OOD Control', fontsize=14, weight='bold')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase5_controlled_pinch.png', dpi=300)
    print("Plot saved to 'phase5_controlled_pinch.png'")

if __name__ == "__main__":
    model = example_2nd_argmax()
    phase5_controlled_pinch(model, num_test=2000, strength=6.0)