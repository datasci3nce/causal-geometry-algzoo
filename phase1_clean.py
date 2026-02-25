#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax

def phase1_mri_scan(model, num_sequences=3000, seq_len=10):
    device = "cpu"
    model.eval()
    hidden_size = model.rnn.hidden_size
    
    print("🧠 Phase 1: Running Causal Ablation Scan...")
    
    # Baseline accuracy
    test_inputs = torch.randn(num_sequences, seq_len)
    truths = torch.tensor([np.argsort(seq.numpy())[-2] for seq in test_inputs])
    
    with torch.no_grad():
        preds = torch.argmax(model(test_inputs), dim=1)
        base_acc = (preds == truths).float().mean().item()
        
    print(f"   Baseline Accuracy: {base_acc:.2%}")
    
    importance = []
    W_ih, W_hh = model.rnn.weight_ih_l0, model.rnn.weight_hh_l0
    
    # Ablate each neuron
    for n in range(hidden_size):
        correct = 0
        with torch.no_grad():
            for i in range(num_sequences):
                h = torch.zeros(1, hidden_size)
                for t in range(seq_len):
                    h = torch.relu(h @ W_hh.T + test_inputs[i, t].view(1,1) @ W_ih.T)
                    # Ablate!
                    h[0, n] = 0.0 
                
                pred = torch.argmax(model.linear(h)).item()
                if pred == truths[i].item():
                    correct += 1
                    
        ablated_acc = correct / num_sequences
        drop = base_acc - ablated_acc
        importance.append(drop)
        
    importance = np.array(importance)
    dynamic_threshold = np.max(importance) * 0.5
    core_neurons = np.where(importance > dynamic_threshold)[0]
    
    # Plotting
    plt.figure(figsize=(10, 5))
    neurons = np.arange(hidden_size)
    
    colors = ['#e74c3c' if i in core_neurons else '#bdc3c7' for i in neurons]
    
    bars = plt.bar(neurons, importance, color=colors, edgecolor='black', zorder=3)
    plt.axhline(dynamic_threshold, color='#f39c12', linestyle='--', linewidth=2, 
                label=f'Elbow Threshold (50% max drop)', zorder=4)
                
    plt.xticks(neurons)
    plt.xlabel('Hidden Neuron Index', fontsize=12)
    plt.ylabel('Accuracy Drop (Δ) upon Ablation', fontsize=12)
    plt.title('Phase 1: Holistic Circuit Discovery (Causal Ablation)', fontsize=14, weight='bold')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    # Add text box
    plt.text(0.5, max(importance)*0.85, f"Distributed Substrate:\n{len(core_neurons)} / {hidden_size} Neurons Required", 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
             
    plt.tight_layout()
    plt.savefig('phase1_causal_ablation.png', dpi=300)
    print(f"✅ Saved 'phase1_causal_ablation.png'. Found {len(core_neurons)} core neurons.")

if __name__ == "__main__":
    model = example_2nd_argmax()
    phase1_mri_scan(model)