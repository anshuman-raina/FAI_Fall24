import matplotlib.pyplot as plt
import numpy as np

def visualize_detailed_cnn():
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)
    
    # Layer configurations with detailed shapes
    layers = [
        {'name': 'Input', 'shape': '8@128x128', 'type': 'input', 'kernel': None},
        {'name': 'Conv1', 'shape': '24@48x48', 'type': 'conv', 'kernel': '3x3'},
        {'name': 'MaxPool1', 'shape': '24@24x24', 'type': 'pool', 'kernel': '2x2'},
        {'name': 'Conv2', 'shape': '48@12x12', 'type': 'conv', 'kernel': '3x3'},
        {'name': 'MaxPool2', 'shape': '48@6x6', 'type': 'pool', 'kernel': '2x2'},
        {'name': 'Dense', 'shape': '1x256', 'type': 'dense', 'kernel': None},
        {'name': 'Output', 'shape': '1x128', 'type': 'output', 'kernel': None}
    ]
    
    # Enhanced colors and styling
    colors = {
        'input': '#E6F3FF',    # Light blue
        'conv': '#FFE6E6',     # Light red
        'pool': '#E6FFE6',     # Light green
        'dense': '#FFE6FF',    # Light purple
        'output': '#FFFFF0'    # Light yellow
    }
    
    x_start = 1
    x_spacing = 2.5
    
    # Draw feature extraction part
    for i, layer in enumerate(layers):
        x = x_start + i * x_spacing
        
        if layer['type'] in ['input', 'conv', 'pool']:
            # Draw stacked feature maps
            num_squares = 5 if layer['type'] != 'input' else 3
            square_size = 1.5 if layer['type'] == 'input' else 1.2
            
            # Draw main squares with 3D effect
            for j in range(num_squares):
                offset = j * 0.15
                # Draw shadow
                shadow = plt.Rectangle((x + offset + 0.05, 2 + offset - 0.05), 
                                    square_size, square_size,
                                    facecolor='gray', alpha=0.2)
                ax.add_patch(shadow)
                # Draw main square
                rect = plt.Rectangle((x + offset, 2 + offset), 
                                   square_size, square_size,
                                   facecolor=colors[layer['type']],
                                   edgecolor='black',
                                   linewidth=1.5)
                ax.add_patch(rect)
                
                # Add kernel visualization for conv/pool layers
                if layer['type'] in ['conv', 'pool'] and j == num_squares-1:
                    kernel_size = 0.3
                    kernel = plt.Rectangle((x + offset + square_size - kernel_size/2,
                                         2 + offset + square_size - kernel_size/2),
                                        kernel_size, kernel_size,
                                        facecolor='red', alpha=0.5)
                    ax.add_patch(kernel)
        
        else:  # Dense and output layers
            # Draw neurons
            neuron_spacing = 0.3
            num_neurons = 4
            for j in range(num_neurons):
                circle = plt.Circle((x, 2 + j * neuron_spacing), 
                                  0.1, 
                                  facecolor=colors[layer['type']],
                                  edgecolor='black')
                ax.add_patch(circle)
            
            # Add dots for more neurons
            plt.plot([x], [2 + num_neurons * neuron_spacing + 0.2], 'k.')
            plt.plot([x], [2 + num_neurons * neuron_spacing + 0.4], 'k.')
        
        # Add layer information
        plt.text(x, 1.2, f"{layer['name']}\n{layer['shape']}", 
                ha='center', va='top', fontsize=10, fontweight='bold')
        if layer['kernel']:
            plt.text(x, 4, f"Kernel: {layer['kernel']}", 
                    ha='center', va='bottom', fontsize=8)
        
        # Draw arrows between layers
        if i < len(layers) - 1:
            ax.arrow(x + 1.5, 2.5, x_spacing - 1.6, 0,
                    head_width=0.1, head_length=0.1,
                    fc='black', ec='black', linewidth=1.5)
    
    # Add section labels
    plt.text(x_start + x_spacing * 2, 0.5, 'Feature Extraction', 
             ha='center', fontsize=14, fontweight='bold')
    plt.text(x_start + x_spacing * 5.5, 0.5, 'Classification', 
             ha='center', fontsize=14, fontweight='bold')
    
    plt.title('Convolutional Neural Network Architecture', pad=20, fontsize=16, fontweight='bold')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 5)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

visualize_detailed_cnn()