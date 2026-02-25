import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define file paths
base_path = ""  # Adjust with your correct path after downloading
datasets = {
    "CIFAR": [
        "cifar_Fed_C_non_iid_round_metrics.csv",
        "cifar_Fed_B_non_iid_round_metrics.csv",
        "CIFAR_Fed_A_noniid_round_metrics.csv",
        "cifar_Fed_ABC_non_iid_hybrid_round_metrics.csv"
    ],
    "HAR": [
        "HAR_Fed_C_non_iid_round_metrics.csv",
        "HAR_Fed_B_non_iid_round_metrics.csv",
        "HAR_Fed_A_non_iid_round_metrics.csv",
        "HAR_Fed_ABC_non_iid_hybrid_round_metrics.csv",
    ],
    "MNIST": [
        "Mnist_Fed_C_non_iid_round_metrics.csv",        
        "Mnist_Fed_B_non_iid_round_metrics.csv",
        "Mnist_Fed_A_non_iid_round_metrics.csv",
        "Mnist_Fed_ABC_non_iid_hybrid_round_metrics.csv",
    ]
}

# Labels for different federated learning strategies
labels = {
    "CIFAR": [
        "Hospital A", 
        "Hospital B", 
        "Hospital C", 
        "Collab"
    ],
    "HAR": [
        "Hospital A", 
        "Hospital B", 
        "Hospital C", 
        "Collab"
    ],
    "MNIST": [
        "Hospital A", 
        "Hospital B", 
        "Hospital C", 
        "Collab"
    ]
}

# Define colorblind-friendly colors
colors = plt.cm.Set2(np.linspace(0, 1, len(labels["CIFAR"])))
hatch_patterns = ["", "//", "xx", "--", "||", ".."]  # For better distinction in grayscale

# Store results
highest_accuracies = {}
total_training_times = {}

# Ensure proper font sizes and spacing
plt.rcParams.update({
    "axes.labelsize": 22,
    "axes.titlesize": 26,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "grid.alpha": 0.5
})

# Loop through datasets separately
for dataset_name, file_names in datasets.items():
    print(f"\nProcessing {dataset_name}...\n")
    dataset_accuracies = {}
    dataset_times = {}

    # Plot Loss
    plt.figure(figsize=(10, 6))
    for file_name, label, color in zip(file_names, labels[dataset_name], colors):
        file_path = os.path.join(base_path, file_name)
        df = pd.read_csv(file_path)

        # Store highest accuracy achieved
        highest_accuracy = df["accuracy"].max()
        dataset_accuracies[label] = highest_accuracy

        # Store total training time
        total_time = np.sum(df["round_duration"])
        dataset_times[label] = total_time

        # Plot Loss
        plt.plot(df["round"], df["loss"], label=label, color=color, linestyle='-', linewidth=2, marker='o', markersize=3)

    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.title(f"{dataset_name} - Loss over Rounds")
    plt.legend(loc='best', fontsize=16)
    plt.grid(True, linestyle='--')
    plt.savefig(f"{dataset_name}_Loss.png", dpi=300, bbox_inches='tight')
    plt.show()

    highest_accuracies[dataset_name] = dataset_accuracies
    total_training_times[dataset_name] = dataset_times
    
    # Plot Accuracy over Rounds
    plt.figure(figsize=(10, 6))
    for file_name, label, color in zip(file_names, labels[dataset_name], colors):
        file_path = os.path.join(base_path, file_name)
        df = pd.read_csv(file_path)
        plt.plot(df["round"], df["accuracy"], label=label, color=color, linestyle='-', linewidth=2, marker='s', markersize=3)

    plt.xlabel("Rounds")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{dataset_name} - Accuracy over Rounds")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--')
    plt.savefig(f"{dataset_name}_Accuracy.png", dpi=300, bbox_inches='tight')
    plt.show() 
    
    # Plot Round Duration with Average
    plt.figure(figsize=(10, 6))
    for file_name, label, color in zip(file_names, labels[dataset_name], colors):
        file_path = os.path.join(base_path, file_name)
        df = pd.read_csv(file_path)
       
        # Compute Average Duration
        avg_duration = df["round_duration"].mean()
       
        # Plot Round Duration
        plt.plot(df["round"], df["round_duration"], label=label, color=color, linestyle='-', linewidth=2, marker='x', markersize=3)
       
        # Add horizontal line for average duration
        plt.axhline(y=avg_duration, color=color, linestyle='dashed', alpha=0.6, linewidth=1.5,
                   label=f"Avg {label}: {avg_duration:.2f}s")

    plt.xlabel("Rounds")
    plt.ylabel("Round Duration (s)")
    plt.title(f"{dataset_name} - Round Duration over Rounds")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--')
    plt.savefig(f"{dataset_name}_Round_Duration.png", dpi=300, bbox_inches='tight')
    plt.show()

# **Highest Accuracy Achieved per Strategy**
plt.figure(figsize=(12, 6))
bar_width = 0.3
bar_positions = np.arange(len(labels["CIFAR"]))
z=0
for i, (dataset_name, acc_dict) in enumerate(highest_accuracies.items()):
    bars = plt.bar(bar_positions + i * bar_width, acc_dict.values(), width=bar_width, color=colors[i], hatch=hatch_patterns[i])
    
    # Annotate bars with exact accuracy values
    for bar in bars:
        print(bar)
        height = bar.get_height()
        x_offset = 4  # Tweak this if needed
        x_position = bar.get_x() + bar.get_width()/2
        z=z+1
        # Add slight horizontal offset depending on dataset index
        print ("i "+ str(i))
        if i == 0:
            x_position -= x_offset
        elif i == 1:
            x_position += x_offset
        print(z)    
        if(z>15):
            x_position =0
        
        plt.text(bar.get_x() + bar.get_width()/2, height + x_position, f"{height:.2f}%", ha='center', va='bottom', fontsize=16, fontweight='bold')


        #plt.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}%", ha='center', va='bottom', fontsize=16, fontweight='bold')

plt.xlabel("Federated Learning Strategies")
plt.ylabel("Highest Accuracy (%)")
plt.title("Highest Accuracy Achieved per Strategy")
plt.xticks(bar_positions + bar_width, labels["CIFAR"], rotation=45, ha="right")
plt.legend(highest_accuracies.keys(), loc='upper left', bbox_to_anchor=(1,1))
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("Highest_Accuracy_Per_Strategy.png", dpi=300, bbox_inches='tight')
plt.show()

# **Total Training Time per Strategy**
plt.figure(figsize=(12, 6))
for i, (dataset_name, time_dict) in enumerate(total_training_times.items()):
    bars = plt.bar(bar_positions + i * bar_width, time_dict.values(), width=bar_width, color=colors[i], hatch=hatch_patterns[i])

    # Annotate bars with exact time values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f"{height:.1f}s", ha='center', va='bottom', fontsize=16, fontweight='bold')

plt.xlabel("Federated Learning Strategies")
plt.ylabel("Total Training Time (s)")
plt.title("Total Training Time per Strategy")
plt.xticks(bar_positions + bar_width, labels["CIFAR"], rotation=45, ha="right")
plt.legend(total_training_times.keys(), loc='upper left', bbox_to_anchor=(1,1))
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("Total_Training_Time_Per_Strategy.png", dpi=300, bbox_inches='tight')
plt.show()
