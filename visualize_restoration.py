import matplotlib.pyplot as plt
import torch

def visualize_restoration(model, loader, device="cuda"):
    model.eval()

    # 1. Get a single batch
    # We create a fresh iterator to grab a batch without disrupting the main loop
    batch = next(iter(loader))
    inputs = batch['pixel_values'].to(device)
    targets = batch['labels'].to(device)
    q_labels = batch['q_scale'].to(device).long()

    # 2. Run the model
    # Fixed timestep of 0 for direct restoration visualization
    dummy_timesteps = torch.zeros((inputs.size(0),), device=device).long()

    with torch.no_grad():
        predicted_error = model(inputs, dummy_timesteps, q_labels)

    # 3. Visualization Setup
    # Let's visualize the "Error Maps" instead of full images for now (easier to debug)
    idx = 0 # First image in batch
    channel = 0 # DC Component (Low frequency)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # A. Input (The Quantized "Zeroed" Error - effectively what the model sees as 'missing')
    # Let's visualize the Target Error (The ground truth missing info)
    target_img = targets[idx, channel].cpu().numpy()
    
    # We set vmin/vmax based on the TARGET so all plots share the same scale
    vmin, vmax = target_img.min(), target_img.max()

    ax = axes[0]
    im = ax.imshow(target_img, cmap='bwr', vmin=vmin, vmax=vmax)
    ax.set_title("True Missing Info (Target)")
    plt.colorbar(im, ax=ax)

    # B. Model Prediction
    pred_img = predicted_error[idx, channel].cpu().numpy()
    ax = axes[1]
    im = ax.imshow(pred_img, cmap='bwr', vmin=vmin, vmax=vmax)
    ax.set_title("Model Prediction")
    plt.colorbar(im, ax=ax)

    # C. Residual (Difference)
    diff = target_img - pred_img
    ax = axes[2]
    im = ax.imshow(diff, cmap='bwr', vmin=vmin, vmax=vmax)
    ax.set_title(f"Remaining Error\n(MSE: {torch.mean(torch.tensor(diff)**2):.4f})")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()
