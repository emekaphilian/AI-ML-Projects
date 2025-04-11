# GAN for MNIST Digit Generation
# Author: [Your Name]
# This script trains a Generative Adversarial Network (GAN) using PyTorch to generate handwritten digits.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# ========== Hyperparameters ==========
latent_size = 64         # Size of the noise vector (input to generator)
hidden_size = 256        # Hidden layer size for both Generator and Discriminator
image_size = 784         # 28x28 pixels = 784
batch_size = 100
num_epochs = 200
learning_rate = 0.0002

# ========== Device configuration ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== MNIST Dataset ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize images to [-1, 1]
])
mnist = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

# ========== Discriminator Model ==========
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
).to(device)

# ========== Generator Model ==========
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
).to(device)

# ========== Loss and Optimizers ==========
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)

# ========== Utility Functions ==========
def denorm(x):
    """Convert output images from [-1, 1] to [0, 1]"""
    return (x + 1) / 2

def save_fake_images(epoch):
    """Save generated fake images as a PNG file"""
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = G(z).view(-1, 1, 28, 28)
    fake_images = denorm(fake_images)
    grid = torchvision.utils.make_grid(fake_images, nrow=10, normalize=True)
    npimg = grid.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f"Epoch {epoch}")
    plt.axis("off")
    plt.savefig(f"fake_images_epoch_{epoch}.png")
    plt.close()

# ========== Training Loop ==========
print("Starting GAN training...")
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # Flatten images into vectors
        images = images.reshape(batch_size, -1).to(device)

        # Create labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ======== Train Discriminator ========
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ======== Train Generator ========
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)  # Fool the discriminator

        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Print loss occasionally
        if (i+1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
                  f"D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}")

    # Save sample images
    if (epoch+1) % 20 == 0:
        save_fake_images(epoch+1)
