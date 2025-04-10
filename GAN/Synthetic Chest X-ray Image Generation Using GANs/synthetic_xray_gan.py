# 📦 Core Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from glob import glob
from sklearn.utils import shuffle
import imageio
import time
from IPython.display import clear_output

# 📈 Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, BatchNormalization
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Dense, Reshape, Flatten, BatchNormalization, Conv2D, Conv2DTranspose, LeakyReLU, Dropout

# 🛠️ Utilities
from tqdm.notebook import tqdm


# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

!ls "/content/drive/MyDrive"

# sTEP 2: Define path to the chest_xray folder

import os

base_path = '/content/drive/My Drive/Projects/chest_xray'
train_path = os.path.join(base_path, 'train')
test_path = os.path.join(base_path, 'test')
val_path = os.path.join(base_path, 'val')

# Check NORMAL & PNEUMONIA folder paths
print("NORMAL Folder exists:", os.path.exists(os.path.join(train_path, "NORMAL")))
print("PNEUMONIA Folder exists:", os.path.exists(os.path.join(train_path, "PNEUMONIA")))

# Confirming if files exists in the folder

print('There are:',os.listdir(train_path), 'folders in the train folder')
print('There are:',os.listdir(test_path), 'folders in the test folder')
print('There are:',os.listdir(val_path), 'folders in the val folder')

# List Sample Files from Each Folders
print("NORMAL images in train:", os.listdir(os.path.join(train_path, 'NORMAL'))[:3])
print("PNEUMONIA images in train:", os.listdir(os.path.join(train_path, 'PNEUMONIA'))[:3])


# Set the target image size
IMAGE_SIZE = (28, 28)


# Defining the image Preprocessing function
def load_and_preprocess_images(data_dir, img_size=(28, 28), limit=None):
    images = []
    labels = []
    label_map = {'NORMAL': 0, 'PNEUMONIA': 1}  # Label mapping

  # Looping Through the Directories
    for label in os.listdir(data_dir):  # Iterate through class directories
        class_dir = os.path.join(data_dir, label)

        if os.path.isdir(class_dir):  # Check if it's a directory
            img_files = os.listdir(class_dir)
            if limit:
                img_files = img_files[:limit]  # Apply the limit if specified
            for img_file in img_files:
                img_path = os.path.join(class_dir, img_file)

                # Read image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue

                # Resize the image
                img_resized = cv2.resize(img, img_size)

                # Normalize the image to the range [-1, 1]
                img_resized = (img_resized / 127.5) - 1.0

                images.append(img_resized)
                labels.append(label_map[label])

    return np.array(images), np.array(labels)


# Paths for the training, testing, and validation data
base_path = '/content/drive/My Drive/Projects/chest_xray'
train_path = os.path.join(base_path, 'train')
test_path = os.path.join(base_path, 'test')
val_path = os.path.join(base_path, 'val')

# Load training and test data
train_images, train_labels = load_and_preprocess_images(train_path, limit=1000)
test_images, test_labels = load_and_preprocess_images(test_path)



# Print shapes of the processed datasets
print("Train Images Shape:", train_images.shape)
print("Train Labels Shape:", train_labels.shape)
print("Test Images Shape:", test_images.shape)
print("Test Labels Shape:", test_labels.shape)


# Add channel dimension for CNNs: (N, 28, 28, 1)
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)


print("Train Images Shape with channel:", train_images.shape)
print("Test Images Shape with channel:", test_images.shape)

# Function to define batching

def preprocess_and_batch_data(images, batch_size=32):
    """ Preprocess data for GAN and return batched dataset. """
    # Normalize the images (already done in preprocessing above)
    images = tf.convert_to_tensor(images, dtype=tf.float32)

    # Create a batched dataset
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size, drop_remainder=True)

    return dataset

# Prepare training and testing datasets
train_dataset = preprocess_and_batch_data(train_images)
test_dataset = preprocess_and_batch_data(test_images)

# Show example of the dataset structure
for batch in train_dataset.take(1):
    print(f"Batch shape: {batch.shape}")


#Generator Architecture
from tensorflow.keras.layers import Dense, Reshape, Flatten, BatchNormalization, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras import layers, models

def build_generator(noise_dim=100):
    model = models.Sequential(name="Generator")

    model.add(layers.Dense(7*7*128, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 128)))  # Reshape into a small image
    assert model.output_shape == (None, 7, 7, 128)

    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', use_bias=False, activation='tanh'))

    return model

# Instantiate the generator
generator = build_generator()
generator.summary()



# Discriminator Architecture

def build_discriminator():
    model = models.Sequential(name="Discriminator")

    model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same',
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))  # Outputs a single probability

    return model

# Instantiate the discriminator
discriminator = build_discriminator()
discriminator.summary()


# Loss functions for the generator and discriminatorLoss Functions
# Binary Cross-Entropy Loss:
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)



# Discriminator Loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)   # Real images = 1
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)  # Fake images = 0
    total_loss = real_loss + fake_loss
    return total_loss

# Generator Loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)  # Try to trick discriminator


# Optimizers for both generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Optimizers for both generator and discriminator with gradient clipping
generator_optimizer = tf.keras.optimizers.Adam(1e-4, clipvalue=1.0)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, clipvalue=1.0)


import os

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


import tensorflow as tf
import matplotlib.pyplot as plt

# Set the noise dimension (e.g., 100)
noise_dim = 100  # Add this line

# Track losses for both the discriminator and generator
def train_step(real_images):
    batch_size = real_images.shape[0] # Get batch size from real_images shape
    noise = tf.random.normal([batch_size, noise_dim])  # Generate noise for the generator
    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:

        # Forward pass through the discriminator and generator
        generated_images = generator(noise, training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Compute losses for discriminator and generator
        disc_loss = discriminator_loss(real_output, fake_output)
        gen_loss = generator_loss(fake_output)

    # Compute gradients and update weights
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return disc_loss, gen_loss, generated_images

import time
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Number of epochs (full passes through the dataset)
EPOCHS = 100

# Display images every this many epochs
DISPLAY_STEP = 10

# Track losses
generator_losses = []
discriminator_losses = []

# Define generate_and_show_images function before the training loop
def generate_and_show_images(num_images=5):
    noise = tf.random.normal([num_images, noise_dim])
    generated_images = generator(noise, training=False)

    # Normalize to [0, 1] for better visualization
    generated_images = (generated_images + 1) / 2  # Convert to [0, 1]

    # Plot the images
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(generated_images[i].numpy().squeeze(), cmap='gray')  # Remove extra dimensions for display
        plt.axis('off')
    plt.show()

# Start training
for epoch in range(EPOCHS):
    start = time.time()

    for batch in train_dataset:  # train_dataset is a tf.data.Dataset
        disc_loss, gen_loss, _ = train_step(batch)

    # Save losses for monitoring
    generator_losses.append(gen_loss)
    discriminator_losses.append(disc_loss)

    # Print progress
    print(f"Epoch {epoch+1}/{EPOCHS} completed. Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")

    # Show generated samples every few epochs
    if (epoch + 1) % DISPLAY_STEP == 0:
        generate_and_show_images(num_images=5)

print("Training finished.")

# Generate and show a few images from the generator
def generate_and_show_images(num_images=5):
    noise = tf.random.normal([num_images, noise_dim])
    generated_images = generator(noise, training=False)

    # Normalize to [0, 1] for better visualization
    generated_images = (generated_images + 1) / 2  # Convert to [0, 1]

    # Plot the images
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(generated_images[i].numpy().squeeze(), cmap='gray')  # Remove extra dimensions for display
        plt.axis('off')
    plt.show()


# Show generated images after every 10 epochs
if (epoch + 1) % DISPLAY_STEP == 0:
    generate_and_show_images(num_images=5)


# After training loop, plot losses
plt.figure(figsize=(10, 5))
plt.plot(generator_losses, label='Generator Loss')
plt.plot(discriminator_losses, label='Discriminator Loss')
plt.title('Generator vs Discriminator Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Compare synthetic vs real images
def compare_real_and_fake_images(real_images, num_images=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        # Real image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(real_images[i].squeeze(), cmap='gray') # Removed .numpy()
        plt.title('Real')
        plt.axis('off')

        # Fake image
        noise = tf.random.normal([1, noise_dim])
        generated_image = generator(noise, training=False)
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(generated_image.numpy().squeeze(), cmap='gray')
        plt.title('Fake')
        plt.axis('off')

    plt.show()

# Call the function with test images
compare_real_and_fake_images(test_images[:5])

# Plot generator and discriminator loss after training
plt.figure(figsize=(10, 5))
plt.plot(generator_losses, label="Generator Loss")
plt.plot(discriminator_losses, label="Discriminator Loss")
plt.title('Generator vs Discriminator Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import Image, display

# 🔧 Set your noise dimension (should match what you used to train the GAN)
noise_dim = 100

# 🧠 Define the function to generate and save images
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)  # Generate images from noise
    predictions = (predictions + 1) / 2.0  # Scale from [-1, 1] to [0, 1]

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')  # Show grayscale image
        plt.axis('off')

    plt.suptitle(f'Synthetic Chest X-ray Samples - Epoch {epoch}')
    file_name = f'image_at_epoch_{epoch:04d}.png'
    plt.savefig(file_name)
    plt.show()

    # Display the saved image inline (for Colab or notebooks)
    display(Image(filename=file_name))


# 📦 Generate test noise input (for 16 synthetic images in a 4x4 grid)
test_input = tf.random.normal([16, noise_dim])

# 🖼️ Generate and display images (e.g., at epoch 100)
generate_and_save_images(generator, epoch=100, test_input=test_input)


# Save the models
from keras.models import load_model
generator = load_model('generator_model.keras')
discriminator = load_model('discriminator_model.keras')

import os

# Check for saved model files
print("Generator saved:", os.path.exists("generator_model.keras"))
print("Discriminator saved:", os.path.exists("discriminator_model.keras"))

# Check for checkpoint files
print("\nCheckpoint contents:")
print(os.listdir('./training_checkpoints'))



# Load saved models
generator = tf.keras.models.load_model('generator_model.h5')
discriminator = tf.keras.models.load_model('discriminator_model.h5')


noise = tf.random.normal([1, 100])
fake_image = generator(noise, training=False)

plt.imshow(fake_image[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
plt.title("Image from Loaded Generator")
plt.axis("off")
plt.show()
