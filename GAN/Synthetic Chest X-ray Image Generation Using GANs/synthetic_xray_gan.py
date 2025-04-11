# 📦 Core Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from glob import glob
from sklearn.utils import shuffle

# 📈 Deep Learning Libraries
from tensorflow.keras import layers, models

# 🛠️ Utilities
import streamlit as st


# -----------------------------
# USER INPUTS (ADJUST PATH)
# -----------------------------

st.title("Chest X-Ray GAN Image Generator")

base_path = st.text_input("Enter path to chest_xray directory:", "chest_xray")
train_path = os.path.join(base_path, 'train')
test_path = os.path.join(base_path, 'test')
val_path = os.path.join(base_path, 'val')


# -----------------------------
# IMAGE PREPROCESSING FUNCTION
# -----------------------------
def load_and_preprocess_images(data_dir, img_size=(28, 28), limit=None):
    images = []
    labels = []
    label_map = {'NORMAL': 0, 'PNEUMONIA': 1}

    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            img_files = os.listdir(class_dir)
            if limit:
                img_files = img_files[:limit]
            for img_file in img_files:
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img_resized = cv2.resize(img, img_size)
                img_resized = (img_resized / 127.5) - 1.0
                images.append(img_resized)
                labels.append(label_map[label])
    return np.array(images), np.array(labels)


# -----------------------------
# LOAD & PREPARE DATA
# -----------------------------
IMAGE_SIZE = (28, 28)
train_images, train_labels = load_and_preprocess_images(train_path, limit=1000)
test_images, test_labels = load_and_preprocess_images(test_path)

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

def preprocess_and_batch_data(images, batch_size=32):
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(images)
    return dataset.shuffle(buffer_size=10000).batch(batch_size, drop_remainder=True)

train_dataset = preprocess_and_batch_data(train_images)
test_dataset = preprocess_and_batch_data(test_images)

# -----------------------------
# GAN MODEL DEFINITIONS
# -----------------------------
def build_generator(noise_dim=100):
    model = models.Sequential(name="Generator")
    model.add(layers.Dense(7*7*128, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 128)))
    model.add(layers.Conv2DTranspose(64, 5, 1, 'same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(32, 5, 2, 'same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, 5, 2, 'same', use_bias=False, activation='tanh'))
    return model

def build_discriminator():
    model = models.Sequential(name="Discriminator")
    model.add(layers.Conv2D(64, 5, 2, 'same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, 5, 2, 'same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

generator = build_generator()
discriminator = build_discriminator()

# -----------------------------
# LOSS & OPTIMIZERS
# -----------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4, clipvalue=1.0)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, clipvalue=1.0)

# -----------------------------
# TRAINING STEP
# -----------------------------
@tf.function
def train_step(real_images, noise_dim):
    batch_size = real_images.shape[0]
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)
        disc_loss = discriminator_loss(real_output, fake_output)
        gen_loss = generator_loss(fake_output)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return disc_loss, gen_loss

# -----------------------------
# IMAGE GENERATION
# -----------------------------
def generate_and_show_images(generator, noise_dim=100, num_images=5):
    noise = tf.random.normal([num_images, noise_dim])
    generated_images = generator(noise, training=False)
    generated_images = (generated_images + 1) / 2
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(generated_images[i, :, :, 0], cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)

# -----------------------------
# STREAMLIT CONTROL PANEL
# -----------------------------
if st.button("Generate Chest X-ray Images"):
    st.write("Training model and generating images...")

    EPOCHS = 1  # Set low to avoid long wait times
    noise_dim = 100

    for epoch in range(EPOCHS):
        for image_batch in train_dataset.take(1):
            disc_loss, gen_loss = train_step(image_batch, noise_dim)
    generate_and_show_images(generator, noise_dim=noise_dim)

