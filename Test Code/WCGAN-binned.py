import numpy as np
import random
from math import floor
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import gc
print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices[0])
tf.random.set_seed(1234)


filenames = {
    "herwig": "GAN-data/events_anomalydetection_DelphesHerwig_qcd_features.h5",
    "pythiabg": "GAN-data/events_anomalydetection_DelphesPythia8_v2_qcd_features.h5",
    "pythiasig": "GAN-data/events_anomalydetection_DelphesPythia8_v2_Wprime_features.h5"
}

datatypes = ["herwig", "pythiabg", "pythiasig"]
features = ["px", "py", "pz", "m", "tau1", "tau2", "tau3"]

train_features = ["tau21j1", "tau21j2"] # Can be flexibly changed to suit GAN needs
dim = len(train_features)
assert (dim == 2)

def load_data(datatype, stop = None):
    input_frame = pd.read_hdf(filenames[datatype], stop = stop)
    output_frame = input_frame.copy()
    for feature in features:
        output_frame[feature + "j1"] = (input_frame["mj1"] >= input_frame["mj2"])*input_frame[feature + "j1"] + (input_frame["mj1"] < input_frame["mj2"])*input_frame[feature + "j2"]
        output_frame[feature + "j2"] = (input_frame["mj1"] >= input_frame["mj2"])*input_frame[feature + "j2"] + (input_frame["mj1"] < input_frame["mj2"])*input_frame[feature + "j1"]
    del input_frame
    gc.collect()
    output_frame["mjdelta"] = output_frame["mj1"] - output_frame["mj2"]
    output_frame["ej1"] = np.sqrt(output_frame["mj1"]**2 + output_frame["pxj1"]**2 + output_frame["pyj1"]**2 + output_frame["pzj1"]**2)
    output_frame["ej2"] = np.sqrt(output_frame["mj2"]**2 + output_frame["pxj2"]**2 + output_frame["pyj2"]**2 + output_frame["pzj2"]**2)
    output_frame["ejj"] = output_frame["ej1"] + output_frame["ej2"]
    output_frame["pjj"] = np.sqrt((output_frame["pxj1"] + output_frame["pxj2"])**2 + (output_frame["pyj1"] + output_frame["pyj2"])**2 + (output_frame["pzj1"] + output_frame["pzj2"])**2)
    output_frame["mjj"] = np.sqrt(output_frame["ejj"]**2 - output_frame["pjj"]**2)
    output_frame["tau21j1"] = output_frame["tau2j1"] / output_frame["tau1j1"]
    output_frame["tau32j1"] = output_frame["tau3j1"] / output_frame["tau2j1"]
    output_frame["tau21j2"] = output_frame["tau2j2"] / output_frame["tau1j2"]
    output_frame["tau32j2"] = output_frame["tau3j2"] / output_frame["tau2j2"]
    return output_frame


# Network hyperparameters
BATCH_SIZE = 1024
EPOCHS = 100
BINS = 28
assert (BINS % 4 == 0), "Ensure BINS is a multiple of 4"

# Adam hyperparameters as recommended by arXiv:1704.00028
LEARNING_RATE = 1e-4
BETA_1 = 0
BETA_2 = 0.9

# WGAN hyperparameters
N_CRITIC = 5
C_LAMBDA = 10

PLOTINTERVAL = 10
PREFIX = "img/{:.0f}D-{}bins-{}batchsize-".format(dim, BINS, BATCH_SIZE)


df = load_data("herwig")


# Ensures all batches have same size

df.dropna(inplace = True)
df.drop([i for i in range(df.shape[0] % (BATCH_SIZE * 4))], inplace = True)
df.reset_index(drop = True, inplace = True)
df = df.astype('float32')


np_reduced = np.array(df[train_features])
del df
gc.collect()


# Binning

data_range = np.array([np_reduced.min(axis = 0), np_reduced.max(axis = 0)]).T

for i in range(2):
    print("{} range: {} to {}".format(train_features[i], data_range[i,0], data_range[i,1]))

def binNum(val, min, max, bins = BINS):
    if val == max:
        return bins - 1
    return int(floor(bins * (val - min) / (max - min)))

np_transformed = np.zeros((np_reduced.shape[0], BINS, BINS, 1), dtype = "float32")

for i in tqdm(range(np_reduced.shape[0])):
    np_transformed[i, binNum(np_reduced[i,0], data_range[0,0], data_range[0,1]), binNum(np_reduced[i,1], data_range[1,0], data_range[1,1]), 0] = 1

def inverseMat(mat, bins = BINS):
    x, y, _ = np.unravel_index(np.argmax(mat, axis = None), mat.shape)
    val1 = data_range[0,0] + (x + 0.5) * (data_range[0, 1] - data_range[0, 0]) / bins
    val2 = data_range[1,0] + (y + 0.5) * (data_range[1, 1] - data_range[1, 0]) / bins
    return (val1, val2)

def featureList(matList, bins = BINS):
    np_retval = np.zeros((matList.shape[0], 2))
    for i in range(matList.shape[0]):
        np_retval[i] = np.array(inverseMat(matList[i], bins))
    return np_retval

X_train, X_test = train_test_split(np_transformed, test_size = 0.25, random_state = 1234)
len_dataset = int(X_train.shape[0] / BATCH_SIZE)
len_testset = int(X_test.shape[0] / BATCH_SIZE)
print("Dataset consists of {} batches of {} samples each, total {} samples".format(len_dataset, BATCH_SIZE, len_dataset * BATCH_SIZE))
print("Testset consists of {} batches of {} samples each, total {} samples".format(len_testset, BATCH_SIZE, len_testset * BATCH_SIZE))

realdata = featureList(X_train)

plt.clf()
plt.title("Herwig Background Features")
plt.ylabel("$\\tau_{21J_2}$")
plt.xlabel("$\\tau_{21J_1}$")

plt.hist2d(realdata[:,0], realdata[:,1], bins = BINS, range = data_range, density = True, alpha = 0.5, cmap = 'Oranges', label = "Herwig Background")
plt.colorbar()
plt.savefig("{}trainset-img.png".format(PREFIX))

train_dataset = tf.data.Dataset.from_tensor_slices(np.array(X_train)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(np.array(X_test)).batch(BATCH_SIZE)


def make_generator_model():
    quarter_bins = int(BINS / 4)

    model = tf.keras.Sequential()
    model.add(layers.Dense(quarter_bins*quarter_bins*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((quarter_bins, quarter_bins, 256)))
    assert model.output_shape == (None, quarter_bins, quarter_bins, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, quarter_bins, quarter_bins, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 2*quarter_bins, 2*quarter_bins, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='softmax'))
    assert model.output_shape == (None, BINS, BINS, 1)

    return model


generator = make_generator_model()


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[BINS, BINS, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


discriminator = make_discriminator_model()


generator.summary()


discriminator.summary()


@tf.function
def gradient_penalty(real, fake, epsilon): 
    # mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_images = fake + epsilon * (real - fake)
    with tf.GradientTape() as tape:
        tape.watch(mixed_images) 
        mixed_scores = discriminator(mixed_images)
        
    gradient = tape.gradient(mixed_scores, mixed_images)[0]
    
    gradient_norm = tf.norm(gradient)
    penalty = tf.math.reduce_mean((gradient_norm - 1)**2)
    return penalty


@tf.function
def discriminator_loss(real_output, fake_output, gradient_penalty):
    loss = tf.math.reduce_mean(fake_output) - tf.math.reduce_mean(real_output) + C_LAMBDA * gradient_penalty
    return loss


@tf.function
def generator_loss(fake_output):
    gen_loss = -1. * tf.math.reduce_mean(fake_output)
    return gen_loss


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)


# Convert tensor to float for loss function plotting
def K_eval(x):
    try:
        return K.get_value(K.to_dense(x))
    except:
        eval_fn = K.function([], [x])
        return eval_fn([])[0]


@tf.function
def train_step_generator():
  noise = tf.random.normal([BATCH_SIZE, 100])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)
    fake_output = discriminator(generated_images, training=True)
    gen_loss = generator_loss(fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  
  return gen_loss


@tf.function
def train_step_discriminator(images):
  noise = tf.random.normal([BATCH_SIZE, 100])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    epsilon = tf.random.uniform([BATCH_SIZE, BINS, BINS, 1])
    gp = gradient_penalty(images, generated_images, epsilon)
    
    disc_loss = discriminator_loss(real_output, fake_output, gp)

  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  
  return disc_loss


@tf.function
def evaluate_generator():
    noise = tf.random.normal([BATCH_SIZE, 100])
    generated_images = generator(noise, training=False)

    fake_output = discriminator(generated_images, training=False)

    gen_loss = generator_loss(fake_output)

    return gen_loss


@tf.function
def evaluate_discriminator(images):
    noise = tf.random.normal([BATCH_SIZE, 100])
    generated_images = generator(noise, training=False)

    real_output = discriminator(images, training=False)
    fake_output = discriminator(generated_images, training=False)

    epsilon = tf.random.uniform([BATCH_SIZE, BINS, BINS, 1])
    gp = gradient_penalty(images, generated_images, epsilon)
    
    disc_loss = discriminator_loss(real_output, fake_output, gp)

    return disc_loss



def graph_gan(generator, epoch):
    fakedata = featureList(generator(tf.random.normal((5000, 100)), training=False))

    plt.clf()

    f, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
    f.suptitle("Features")

    ax1.set_title("N-subjettiness ratio")
    ax1.set_ylabel("Normalized to Unity")
    ax1.set_xlabel("$\\tau_{21J_1}$")
    ax1.hist(realdata[:,0], bins = BINS, range = (data_range[0,0], data_range[0,1]), color = "tab:orange", alpha = 0.5, label = "Herwig Background", density = True)
    ax1.hist(fakedata[:,0], bins = BINS, range = (data_range[0,0], data_range[0,1]), color = "tab:blue", histtype = "step", label = "GAN", density = True)
    ax1.legend(loc="upper right")

    ax2.set_title("N-subjettiness ratio")
    ax2.set_ylabel("Normalized to Unity")
    ax2.set_xlabel("$\\tau_{21J_2}$")
    ax2.hist(realdata[:,1], bins = BINS, range = (data_range[1,0], data_range[1,1]), color = "tab:orange", alpha = 0.5, label = "Herwig Background", density = True)
    ax2.hist(fakedata[:,1], bins = BINS, range = (data_range[1,0], data_range[1,1]), color = "tab:blue", histtype = "step", label = "GAN", density = True)
    
    plt.savefig("{}epoch{}-gan.png".format(PREFIX, epoch))

def graph_image(generator, epoch):
    fakedata = featureList(generator(tf.random.normal((5000, 100)), training=False))
    plt.title("Herwig Background Features")
    plt.ylabel("$\\tau_{21J_2}$")
    plt.xlabel("$\\tau_{21J_1}$")

    plt.clf()

    # plt.hist2d(realdata[:,0], realdata[:,1], bins = BINS, range = data_range, density = True, alpha = 0.5, cmap = 'Oranges', label = "Herwig Background")
    plt.hist2d(fakedata[:,0], fakedata[:,1], bins = BINS, range = data_range, density = True, alpha = 1.0, cmap = 'Blues', label = "GAN")

    plt.colorbar()
    # plt.legend()

    plt.savefig("{}epoch{}-img.png".format(PREFIX, epoch))



train_gen_losses = []
train_disc_losses = []
test_gen_losses = []
test_disc_losses = []


def graph_losses(epoch):
    plt.clf()

    f, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)

    f.suptitle("Loss Functions")

    ax1.set_title("Generator Loss")
    ax1.set_ylabel("Wasserstein Loss")
    ax1.set_xlabel("Epoch")
    ax1.plot(train_gen_losses, 'b', label = "Training loss")
    ax1.plot(test_gen_losses, 'r', label = "Validation loss")
    ax1.legend(loc="upper right")

    ax2.set_title("Discriminator Loss")
    ax2.set_ylabel("Wasserstein Loss")
    ax2.set_xlabel("Epoch")
    ax2.plot(train_disc_losses, 'b', label = "Training loss")
    ax2.plot(test_disc_losses, 'r', label = "Validation loss")

    plt.savefig("{}epoch{}-genloss.png".format(PREFIX, epoch))

def train(dataset, testset, epochs, n_critic):
  for epoch in tqdm(range(epochs)):
    print_losses = False # ((epoch + 1) % 10 == 0)
    draw_outputs = ((epoch + 1) % PLOTINTERVAL == 0)

    train_gen_loss = 0
    train_disc_loss = 0

    test_gen_loss = 0
    test_disc_loss = 0

    # Training

    for batchnum, image_batch in enumerate(dataset):
      if random.random() < 1 / n_critic:
        train_gen_loss += K_eval(train_step_generator()) / len_dataset * n_critic
      train_disc_loss += K_eval(train_step_discriminator(image_batch)) / len_dataset
    
    train_gen_losses.append(train_gen_loss)
    train_disc_losses.append(train_disc_loss)

    # Evaluation

    for batchnum, test_batch in enumerate(testset):
      test_gen_loss += K_eval(evaluate_generator()) / len_testset
      test_disc_loss += K_eval(evaluate_discriminator(test_batch)) / len_testset

    test_gen_losses.append(test_gen_loss)
    test_disc_losses.append(test_disc_loss)

    # Logging

    if print_losses:
      print()
      print("Epoch " + str(epoch + 1) + ":")
      print()
      print("Generator training loss: " + str(train_gen_losses[-1]))
      print("Discriminator training loss: " + str(train_disc_losses[-1]))
      print()
      print("Generator validation loss: " + str(test_gen_losses[-1]))
      print("Discriminator validation loss: " + str(test_disc_losses[-1]))

    if draw_outputs:
      print()
      print("Epoch " + str(epoch + 1) + ":")
      graph_image(generator, epoch + 1)
      graph_gan(generator, epoch + 1)
      graph_losses(epoch + 1)


train(train_dataset, test_dataset, EPOCHS, N_CRITIC)