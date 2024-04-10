from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


model = EfficientNetB0(include_top=False, weights="imagenet")


IMG_SIZE = 224

# ///////////////////////////////////////////////////

batch_size = 64

data_dir = "/home/erfan/Documents/Projects/AtworkTasks/dataset/PPT/cavity_images"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size,
)


test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size,
)
# ///////////////////////////////////////////////////


normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))


# ///////////////////////////////////////////////////


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)
# ///////////////////////////////////////////////////


for image, label in train_ds.take(0):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        # aug_img = img_augmentation(tf.expand_dims(image, axis=0))
        aug_img = img_augmentation(image)
        plt.imshow(aug_img[0].numpy().astype("uint8"))
        plt.title("{}".format(label))
        plt.axis("off")

# ///////////////////////////////////////////////////


def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    # outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="relu", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer,
        # loss="categorical_crossentropy",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model(num_classes=NUM_CLASSES)

epochs = 25  # @param {type: "slider", min:8, max:80}
hist = model.fit(train_ds, epochs=epochs, validation_data=train_ds, verbose=2)
plot_hist(hist)


# !mkdir -p saved_model
model.save("/home/erfan/Documents/Projects/AtworkTasks/PPT/saved_model/main")


try:
    cavity_url = "/home/erfan/Documents/Projects/AtworkTasks/dataset/PPT/cavity_images/F20_20_horizontal/1442-1.jpg"
    cavity_url = "/home/erfan/Documents/Projects/AtworkTasks/dataset/PPT/cavity_images/M20_100_horizontal/M20_100_downward506.jpg"

    image = tf.keras.preprocessing.image.load_img(cavity_url)
    image = tf.image.resize(
        image,
        [224, 224],
        # method=ResizeMethod.BILINEAR,
        preserve_aspect_ratio=False,
        antialias=False,
        name=None,
    )
except:
    print("predection error")

""" 



first_image = image_batch[0]



class_names = train_ds.class_names
print(class_names)


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


## Standardize the data

# print(np.min(first_image), np.max(first_image))



for image, label in train_ds.take(0):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        # aug_img = img_augmentation(tf.expand_dims(image, axis=0))
        aug_img = img_augmentation(image)
        plt.imshow(aug_img[0].numpy().astype("uint8"))
        plt.title("{}".format(label))
        plt.axis("off")

num_classes = len(class_names)

model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(224, 224, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

# model.compile(
#    optimizer="adam",
#    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#    metrics=["accuracy"],
# )



model = tf.keras.models.load_model(
    "/home/erfan/Documents/Projects/AtworkTasks/PPT/PPT_bbox_classifier/saved_model/my_model"
)
model.summary()

epochs = 10
# history = model.fit(train_ds, validation_data=test_ds, epochs=epochs)

# acc = history.history["accuracy"]
# val_acc = history.history["val_accuracy"]

# loss = history.history["loss"]
# val_loss = history.history["val_loss"]

epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label="Training Accuracy")
# plt.plot(epochs_range, val_acc, label="Validation Accuracy")
# plt.legend(loc="lower right")
# plt.title("Training and Validation Accuracy")
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label="Training Loss")
# plt.plot(epochs_range, val_loss, label="Validation Loss")
# plt.legend(loc="upper right")
# plt.title("Training and Validation Loss")
# plt.show()


model.summary()
len(model.trainable_variables)

cavity_url = "/home/erfan/Documents/Projects/AtworkTasks/dataset/PPT/cavity_images/F20_20_horizontal/1442-1.jpg"
cavity_url = "/home/erfan/Documents/Projects/AtworkTasks/dataset/PPT/cavity_images/M20_100_horizontal/M20_100_downward506.jpg"

image = tf.keras.preprocessing.image.load_img(cavity_url)
image = tf.image.resize(
    image,
    [224, 224],
    # method=ResizeMethod.BILINEAR,
    preserve_aspect_ratio=False,
    antialias=False,
    name=None,
)

img_array = tf.keras.utils.img_to_array(image)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
        class_names[np.argmax(score)], 100 * np.max(score)
    )
) """
