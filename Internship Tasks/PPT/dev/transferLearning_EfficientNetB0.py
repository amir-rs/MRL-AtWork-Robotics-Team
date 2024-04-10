#!/usr/bin/env python
# coding: utf-8

# In[4]:


from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# In[119]:


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# In[120]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[121]:


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


# In[122]:


model = EfficientNetB0(include_top=False, weights="imagenet")
#model.summary()


# In[123]:


IMG_SIZE = 224
batch_size = 64

data_dir = "/tf/dataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    #label_mode='categorical',
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    #label_mode='categorical',
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size,
)
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)



# In[124]:


print(train_ds)
print(test_ds)

for image_batch, labels_batch in train_ds.take(1):
  print(image_batch.shape, labels_batch.shape)
  
len(train_ds)  


# In[125]:


import matplotlib.pyplot as plt

plt.figure(figsize=(7, 7))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# In[126]:


for image_batch, labels_batch in train_ds.take(1):
  print(image_batch.shape)
  print(labels_batch[0].numpy())
  


# In[127]:


train_ds, test_ds.take(0)


# In[128]:


# One-hot / categorical encoding
def input_preprocess(image, label):
    label = tf.one_hot(tf.cast(label, tf.dtypes.int32), tf.cast(NUM_CLASSES, tf.dtypes.int32))
    print(image, label)
    return image, label


train_ds = train_ds.map(input_preprocess)



test_ds = test_ds.map(input_preprocess)

train_ds, test_ds


# In[129]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


# In[130]:


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


# In[140]:


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
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = img_augmentation(inputs)
x


# In[141]:


model = build_model(NUM_CLASSES)
#model.summary()

epochs = 100  # @param {type: "slider", min:8, max:80}
hist = model.fit(train_ds, epochs=epochs, validation_data=test_ds)
plot_hist(hist)


# In[142]:


# !mkdir -p saved_model

#model.save("/home/erfan/Documents/Projects/AtworkTasks/PPT/saved_model/Pretrained_EfficientNetB0")
model.save("/tf/dev/PPT_bbox_classifier/saved_model/Pretrained_EfficientNetB0")



# In[143]:


model.save("/tf/dev/PPT_bbox_classifier/saved_model/Pretrained_EfficientNetB0.h5")


# In[111]:


tf.saved_model.save(model, "/tf/dev/PPT_bbox_classifier/saved_model/Pretrained_EfficientNetB0")


# In[147]:


#cavity_url = "/home/erfan/Documents/Projects/AtworkTasks/dataset/PPT/cavity_images/F20_20_horizontal/1442-1.jpg"
cavity_url = "/tf/dataset/M20_vertical/M20_vertical_010.jpg"

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
print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))


# In[114]:





# In[ ]:





# In[ ]:




