import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import keras as ks
from sklearn.metrics import f1_score, confusion_matrix

test_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/Fruit recognition/test',
     labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,

)

training_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/Fruit recognition/train',
     labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,

)

validation_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/Fruit recognition/validation',
     labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)
Training Model
cnn = tf.keras.models.Sequential()
CNN layers
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.5)) #drop some nuerons to avoid overfitting
cnn.add(tf.keras.layers.Dense(units=36, activation='softmax'))
Compiling
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
cnn.summary()
training_history = cnn.fit(x = training_set, validation_data = validation_set, epochs = 30)
Saving Model
cnn.save('model.h5')
training_history.history
import json
with open('training_history.json', 'w') as f:
    json.dump(training_history.history, f)
print(training_history.history.keys())
#Calculate Accuracy
print("Training set Accuracy: {} %".format(training_history.history['accuracy'][-1] * 100))

epochs = [i for i in range(1,31)]
plt.plot(epochs, training_history.history['accuracy'], color ='red')
plt.title('Visualization Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
print("Validation set Accuracy: {} %".format(training_history.history['val_accuracy'][-1] * 100))
plt.plot(epochs, training_history.history['val_accuracy'], color='blue')
plt.title('Visualization Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#Testing Model
cnn = tf.keras.models.load_model('model.h5')
#Prediction
import cv2
image_path = "/content/drive/MyDrive/Fruit recognition/validation/apple/Image_3.jpg"
image = cv2.imread(image_path)
plt.imshow(image)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()
##Testing Model

image = tf.keras.preprocessing.image.load_img(image_path, target_size = (64, 64))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])
predictions = cnn.predict(input_arr)
print(predictions[0])

result_index = np.where(predictions[0] == max(predictions[0]))
print(result_index)
plt.imshow(image)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()
#Single Prediction
print('Its a {}'.format(test_set.class_names[result_index[0][0]]))
test_set.class_names
Exporting the labels to be used in the web app
file = open("labels.txt", "w")
for i in test_set.class_names:
  file.write(i + "\n")
file.close()
with open("labels.txt") as f:
    content = f.readlines()
    print(content)
Calculating the F-1 measure

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

# Get predictions for the entire test set
y_pred = []
y_true = []

for images, labels in test_set:
    predictions = cnn.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(np.argmax(labels, axis=1))

# Calculate F1 score
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1 Score: {f1:.4f}")

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

Precision and recall
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

Class-wise Performance
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=test_set.class_names))

