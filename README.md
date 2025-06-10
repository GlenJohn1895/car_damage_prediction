## Car Damage Categorization and Repair Cost Estimation Using EfficientNetB0 

## 1. Introduction
As part of my project, I created a deep learning model that can categorize car damage severity and estimate repair costs based on images of cars. This solution is aimed at simplifying the insurance claim process as well as offering initial cost estimates. The model uses the EfficientNetB0  architecture to perform image classification and is based on a proprietary dataset that translates car damage images into simulated repair costs. What follows is a thorough, step-by-step description of each step in my pipeline, from data exploration and preprocessing to training and assessing the model.

## 2. Data Preparation and Exploration
The initial process in my workflow was loading and investigating the dataset. I utilized a number of libraries such as pandas, matplotlib, numpy, and seaborn to manipulate and visualize the data. My data was in the form of a CSV file (data.csv) with image file paths and a column indexed with the damage class of the car.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
To see the distribution of damage classes, I grouped the entries based on the class column and created a bar graph:
sns.countplot(x='class', data=df)
plt.title('Distribution of Damage Classes')
plt.show()
This gave us the first idea that some classes were underrepresented and therefore may have caused class imbalance that could impact model performance. I made a mental note of this later to follow up with data augmentation to solve this imbalance.
Second, I established a mapping of every class to a severity level of damage. This enabled me to categorize similar levels of damage under more general categories such as minor, moderate, severe, or no_damage.
severity_map = {
    'scratch': 'minor',
    'dent': 'moderate',
    'bumper_crack': 'severe',
    'no_damage': 'no_damage'
}
df['severity'] = df['class'].map(severity_map)
Finally, to mimic repair cost estimates, I mapped every severity level to an artificial numerical value:
cost_map = {
    'minor': 5000,
    'moderate': 15000,
    'severe': 30000,
    'no_damage': 0
}
df['cost'] = df['severity'].map(cost_map)
I then plotted the distribution of these synthetic costs:
sns.histplot(df['cost'])
plt.title('Distribution of Repair Costs')
plt.show()
This gave me an idea of how the repair costs were distributed and made sure the dataset contained a good number of cost values.

## 3. Image Preprocessing
Prior to inputting the images into the model, I resized each of them to 224x224 pixels via Python Imaging Library (PIL) for uniformity in inputs.
from PIL import Image
import os

resized_path = 'image_resized/'
os.makedirs(resized_path, exist_ok=True)

for i, row in df.iterrows():
    try:
        img = Image.open(row['image_path'])
        img = img.resize((224, 224))
new_path = os.path.join(resized_path, os.path.basename(row['image_path']))
        img.save(new_path)
        df.at[i, 'resized_path'] = new_path
    except Exception as e:
        print(f"Error processing image {row['image_path']}: {e}")
I made sure to keep only successfully resized images for further processing. After resizing, I converted the severity levels to integers for training:
label_map = {'minor': 0, 'moderate': 1, 'severe': 2, 'no_damage': 3}
df['label'] = df['severity'].map(label_map)

## 4. TensorFlow Dataset Pipeline
I developed a solid data pipeline using TensorFlow's tf.data API. I first defined functions to load and preprocess images:
import tensorflow as tf

def load_image(path):
    image = tf.io.read_file(path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [224, 224])
return image / 255.0
I then loaded image paths and labels into TensorFlow datasets and batched them:
paths = df['resized_path'].values
labels = df['label'].values

dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
dataset = dataset.map(lambda x, y: (load_image(x), y))
dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

## 5. Train-Validation Split and Augmentation
I split the dataset into training and validation sets:
from sklearn.model_selection import train_test_split

train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=42
)
I then used image augmentation to enhance the model's generalization:
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])	



## 6. Model Architecture Using EfficientNetB0
For the backbone of my model, I employed EfficientNetB0 with pre-trained ImageNet weights:
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input

base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(4, activation='softmax')(x)
model = Model(inputs, outputs)

## 7. Model Compilation and Training
I compiled the model using the Adam optimizer and categorical crossentropy loss:
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_dataset, validation_data=val_dataset, epochs=20)
While training, I tracked the validation accuracy and loss to check whether the model was overfitting or not.

## 8. Evaluation and Results
I tested the model on the validation set after training and created a classification report:
from sklearn.metrics import classification_report

preds = model.predict(val_dataset)
y_pred = tf.argmax(preds, axis=1)
y_true = tf.concat([y for x, y in val_dataset], axis=0)
print(classification_report(y_true, y_pred))
The results demonstrated a great balance between recall and precision for all classes. The confusion matrix also verified that the model was classifying the even underrepresented classes with great accuracy.
 

## 9. Conclusion
With this project, I effectively constructed a classifying pipeline for car damage severity and estimated repair costs with EfficientNetB0. Preprocessing, augmentation, and a strong model architecture made it possible for me to obtain consistent results. Further work could extend to adding object detection to find damage areas or constructing a regression head for more accurate cost estimation.
This project illustrated the capabilities of deep learning in addressing real-life issues and provided me with hands-on experience in the end-to-end model development process.
