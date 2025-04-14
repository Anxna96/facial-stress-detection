
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import cv2
import mediapipe as mp

# Enable TensorFlow resource variables
tf.compat.v1.enable_resource_variables()

# Paths to dataset
train_dir = "C:/Users/yetij/Desktop/Stress/FER2013CK_filtered/train"
test_dir = "C:/Users/yetij/Desktop/Stress/FER2013CK_filtered/test"

# Dataset configurations
batch_size = 32
img_size = (48, 48)  # Reduce to 48x48 for faster processing
emotions_to_stress_mapping = {
    'angry': 1, 'fear': 1, 'disgust': 1,  # Stress
    'happy': 0, 'neutral': 0, 'surprise': 0, 'sad': 0  # No Stress
}

# Function to preprocess dataset
def preprocess_dataset(directory):
    raw_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode="int"
    )
    class_names = raw_dataset.class_names
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_dataset = raw_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Map integer labels to stress/no-stress
    def map_to_stress_labels(images, labels):
        mapped_labels = tf.constant(
            [emotions_to_stress_mapping[class_names[label]] for label in labels.numpy()], dtype=tf.int32)
        return images, mapped_labels

    stress_dataset = normalized_dataset.map(lambda x, y: tf.py_function(
        func=map_to_stress_labels, inp=[x, y], Tout=(tf.float32, tf.int32)
    ))
    stress_dataset = stress_dataset.map(lambda x, y: (
        tf.ensure_shape(x, (None, 48, 48, 1)),
        tf.ensure_shape(y, (None,))
    ))
    return stress_dataset

# Load and preprocess datasets
train_dataset = preprocess_dataset(train_dir)
test_dataset = preprocess_dataset(test_dir)

# Build a simple CNN model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
model = create_model()
history = model.fit(train_dataset, epochs=20, validation_data=test_dataset)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")

# Save the model
model.save("stress_detection_model.h5")

# Function to predict from a supplied image of the directory
def predict_stress_from_image(image_path, img_size=(48, 48)):
    
    # Verify that the image exists
    if not os.path.exists(image_path):
        print(f"A imagem no caminho '{image_path}' não foi encontrada.")
        return

    # Upload image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)  # Black and white image
    image = tf.image.resize(image, img_size)  # Redimension
    image = image / 255.0  # Normalize image to [0, 1]

    image = tf.reshape(image, (1, img_size[0], img_size[1], 1))  

    # Make the prediction
    prediction = model.predict(image)
    threshold = 0.41  
    stress_label = "Stress" if prediction[0][0] >= threshold else "No Stress"

    # Show the image and the expected result
    plt.imshow(tf.squeeze(image), cmap='gray')  
    plt.title(f"Predicted: {stress_label}")
    plt.axis('off')
    plt.show()

    return stress_label

# Test a specific image
image_path = "C:/Users/yetij/Desktop/Stress/FER2013CK_filtered/test/angry/PublicTest_83042442.jpg"  # Substitute for each image
predicted_stress = predict_stress_from_image(image_path)
print(f"Predicted Stress Status: {predicted_stress}")


# Evaluate with metrics
y_true = []
y_pred = []
   
threshold = 0.41 # Adjust threshold for stress classification
for images, labels in test_dataset:
    y_true.extend(labels.numpy())
    predictions = model.predict(images)
    y_pred.extend((predictions >= threshold).astype(int).flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# Classification report
print("Classification Report:")
class_names = ['No Stress', 'Stress']
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Real-Time Detection with MediaPipe
def real_time_detection():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    video_capture = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Preprocess the detected face
                    face = frame[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (48, 48))
                    face_grayscale = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                    face_normalized = face_grayscale.astype('float32') / 255.0
                    face_reshaped = np.expand_dims(face_normalized, axis=(0, -1))

                    # Predict stress
                    prediction = model.predict(face_reshaped)
                    stress_label = 'Stress' if prediction[0][0] > threshold else 'No Stress'

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, stress_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Show the frame
            cv2.imshow('Real-Time Stress Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()

# Run Real-Time Detection
real_time_detection()
