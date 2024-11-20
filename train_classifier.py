import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model 
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Load processed data
with open('processed_data.pickle', 'rb') as f:
    train_data = pickle.load(f)
    test_data = pickle.load(f)

x_train = train_data['features']
y_train = train_data['labels']
x_test = test_data['features']
y_test = test_data['labels']

# Load the trained model
model = load_model("C:\\Users\\ayaan\\Downloads\\converted_keras (1)\\keras_model.h5")

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=len(set(y_train)))
y_test = to_categorical(y_test, num_classes=len(set(y_test)))

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test Accuracy: {test_acc * 100}%")

# Save the model after training
model.save('gesture_classifier.h5')
print("Model trained and saved.")