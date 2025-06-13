import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model.json', compile=False)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted successfully to TFLite format!") 