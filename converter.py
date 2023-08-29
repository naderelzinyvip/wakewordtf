import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('wake_word_model/WWD.h5')

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set optimization options (optional)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()
with open('converted_model.tflite', 'wb') as f:
    f.write(tflite_model)
