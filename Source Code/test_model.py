import tensorflow as tf

print("Trying to load model...")

try:
    model = tf.keras.models.load_model("breast_classification_mobilenetv2(1).keras", compile=False)
    print("MODEL LOADED SUCCESSFULLY!")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

except Exception as e:
    print("\n❌ ERROR WHILE LOADING MODEL ❌")
    print(e)
