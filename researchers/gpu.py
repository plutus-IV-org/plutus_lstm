import tensorflow as tf

# Checks if any GPUs are available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))