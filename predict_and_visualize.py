import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load trained model
model = tf.keras.models.load_model('cifar10_model.h5')

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize data
x_test = x_test / 255.0

# Make predictions
predictions = model.predict(x_test[:10])

# Visualize predictions in HD quality
plt.figure(figsize=(19.2, 10.8), dpi=300)  # High resolution for HD quality (1920x1080)
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])  # Display image
    plt.title(f"Pred: {class_names[np.argmax(predictions[i])]}\nTrue: {class_names[np.argmax(y_test[i])]}", fontsize=16)
    plt.axis('off')

plt.tight_layout()
plt.show()

# Optionally save the HD quality image
plt.figure(figsize=(19.2, 10.8), dpi=300)  # Ensure high DPI when saving
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"Pred: {class_names[np.argmax(predictions[i])]}\nTrue: {class_names[np.argmax(y_test[i])]}", fontsize=16)
    plt.axis('off')

plt.tight_layout()
plt.savefig('output/predictions_HD_quality.png', dpi=300)  # Save HD quality image
