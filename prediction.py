{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Model loaded successfully! Expected input shape: (None, 128, 128, 3)\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 92ms/step\n",
      "🧪 Predicted White Blood Cell Type: Basophil\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "import cv2\n",
    "import numpy as np\n",
    "\n",
    "# ✅ Load the trained model\n",
    "model_path = r\"C:\\Users\\Ganesh prasad sahoo\\OneDrive\\Documents\\Downloads\\All DataSet\\White_blood_cells_classification\\cnn_model.h5\"\n",
    "model = tf.keras.models.load_model(model_path)\n",
    "\n",
    "# ✅ Print model input shape\n",
    "print(f\"✅ Model loaded successfully! Expected input shape: {model.input_shape}\")\n",
    "\n",
    "# Function to preprocess the image\n",
    "def preprocess_image(image_path):\n",
    "    img = cv2.imread(image_path)  # Read image\n",
    "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (if needed)\n",
    "    \n",
    "    # ✅ Resize image to match model input shape\n",
    "    target_size = model.input_shape[1:3]  # Extract (height, width) from model\n",
    "    img = cv2.resize(img, target_size)\n",
    "    \n",
    "    img = img.astype(np.float32) / 255.0  # Normalize pixel values\n",
    "    img = np.expand_dims(img, axis=0)  # Add batch dimension\n",
    "    return img\n",
    "\n",
    "# Path to the test image\n",
    "test_image_path = r\"C:\\Users\\Ganesh prasad sahoo\\OneDrive\\Documents\\Downloads\\All DataSet\\White_blood_cells_classification\\Train\\Lymphocyte\\95-5-4-1_6_2.jpg\"\n",
    "# Preprocess the image\n",
    "processed_img = preprocess_image(test_image_path)\n",
    "\n",
    "# ✅ Make a prediction\n",
    "prediction = model.predict(processed_img)\n",
    "\n",
    "# ✅ Get the predicted class index\n",
    "predicted_class = np.argmax(prediction)\n",
    "\n",
    "# ✅ Define class labels (Update according to your dataset)\n",
    "class_labels = [\"Neutrophil\", \"Eosinophil\", \"Basophil\", \"Lymphocyte\", \"Monocyte\"]\n",
    "\n",
    "print(f\"🧪 Predicted White Blood Cell Type: {class_labels[predicted_class]}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
