
import zipfile

zip_path = "/content/converted_savedmodel.zip"
extract_path = "/content/converted_savedmodel"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# ✅ تحميل النموذج والملصقات
import tensorflow as tf
import numpy as np
from PIL import Image

model_path = "/content/converted_savedmodel/model.savedmodel"
labels_path = "/content/converted_savedmodel/labels.txt"

model = tf.saved_model.load(model_path)
infer = model.signatures["serving_default"]

with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ✅ تجهيز الصورة وتنفيذ التنبؤ
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

def predict(image_path):
    image_tensor = preprocess_image(image_path)
    predictions = infer(image_tensor)
    probs = list(predictions.values())[0].numpy()[0]
    class_index = np.argmax(probs)
    return labels[class_index], probs[class_index]

# ✅ تجربة النموذج 
label, confidence = predict("ولتر3.jpeg")
print(f"✅ Predicted class: {label} (Confidence: {confidence * 100:.2f}%)")