import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model_path = "ssd_mobilenet_v1_coco/frozen_inference_graph.pb"
with tf.io.gfile.GFile(model_path, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Load the model into TensorFlow session
with tf.compat.v1.Session() as sess:
    tf.import_graph_def(graph_def, name="")

# Load image
image = cv2.imread("test_image.jpg")
height, width, _ = image.shape

# Resize and preprocess
blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)

# Run the model
output_tensors = ["num_detections:0", "detection_scores:0", "detection_boxes:0", "detection_classes:0"]
outputs = sess.run([sess.graph.get_tensor_by_name(tensor) for tensor in output_tensors], feed_dict={"image_tensor:0": blob})

# Extract results
num_detections = int(outputs[0][0])
for i in range(num_detections):
    score = float(outputs[1][0][i])
    bbox = [float(v) for v in outputs[2][0][i]]

    if score > 0.3:  # Confidence threshold
        x, y, right, bottom = int(bbox[1] * width), int(bbox[0] * height), int(bbox[3] * width), int(bbox[2] * height)
        cv2.rectangle(image, (x, y), (right, bottom), (125, 255, 51), thickness=2)

cv2.imshow("Object Detection", image)
cv2.waitKey(0)