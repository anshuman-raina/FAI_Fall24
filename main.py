from EmptyShelfDetectionCustomRCNN import predict_on_image
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import logging


def calculate_iou(true_bbox, pred_bbox):
    """
    Calculate Intersection over Union (IoU) between true and predicted bounding boxes.
    """
    # Calculate the intersection
    x1 = max(true_bbox[0], pred_bbox[0])
    y1 = max(true_bbox[1], pred_bbox[1])
    x2 = min(true_bbox[2], pred_bbox[2])
    y2 = min(true_bbox[3], pred_bbox[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the areas
    true_area = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])

    # Calculate the union
    union = true_area + pred_area - intersection

    # Compute IoU
    iou = intersection / union if union > 0 else 0
    return iou


def evaluate_model(model, test_data, iou_threshold=0.25, target_size=(640, 640)):
    """
    Evaluate the model on a given dataset.

    Parameters:
        model: Trained model.
        test_data: DataFrame containing test data with columns for image paths, bounding boxes, and labels.
        iou_threshold: IoU threshold to consider a bounding box prediction correct.
        target_size: Target image size for resizing during prediction.
    """
    total_samples = len(test_data)
    correct_classifications = 0
    correct_bboxes = 0

    for idx, row in test_data.iterrows():
        image_path = row['image_name']
        true_class = row['label']
        true_bbox = [
            row['bbox_x'],
            row['bbox_y'],
            row['bbox_x'] + row['bbox_width'],
            row['bbox_y'] + row['bbox_height']
        ]

        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image at {image_path}")
            continue

        original_height, original_width = image.shape[:2]
        resized_image = cv2.resize(image, target_size)
        resized_image = resized_image.astype(np.float32) / 255.0
        resized_image = np.expand_dims(resized_image, axis=0)

        # Predict bounding box and class
        pred_bbox, pred_class_probs = model.predict(resized_image)
        pred_class = np.argmax(pred_class_probs[0])
        pred_bbox = pred_bbox[0]

        # Scale predicted bbox to original image size
        pred_bbox_scaled = [
            pred_bbox[0] * original_width / target_size[0],
            pred_bbox[1] * original_height / target_size[1],
            pred_bbox[2] * original_width / target_size[0],
            pred_bbox[3] * original_height / target_size[1],
        ]

        # Classification Accuracy
        if pred_class == true_class:
            correct_classifications += 1

        # IoU for Bounding Box
        iou = calculate_iou(true_bbox, pred_bbox_scaled)
        if iou >= iou_threshold:
            correct_bboxes += 1

    # Compute metrics
    classification_accuracy = correct_classifications / total_samples
    bbox_accuracy = correct_bboxes / total_samples

    # Output the results
    print(f"Classification Accuracy: {classification_accuracy:.2f}")
    print(f"BBox Accuracy (IoU ≥ {iou_threshold}): {bbox_accuracy:.2f}")