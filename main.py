import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from converter import parse_yolov5_obb
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, ConfusionMatrixDisplay

# Set up logging
logging.basicConfig(level=logging.INFO)


def main():
    try:
        # Load the trained model
        model = load_model('empty_shelf_detector_rcnn_resnet-residuals.h5', compile=False)
        logging.info("Model loaded successfully")

        # Define dataset directories
        datasets = {
        'test': ('drive/MyDrive/Dataset/EMPTY SHELF FINAL/EMPTY SHELF FINAL/test/labelTxt', 'drive/MyDrive/Dataset/EMPTY SHELF FINAL/EMPTY SHELF FINAL/test/images'),
        'valid': ('drive/MyDrive/Dataset/EMPTY SHELF FINAL/EMPTY SHELF FINAL/valid/labelTxt', 'drive/MyDrive/Dataset/EMPTY SHELF FINAL/EMPTY SHELF FINAL/valid/images'),
            }

        # Evaluate on datasets
        for split, (annotations_dir, image_folder) in datasets.items():
            logging.info(f"Evaluating on {split} dataset...")

            # Parse the annotations and load the dataset
            data = parse_yolov5_obb(annotations_dir, image_folder)
            if data.empty:
                logging.warning(f"No data found in {split} dataset!")
                continue

            # Evaluate model performance
            logging.info(f"Starting evaluation for {split} dataset...")
            evaluate_model(model, data, image_folder=image_folder)
            logging.info(f"Completed evaluation for {split} dataset.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

def compute_and_visualize_metrics(tp, fp, tn, fn, iou_scores, iou_threshold=0.2):

    print("true positives", tp)
    print("false positives", fp)
    print("true negatives", tn)
    print("false negatives", fn)
    # Compute Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    overall_accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Print Metrics
    print("\nEvaluation Metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"False Discovery Rate (FDR): {fdr:.2f}")
    print(f"True Negative Rate (TNR): {tnr:.2f}")
    print(f"False Negative Rate (FNR): {fnr:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}")

    # Plot Confusion Matrix
    cm = np.array([[tp, fn], [fp, tn]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Empty", "Not Empty"])
    disp.plot(cmap="viridis")
    plt.title("Confusion Matrix")
    plt.show()

    # Plot IoU Distribution
    plt.figure()
    plt.hist(iou_scores, bins=20, range=(0, 1), alpha=0.75)
    plt.title("IoU Distribution")
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    plt.axvline(x=iou_threshold, color='red', linestyle='--', label=f'Threshold={iou_threshold}')
    plt.legend()
    plt.show()

    # Plot Precision-Recall Curve
    y_true = [1 if score >= iou_threshold else 0 for score in iou_scores]
    y_probs = iou_scores  # Using IoU scores as prediction probabilities
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    plt.figure()
    plt.plot(recall_curve, precision_curve, marker='.')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    # Plot F1 Score as IoU Threshold Changes
    thresholds = np.linspace(0, 1, 50)
    f1_scores = [
        2 * (tp / len(iou_scores)) * (recall / len(iou_scores)) / (
                (tp / len(iou_scores)) + (recall / len(iou_scores)))
        if (tp / len(iou_scores)) + (recall / len(iou_scores)) > 0 else 0
        for threshold in thresholds
    ]
    plt.figure()
    plt.plot(thresholds, f1_scores, marker='.')
    plt.title("F1 Score vs IoU Threshold")
    plt.xlabel("IoU Threshold")
    plt.ylabel("F1 Score")
    plt.show()

    # Plot Per-Class Accuracy
    per_class_accuracy = [tp / (tp + fn), tn / (tn + fp)]
    classes = ["Empty", "Not Empty"]
    plt.figure()
    plt.bar(classes, per_class_accuracy)
    plt.title("Per-Class Accuracy")
    plt.ylabel("Accuracy")
    plt.show()

def calculate_iou(true_bbox, pred_bbox):
    """
    Calculate Intersection over Union (IoU) between true and predicted bounding boxes.
    """
    # Calculate the intersection
    x1 = max(true_bbox[0], pred_bbox[0])
    y1 = min(true_bbox[1], pred_bbox[1])
    x2 = min(true_bbox[2], pred_bbox[2])
    y2 = max(true_bbox[3], pred_bbox[3])

    intersection = abs(x2 - x1) * abs(y2 - y1)

    # Calculate the areas
    true_area = abs(true_bbox[2] - true_bbox[0]) * abs(true_bbox[3] - true_bbox[1])
    pred_area = abs(pred_bbox[2] - pred_bbox[0]) * abs(pred_bbox[3] - pred_bbox[1])

    # Calculate the union
    union = true_area + pred_area - intersection

    # Compute IoU
    iou = intersection / union if union > 0 else 0
    return iou


def evaluate_model(model, test_data, image_folder, iou_threshold=0.2):
    """
    Evaluate the model on a given dataset and visualize predictions.

    Parameters:
        model: Trained model.
        test_data: DataFrame containing test data with columns for image paths, bounding boxes, and labels.
        image_folder: Base folder containing images.
        iou_threshold: IoU threshold to consider a bounding box prediction correct.
    """
    total_samples = len(test_data)
    correct_classifications = 0
    correct_bboxes = 0

    tp = 0  # True Positives
    fp = 0  # False Positives
    tn = 0  # True Negatives
    fn = 0  # False Negatives

    iou_list = []

    # Add a counter for tracking progress
    processed_count = 0

    # Create a results directory if it doesn't exist
    results_dir = os.path.join(image_folder, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)

    for idx, row in test_data.iterrows():
        # Construct full image path
        image_path = os.path.join(image_folder, row['image_name'])
        true_label = row['label']  # 0 for empty, 1 for not empty
        true_bbox = [
            row['x_top_left'],
            row['y_top_left'],
            row['x_bottom_right'],
            row['y_bottom_right']
        ]

        # Load and preprocess image
        image = cv2.imread(image_path)

        if image is None:
            logging.warning(f"Warning: Could not load image at {image_path}")
            continue
        #Preprocess image similarly to the updated preprocess_image function
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresholded_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(thresholded_image)
        
        # Normalize image (0-1 range)
        normalized_image = clahe_image.astype(np.float32) / 255.0
        
        # Noise reduction using median blur
        preprocessed_image = cv2.medianBlur(normalized_image, 3)  # 3x3 kernel
        # Create a copy of the image for drawing
        draw_image = image.copy()
        model_input = np.expand_dims(preprocessed_image, axis=(0, -1))
        
        # Predict bounding box and label
        pred_bbox, pred_label_probs = model.predict(model_input)
        pred_label = np.argmax(pred_label_probs[0])
        original_height, original_width = image.shape[:2]
        # Extract predicted bounding box
        pred_bbox = [
        int(pred_bbox[0, 0] * original_width),
        int(pred_bbox[0, 1] * original_height),
        int(pred_bbox[0, 2] * original_width),
        int(pred_bbox[0, 3] * original_height),
    ]

        # Validate bounding box (clamp within image dimensions)
        original_height, original_width = image.shape[:2]
        pred_bbox = [
            max(0, min(pred_bbox[0], original_width)),
            max(0, min(pred_bbox[1], original_height)),
            max(0, min(pred_bbox[2], original_width)),
            max(0, min(pred_bbox[3], original_height)),
        ]

        # Classification Accuracy
        if pred_label == true_label:
            correct_classifications += 1

        if true_label == 1:  # Shelf is empty
            if pred_label == 1:
                tp += 1
            else:
                fn += 1
        else:  # Shelf is not empty
            if pred_label == 1:
                fp += 1
            else:
                tn += 1

        # IoU for Bounding Box
        iou = calculate_iou(true_bbox, pred_bbox)
        if iou >= iou_threshold:
            correct_bboxes += 1

        iou_list.append(iou)

        # Determine label colors and text
        true_color = (0, 255, 0) if true_label == 0 else (0, 0, 255)  # Green for empty, Red for not empty
        pred_color = (0, 255, 0) if pred_label == 0 else (255, 0, 0)  # Green for empty, Red for not empty
        
        true_label_text = "Empty" if true_label == 0 else "Not Empty"
        pred_label_text = "Empty" if pred_label == 0 else "Not Empty"

        # Draw true bounding box
        cv2.rectangle(draw_image, 
                      (int(true_bbox[0]), int(true_bbox[1])), 
                      (int(true_bbox[2]), int(true_bbox[3])), 
                      true_color,  
                      2)  # Line thickness
        
        # Add true label
        label_true = f"True: {true_label_text}"
        (label_width, label_height), _ = cv2.getTextSize(label_true, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(draw_image, 
                      (int(true_bbox[0]), int(true_bbox[1]) - label_height - 10), 
                      (int(true_bbox[0]) + label_width, int(true_bbox[1])), 
                      true_color, 
                      -1)
        cv2.putText(draw_image, label_true, 
                    (int(true_bbox[0]), int(true_bbox[1]) - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Draw predicted bounding box
        cv2.rectangle(draw_image, 
                      (pred_bbox[0], pred_bbox[1]), 
                      (pred_bbox[2], pred_bbox[3]), 
                      pred_color,  
                      2)  # Line thickness
        
        # Add predicted label
        label_pred = f"Pred: {pred_label_text}"
        (label_width, label_height), _ = cv2.getTextSize(label_pred, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(draw_image, 
                      (pred_bbox[0], pred_bbox[1] - label_height - 10), 
                      (pred_bbox[0] + label_width, pred_bbox[1]), 
                      pred_color, 
                      -1)
        cv2.putText(draw_image, label_pred, 
                    (pred_bbox[0], pred_bbox[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Add IoU text
        iou_text = f"IoU: {iou:.2f}"
        cv2.putText(draw_image, iou_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save the image with bounding boxes
        result_image_path = os.path.join(results_dir, f'result_{row["image_name"]}')
        cv2.imwrite(result_image_path, draw_image)

        # Increment processed count and print progress
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == total_samples:
            logging.info(f"Processed {processed_count}/{total_samples} images in directory: {image_folder}...")

    # Compute metrics
    classification_accuracy = correct_classifications / total_samples
    bbox_accuracy = correct_bboxes / total_samples

    compute_and_visualize_metrics(tp, fp, tn, fn, iou_list)
    # Output the results
    logging.info(f"Classification Accuracy: {classification_accuracy:.2f}")
    logging.info(f"BBox Accuracy (IoU ≥ {iou_threshold}): {bbox_accuracy:.2f}")
    logging.info(f"Evaluation images saved in: {results_dir}")


if __name__ == "__main__":
    main()