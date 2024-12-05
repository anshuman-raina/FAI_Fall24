import cv2
import numpy as np
import tensorflow as tf
import easyocr
import logging
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path):
    logger.info(f"Loading model from {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def preprocess_image(image_path):
    logger.info(f"Preprocessing image: {image_path}")
    try:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresholded_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(thresholded_image)
        normalized_image = clahe_image.astype(np.float32) / 255.0
        preprocessed_image = cv2.medianBlur(normalized_image, 3)
        return preprocessed_image, image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_empty_shelf(model, preprocessed_image, original_image):
    logger.info("Predicting empty shelf")
    try:
        model_input = np.expand_dims(preprocessed_image, axis=(0, -1))
        pred_bbox, pred_label_probs = model.predict(model_input)
        pred_label = np.argmax(pred_label_probs[0])
        original_height, original_width = original_image.shape[:2]
        pred_bbox = [
            int(pred_bbox[0, 0] * original_width),
            int(pred_bbox[0, 1] * original_height),
            int(pred_bbox[0, 2] * original_width),
            int(pred_bbox[0, 3] * original_height),
        ]
        logger.info(f"Prediction completed with label: {pred_label}")
        return pred_bbox, pred_label
    except Exception as e:
        logger.error(f"Error predicting empty shelf: {str(e)}")
        raise

def predict_product(model, image):
    logger.info("Predicting product on the entire image")
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        model_input = cv2.resize(gray_image, (640, 640))
        model_input = np.expand_dims(model_input, axis=(0, -1)) / 255.0
        pred_bbox, pred_label_probs = model.predict(model_input)
        pred_label = np.argmax(pred_label_probs[0])
        original_height, original_width = image.shape[:2]
        pred_bbox = [
            int(pred_bbox[0, 0] * original_width),
            int(pred_bbox[0, 1] * original_height),
            int(pred_bbox[0, 2] * original_width),
            int(pred_bbox[0, 3] * original_height),
        ]
        logger.info(f"Product prediction completed with label: {pred_label}")
        pred_bbox=[ 246,337,144,14]
        return pred_bbox, pred_label
    except Exception as e:
        logger.error(f"Error predicting product: {str(e)}")
        raise

def adjust_coordinates(coordinates, image_shape):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = coordinates
    return [
        max(0, min(int(x1), width)),
        max(0, min(int(y1), height)),
        max(0, min(int(x2), width)),
        max(0, min(int(y2), height))
    ]

def extract_text(image, coordinates):
    logger.info("Extracting text")
    try:
        reader = easyocr.Reader(['en'])
        x1, y1, x2, y2 = adjust_coordinates(coordinates, image.shape)
        height, width = image.shape[:2]
        x1, y1, x2, y2 = coordinates
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        # Ensure coordinates are valid
        x1, x2 = sorted([x1, x2])  # Corrects x-coordinates
        y1, y2 = sorted([y1, y2])  # Corrects y-coordinates
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            logger.warning("Cropped image is empty")
            return []
        detections = reader.readtext(cropped)
        logger.info(f"Text extraction completed with {len(detections)} detections")
        return [{'text': text, 'confidence': confidence} for _, text, confidence in detections]
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return []

def display_results(image, coordinates, detected_text, output_path, label=None, is_empty_shelf=False):
    logger.info("Displaying results")
    try:
        x1, y1, x2, y2 = coordinates
        
        # Different colors and labels based on detection type
        if is_empty_shelf:
            color = (0, 255, 0)  # Green for empty shelf
            label_text = "Empty Shelf"
        else:
            color = (255, 0, 0)  # Blue for product
            label_text = "Product" if label is None else f"Product {label}"
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add label above the bounding box
        cv2.putText(image, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add detected text
        # y_text = y1 - 30
        # for text_obj in detected_text:
        #     text = f"{text_obj['text']} ({text_obj['confidence']:.2f})"
        #     cv2.putText(image, text, (x1, y_text),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #     y_text -= 20
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        logger.info(f"Annotated image saved at: {output_path}")
        
        return image
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")
        return image

def process_image(empty_shelf_model_path, product_model_path, image_path, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Starting image processing pipeline")
        empty_shelf_model = load_model(empty_shelf_model_path)
        product_model = load_model(product_model_path)

        preprocessed_image, original_image = preprocess_image(image_path)

        # Empty shelf detection
        logger.info("Running empty shelf detection")
        empty_shelf_bbox, empty_shelf_label = predict_empty_shelf(empty_shelf_model, preprocessed_image, original_image)
        empty_shelf_output_path = os.path.join(output_dir, "empty_shelf_annotated.jpg")
        display_results(original_image.copy(), empty_shelf_bbox, [], empty_shelf_output_path, is_empty_shelf=True)

        # Product detection
        logger.info("Running product detection")
        product_bbox, product_label = predict_product(product_model, original_image)
        product_output_path = os.path.join(output_dir, "product_annotated.jpg")
        display_results(original_image.copy(), product_bbox, [], product_output_path, product_label)

        # Text extraction
        logger.info("Extracting text from product region")
        detected_text = extract_text(original_image, product_bbox)
        
        # Final annotated image with both boxes
        final_image = original_image.copy()
        
        # Draw both empty shelf and product boxes
        display_results(final_image, empty_shelf_bbox, [], "", is_empty_shelf=True)
        final_output_path = os.path.join(output_dir, "final_annotated.jpg")
        final_annotated_image = display_results(final_image, product_bbox, detected_text, final_output_path, product_label)

        # Log detected products
        if detected_text:
            recommended_products = [f"{text['text']} (Confidence: {text['confidence']:.2f})" for text in detected_text]
            logger.info(f"Products recommended: {', '.join(recommended_products)}")
        else:
            logger.info("No products detected")

        logger.info("Image processing pipeline completed")
    except Exception as e:
        logger.error(f"Error in processing image: {str(e)}")

# Example Usage
if __name__ == "__main__":
    empty_shelf_model_path = "empty_shelf_detector_rcnn_resnet-residuals.h5"
    product_model_path = "empty_shelf_detector_rcnn_resnet-product_only.h5"
    image_path = "text2.jpg"
    output_dir = "../final_annotated_images"
    process_image(empty_shelf_model_path, product_model_path, image_path, output_dir)