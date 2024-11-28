import easyocr
import cv2
import numpy as np
import tensorflow as tf
import logging

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

def preprocess_image(image_path, target_size=(640, 640)):
    logger.info(f"Preprocessing image: {image_path}")
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, target_size)
        image = image.astype(np.float32) / 255.0
        preprocessed = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dimensions
        logger.info(f"Preprocessed image shape: {preprocessed.shape}")
        return preprocessed
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_coordinates(model, preprocessed_image, original_image_shape=(640, 640)):
    logger.info("Predicting coordinates and label")
    try:
        pred_bbox, pred_label_probs = model.predict(preprocessed_image)
        pred_label = np.argmax(pred_label_probs[0])
        
        original_height, original_width = original_image_shape[:2]
        
        # Extract predicted bounding box
        pred_bbox = [
            int(pred_bbox[0, 0] * original_width),
            int(pred_bbox[0, 1] * original_height),
            int(pred_bbox[0, 2] * original_width),
            int(pred_bbox[0, 3] * original_height),
        ]
        
        logger.info(f"Predicted bounding box: {pred_bbox}")
        logger.info(f"Predicted label: {pred_label}")
        
        return pred_bbox, pred_label
    except Exception as e:
        logger.error(f"Error predicting coordinates and label: {str(e)}")
        raise

def adjust_coordinates(coordinates, image_shape):
    logger.info("Adjusting coordinates")
    height, width = image_shape[:2]
    x1, y1, x2, y2 = coordinates
    adjusted = [
        max(0, min(int(x1), width)),
        max(0, min(int(y1), height)),
        max(0, min(int(x2), width)),
        max(0, min(int(y2), height))
    ]
    logger.info(f"Adjusted coordinates: {adjusted}")
    return adjusted

def extract_text(image_path, coordinates):
    logger.info("Extracting text")
    try:
        reader = easyocr.Reader(['en'])
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
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
        logger.info(f"Cropped dimensions: {cropped.shape}")
        if cropped.size == 0:
            logger.warning("Cropped image is empty")
            return []
        
        detections = reader.readtext(cropped)
        results = [{'text': text, 'confidence': confidence} for _, text, confidence in detections]
        logger.info(f"Extracted {len(results)} text regions")
        return results
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return []

def display_results(image_path, coordinates, detected_text):
    logger.info("Displaying results")
    image = cv2.imread(image_path)
    x1, y1, x2, y2 = coordinates
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    y_text = y1 - 10
    for text_obj in detected_text:
        text = f"{text_obj['text']} ({text_obj['confidence']:.2f})"
        cv2.putText(image, text, (x1, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_text -= 20
    output_path = "annotated_" + image_path.split('/')[-1]
    cv2.imwrite(output_path, image)
    logger.info(f"Annotated image saved as: {output_path}")

def main(model_path, image_path):
    try:
        model = load_model(model_path)
        preprocessed_image = preprocess_image(image_path)
        
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Preprocessed image shape: {preprocessed_image.shape}")
        
        predicted_bbox, predicted_label = predict_coordinates(model, preprocessed_image)
        original_image = cv2.imread(image_path)
        adjusted_coords = adjust_coordinates(predicted_bbox, original_image.shape)
        detected_text = extract_text(image_path, adjusted_coords)
        
        logger.info("\nDetected Text:")
        for text_obj in detected_text:
            logger.info(f"- {text_obj['text']} (Confidence: {text_obj['confidence']:.2f})")
        
        display_results(image_path, adjusted_coords, detected_text)
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    model_path = "Product_detector_rcnn_resnet_full_product.h5"
    image_path = "text2.jpg"
    main(model_path, image_path)