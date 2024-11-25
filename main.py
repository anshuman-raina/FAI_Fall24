from EmptyShelfDetectionCustomRCNN import predict_on_image
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

try:
    # Load the trained model
    model = load_model('empty_shelf_detector_rcnn_resnet.h5', compile=False)
    logging.info("Model loaded successfully")

    # Make prediction on a single image
    result_image = predict_on_image('C:\\2024 ClassesNEU\\5100\\FAI_Fall24\\sample3.jpg', model)
    logging.info("Prediction completed")

    # Display or save the result
    cv2.imwrite('C:\\2024 ClassesNEU\\5100\\FAI_Fall24\\prediction_result.jpg', result_image)
    logging.info("Result saved as prediction_result.jpg")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")