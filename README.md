# Empty Shelf Detection and Product Recommendation

## Project Description
This innovative system combines computer vision and deep learning to automate retail shelf monitoring and provide intelligent restocking suggestions. The solution detects empty shelf spaces and recommends products based on spatial context analysis.

## Technical Features

### Model Architecture
- 5 convolutional layers with batch normalization
- Residual connections for gradient optimization
- Global average pooling for feature condensation
- 3 fully connected layers with dropout for overfitting prevention
- Dual output: Classification (empty-shelf/product) and Bounding Box coordinates

### Loss Functions
Binary Cross-Entropy for classification
Mean Squared Error for bounding box prediction


## Dataset Information

- **Total Images**: 1,680 retail shelf images (640x640 resolution)
- **Sources**: Stop & Shop, Target, and 7-Eleven
- **Split Ratio**: 80:10:10 (train:test:validation)
- **Classes**: 
  - `empty-shelf`
  - `product`
- **Additional Details**: Includes adjacent product context for improved learning
-  Link to the dataset: https://northeastern-my.sharepoint.com/:f:/r/personal/aher_ar_northeastern_edu/Documents/images?csf=1&web=1&e=baLwM0

## Data Augmentation Techniques

- **Shearing**: -15° to +15° range
- **Image Flipping**: Within 15° range
- **Transformations**: Front and back transformations
  
## Implementation Steps

### Data Processing Pipeline
1. Grayscale transformation
2. Adaptive threshold application
3. CLAHE enhancement
4. Median blur filtering
5. Data normalization


## Recommendation System

### Product Detection Process
1. **Empty Shelf Region Detection**: Using a CNN model.
2. **Adjacent Product Identification**: Through a retrained CNN.
3. **Reference Selection**: Nearest product chosen as reference.
4. **Text Extraction**: Extracting product labels.
5. **Recommendation Generation**: Based on contextual analysis and confidence level.

## Project Components

### Core Functionalities
- **Empty Shelf Detection**: With bounding boxes.
- **Product Identification**: Near empty spaces.
- **Text Recognition**: For product labels.
- **Context-Aware Restocking Suggestions**: Intelligent recommendations based on spatial context.

## System Requirements

- **Python**: Version 3.8+
- **PyTorch**
- **OpenCV**
- **CUDA**: Recommended for GPU acceleration

## Getting Started

```bash
# Clone repository
git clone https://github.com/anshuman-raina/FAI_Fall24
cd empty-shelf-detection
```

# Install dependencies
```
pip install -r requirements.txt
```

## Usage Example

```python
python .\product_recommender.py
```

## Project Structure

```text
.
├───Data
│   ├───test
│   ├───train
│   └───valid
├───Models
├───Notebooks
├───Report and Slides
├───Results
│   ├───evaluation_results
│   ├───Final_Output
│   └───Preprocessing Images
├───Scripts
└───Utils
```

## Performance Metrics

- **High Accuracy**: In empty shelf detection.
- **Efficient Product Identification**: Robust recognition of products.
- **Real-Time Processing**: Capability for quick analysis.
- **Contextual Recommendation Accuracy**: Intelligent and precise suggestions.

## Future Enhancements

- **Integration**: With inventory management systems.
- **Real-Time Monitoring**: Enabling live updates and alerts.
- **Mobile Application Development**: For on-the-go access and usability.
- **Multi-Store Deployment Support**: Scalability for broader usage.


## Team
- Nikhil Satish Kulkarni
- Srinivasan Raghavan
- Aryan Aher
- Anshuman Raina
- Navneet Parab

## License
MIT License
