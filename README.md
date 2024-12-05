# FAI_Fall24
Project on Empty Shelf Detection and product recommendation 



link to the dataset: https://northeastern-my.sharepoint.com/:f:/r/personal/aher_ar_northeastern_edu/Documents/images?csf=1&web=1&e=baLwM0
# Empty Shelf Detection and Product Recommendation

## Project Description
This innovative system combines computer vision and deep learning to automate retail shelf monitoring and provide intelligent restocking suggestions. The solution detects empty shelf spaces and recommends products based on spatial context analysis.

## Technical Features

### Model Architecture
- Custom CNN with 5 convolutional layers
- Batch normalization for training stability
- Residual connections for gradient optimization
- Global average pooling layer
- Triple fully connected layers with dropout

### Dataset Specifications
The dataset comprises retail shelf imagery from multiple sources:
- **Total images**: 1,680 at 640x640 resolution
- **Sources**: Stop & Shop, Target, 7-Eleven
- **Split ratio**: 80% training, 10% testing, 10% validation
- **Classes**: `empty-shelf` and `product`

## Implementation Steps

### Data Processing Pipeline
1. Grayscale transformation
2. Adaptive threshold application
3. CLAHE enhancement
4. Median blur filtering
5. Data normalization

## Getting Started

```bash
# Clone repository
git clone [your-repo-url]
cd empty-shelf-detection

# Install dependencies
pip install -r requirements.txt
## Usage Example

```python
# Import detector
from shelf_detector import ShelfDetector

# Initialize model
detector = ShelfDetector()

# Run detection
results = detector.detect('shelf_image.jpg')


## Project Structure

```text
.
├── data/
│   ├── train/
│   ├── test/
│   └── val/
├── models/
├── utils/
└── notebooks/


## Team
- Nikhil Satish Kulkarni
- Srinivasan Raghavan
- Aryan Aher
- Anshuman Raina
- Navneet Parab

## License
MIT License
