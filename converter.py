
import os
import pandas as pd

import os
import pandas as pd

def parse_yolov5_obb(annotations_dir, image_folder):
    data = []
    for label_file in os.listdir(annotations_dir):
        label_path = os.path.join(annotations_dir, label_file)
        image_name = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(image_folder, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_name} not found for {label_file}")
            continue

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 10:
                    print(f"Skipping malformed line in {label_file}: {line.strip()}")
                    continue

                try:
                    # Extract corner coordinates
                    x1, y1 = float(parts[0]), float(parts[1])
                    x2, y2 = float(parts[2]), float(parts[3])
                    x3, y3 = float(parts[4]), float(parts[5])
                    x4, y4 = float(parts[6]), float(parts[7])

                    # Calculate bounding box center, width, and height
                    x_center = (x1 + x2 + x3 + x4) / 4
                    y_center = (y1 + y2 + y3 + y4) / 4
                    width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
                    height = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)

                    # Extract label
                    label_name = parts[8]
                    label_map = {"empty-shelf": 0, "product": 1}  # Map labels to integers
                    label = label_map.get(label_name, 0)

                except ValueError:
                    print(f"Skipping invalid line in {label_file}: {line.strip()}")
                    continue

                data.append({
                    'image_name': image_name,
                    'label': label,
                    'label_name': label_name,  # Add the label_name here
                    'bbox_x': x_center,
                    'bbox_y': y_center,
                    'bbox_width': width,
                    'bbox_height': height,
                })

    return pd.DataFrame(data)
