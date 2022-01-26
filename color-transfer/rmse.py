import cv2
import numpy as np
import sys
from os.path import exists

if __name__ == "__main__":
    # Some error detection
    if len(sys.argv) != 3:
        sys.exit('Usage: python rmse.py image1_path image2_path')
    
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    
    if not exists(image1_path):
        sys.exit(f'Image "{image1_path}" does not exist.')
    if not exists(image2_path):
        sys.exit(f'Image "{image2_path}" does not exist.')
    
    # Read images
    image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR).astype(np.float32)
    image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR).astype(np.float32)

    # Compute the RMSE
    print(f'RMSE({image1_path}, {image2_path}) = {np.sqrt(np.mean((image1 - image2)**2))}')
