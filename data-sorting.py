# This script extracts images from JAAD dataset videos and saves them in a structured format.
# It assumes the JAAD dataset is organized in a specific directory structure.
# It is used to split the video into frames.
# I have used it to save the images in a folder named "images" with subfolders for each video.
# So far, I have only done the first 7 videos, since it consumes a lot of memory and time.
import sys
import os

# Get the current directory (where this script is located)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)  # Add current directory to Python path

from jaad_data import JAAD  # Now you can import the JAAD class

# Use the current directory as the JAAD path since all files are here
jaad_path = current_dir
imdb = JAAD(data_path=jaad_path)
crossing_data = imdb.extract_and_save_images()

# Print results
print("Image extraction completed successfully!")