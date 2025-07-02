#!/usr/bin/env python3
"""
Script to extract a subset of images from JAAD videos:
- Uses 10% of the videos (random selection)
- Extracts every 5th frame from each selected video
- Clears existing images folder before extraction
"""

import os
import shutil
import cv2
import random
import xml.etree.ElementTree as ET
from pathlib import Path

class SubsetImageExtractor:
    def __init__(self, data_path='.', subset_percentage=0.1, frame_stride=5, random_seed=42):
        """
        Initialize the subset image extractor
        
        Args:
            data_path: Path to the JAAD dataset root folder
            subset_percentage: Percentage of videos to use (0.1 = 10%)
            frame_stride: Extract every Nth frame (5 = every 5th frame)
            random_seed: Seed for reproducible random selection
        """
        self.data_path = data_path
        self.subset_percentage = subset_percentage
        self.frame_stride = frame_stride
        self.random_seed = random_seed
        
        # Paths
        self.clips_path = os.path.join(data_path, 'JAAD_clips')
        self.annotation_path = os.path.join(data_path, 'annotations')
        self.images_path = os.path.join(data_path, 'images')
        
        # Set random seed for reproducibility
        random.seed(random_seed)
    
    def get_video_list(self):
        """Get list of available video files"""
        video_files = [f for f in os.listdir(self.clips_path) if f.endswith('.mp4')]
        video_files.sort()  # Sort for consistency
        return video_files
    
    def select_subset_videos(self, video_files):
        """Select a random subset of videos"""
        num_videos = len(video_files)
        num_subset = max(1, int(num_videos * self.subset_percentage))
        
        selected_videos = random.sample(video_files, num_subset)
        selected_videos.sort()  # Sort for consistent processing order
        
        print(f"Total videos available: {num_videos}")
        print(f"Selected videos ({self.subset_percentage*100}%): {num_subset}")
        print(f"Selected videos: {selected_videos[:5]}{'...' if len(selected_videos) > 5 else ''}")
        
        return selected_videos
    
    def clear_images_folder(self):
        """Clear the existing images folder"""
        if os.path.exists(self.images_path):
            print(f"Clearing existing images folder: {self.images_path}")
            shutil.rmtree(self.images_path)
        
        # Create fresh images folder
        os.makedirs(self.images_path, exist_ok=True)
        print(f"Created fresh images folder: {self.images_path}")
    
    def get_video_frame_count(self, video_id):
        """Get the number of frames from the annotation file"""
        try:
            annotation_file = os.path.join(self.annotation_path, f"{video_id}.xml")
            if os.path.exists(annotation_file):
                tree = ET.parse(annotation_file)
                num_frames = int(tree.find("./meta/task/size").text)
                return num_frames
        except Exception as e:
            print(f"Warning: Could not read frame count for {video_id}: {e}")
        
        # Fallback: return None to let OpenCV determine
        return None
    
    def extract_frames_from_video(self, video_file):
        """Extract every Nth frame from a video"""
        video_id = video_file.replace('.mp4', '')
        video_path = os.path.join(self.clips_path, video_file)
        
        # Create output directory for this video
        output_dir = os.path.join(self.images_path, video_id)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing {video_id}...")
        
        # Get expected frame count
        expected_frames = self.get_video_frame_count(video_id)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}")
            return 0
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract every Nth frame
            if frame_count % self.frame_stride == 0:
                output_path = os.path.join(output_dir, f"{frame_count:05d}.png")
                cv2.imwrite(output_path, frame)
                extracted_count += 1
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 100 == 0:
                progress = (frame_count / expected_frames * 100) if expected_frames else 0
                print(f"  Frame {frame_count}" + (f" ({progress:.1f}%)" if expected_frames else ""))
        
        cap.release()
        
        print(f"  Completed {video_id}: {frame_count} total frames, {extracted_count} extracted")
        return extracted_count
    
    def extract_subset_images(self):
        """Main function to extract subset of images"""
        print("="*60)
        print("JAAD Subset Image Extraction")
        print("="*60)
        print(f"Configuration:")
        print(f"  - Subset percentage: {self.subset_percentage*100}%")
        print(f"  - Frame stride: every {self.frame_stride} frames")
        print(f"  - Random seed: {self.random_seed}")
        print()
        
        # Step 1: Get video list
        video_files = self.get_video_list()
        if not video_files:
            print("Error: No video files found in JAAD_clips folder")
            return
        
        # Step 2: Select subset
        selected_videos = self.select_subset_videos(video_files)
        
        # Step 3: Clear existing images
        self.clear_images_folder()
        
        # Step 4: Extract frames
        print(f"\nExtracting frames from {len(selected_videos)} videos...")
        print("-"*60)
        
        total_extracted = 0
        for i, video_file in enumerate(selected_videos, 1):
            print(f"[{i}/{len(selected_videos)}] ", end="")
            extracted = self.extract_frames_from_video(video_file)
            total_extracted += extracted
        
        print("-"*60)
        print(f"Extraction completed!")
        print(f"Total frames extracted: {total_extracted}")
        print(f"Images saved to: {self.images_path}")
        
        # Step 5: Show summary
        self.show_summary()
    
    def show_summary(self):
        """Show summary of extracted images"""
        print("\nSummary of extracted images:")
        print("-"*40)
        
        if not os.path.exists(self.images_path):
            print("No images folder found")
            return
        
        video_dirs = [d for d in os.listdir(self.images_path) 
                     if os.path.isdir(os.path.join(self.images_path, d))]
        video_dirs.sort()
        
        total_images = 0
        for video_dir in video_dirs:
            video_path = os.path.join(self.images_path, video_dir)
            image_count = len([f for f in os.listdir(video_path) if f.endswith('.png')])
            total_images += image_count
            print(f"  {video_dir}: {image_count} images")
        
        print(f"\nTotal: {len(video_dirs)} videos, {total_images} images")

def main():
    """Main function"""
    # Configuration
    data_path = '.'  # Current directory (adjust if needed)
    subset_percentage = 0.1  # 10% of videos
    frame_stride = 5  # Every 5th frame
    
    # Create extractor and run
    extractor = SubsetImageExtractor(
        data_path=data_path,
        subset_percentage=subset_percentage,
        frame_stride=frame_stride
    )
    
    extractor.extract_subset_images()

if __name__ == "__main__":
    main()
