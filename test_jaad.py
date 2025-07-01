#!/usr/bin/env python3
"""
Test script to verify JAAD dataset setup
"""

import sys
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from jaad_data import JAAD

def test_jaad_setup():
    """Test basic JAAD functionality"""
    print("Testing JAAD dataset setup...")
    print(f"Current directory: {current_dir}")
    
    # Check if required directories exist
    required_dirs = ['annotations', 'JAAD_clips', 'split_ids']
    for dir_name in required_dirs:
        dir_path = os.path.join(current_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ {dir_name} directory found")
        else:
            print(f"✗ {dir_name} directory missing")
    
    try:
        # Initialize JAAD
        imdb = JAAD(data_path=current_dir)
        print("✓ JAAD initialized successfully")
        
        # Test getting video IDs
        video_ids = imdb._get_video_ids()
        print(f"✓ Found {len(video_ids)} video annotations")
        if video_ids:
            print(f"  First few videos: {video_ids[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error initializing JAAD: {e}")
        return False

if __name__ == "__main__":
    test_jaad_setup()
