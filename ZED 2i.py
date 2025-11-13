import os
import sys

# CRITICAL: Add this at the very top of your script, before importing pyzed
zed_bin = r"C:\Program Files (x86)\ZED SDK\bin"
os.environ['PATH'] = zed_bin + os.pathsep + os.environ.get('PATH', '')
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(zed_bin)

import pyzed.sl as sl
import cv2
import numpy as np
from pathlib import Path
import glob

print("=" * 70)
print("ZED SDK BATCH SVO EXTRACTOR")
print("=" * 70)

# Base paths
raw_files_base = Path(r"C:/Users/stitli/Desktop/ZED/Raw_files")
output_base = Path(r"C:/Users/stitli/Desktop/ZED/Extracted_Data")

# Find all .svo files recursively
svo_files = list(raw_files_base.glob("**/*.svo"))

if not svo_files:
    print("ERROR: No .svo files found!")
    print(f"Searched in: {raw_files_base}")
    exit(1)

print(f"\nFound {len(svo_files)} SVO files to process")
print("-" * 70)

# Process each SVO file
for svo_index, svo_path in enumerate(svo_files, 1):
    print(f"\n[{svo_index}/{len(svo_files)}] Processing: {svo_path.name}")
    print("-" * 70)
    
    # Calculate relative path from raw_files_base
    relative_path = svo_path.relative_to(raw_files_base)
    
    # Create output directory maintaining the same structure
    # Remove the .svo extension from the file name for the folder
    svo_output_dir = output_base / relative_path.parent / svo_path.stem
    
    # Create subdirectories
    rgb_dir = svo_output_dir / "rgb"
    depth_raw_dir = svo_output_dir / "depth_raw"
    depth_viz_dir = svo_output_dir / "depth_viz"
    
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_raw_dir.mkdir(parents=True, exist_ok=True)
    depth_viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {svo_output_dir}")
    
    # Initialize ZED Camera
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False  # Process as fast as possible
    
    zed = sl.Camera()
    
    # Open the SVO file
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"  ERROR: Could not open SVO file - {str(status)}")
        print(f"  Skipping to next file...")
        continue
    
    # Get total number of frames
    total_frames = zed.get_svo_number_of_frames()
    print(f"  Total frames: {total_frames}")
    
    # Prepare data containers
    image = sl.Mat()
    depth = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    
    frame_id = 0
    successful_frames = 0
    
    try:
        while True:
            # Grab the next frame
            grab_status = zed.grab(runtime_parameters)
            
            if grab_status == sl.ERROR_CODE.SUCCESS:
                # Retrieve images and depth data
                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                
                # Get data as numpy arrays
                rgb = image.get_data()
                depth_map = depth.get_data()
                
                # Convert RGBA to BGR
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                
                # Save RGB Image
                rgb_path = rgb_dir / f"frame_{frame_id:06d}.png"
                cv2.imwrite(str(rgb_path), rgb_bgr)
                
                # Save RAW Depth Data
                depth_path = depth_raw_dir / f"depth_{frame_id:06d}.npy"
                np.save(str(depth_path), depth_map)
                
                # Save Visualized Depth
                depth_normalized = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
                depth_viz = cv2.normalize(depth_normalized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                depth_viz_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
                
                depth_viz_path = depth_viz_dir / f"depth_viz_{frame_id:06d}.png"
                cv2.imwrite(str(depth_viz_path), depth_viz_colored)
                
                successful_frames += 1
                
                # Progress update (every 10 frames to reduce output)
                if (frame_id + 1) % 10 == 0 or (frame_id + 1) == total_frames:
                    progress = (frame_id + 1) / total_frames * 100
                    print(f"  Progress: {frame_id + 1}/{total_frames} ({progress:.1f}%)", end='\r')
                
                frame_id += 1
                
            elif grab_status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                print(f"\n  Completed: {successful_frames} frames extracted")
                break
                
            else:
                print(f"\n  ERROR during grab: {str(grab_status)}")
                break
    
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        zed.close()
        print("\nBatch processing stopped by user.")
        exit(0)
    
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close the camera
        zed.close()
    
    print(f"  ✓ SVO file processed successfully\n")

print("=" * 70)
print("BATCH PROCESSING COMPLETE")
print("=" * 70)
print(f"\nTotal SVO files processed: {len(svo_files)}")
print(f"Output location: {output_base}")
print("\nFolder structure:")
print("  Extracted_Data/")
print("    Activity1/")
print("      person1/")
print("        ZED 21283893/")
print("          HD1080_SN29755398_17-45-38_Jakub1/")
print("            ├── rgb/")
print("            ├── depth_raw/")
print("            └── depth_viz/")
print("=" * 70)