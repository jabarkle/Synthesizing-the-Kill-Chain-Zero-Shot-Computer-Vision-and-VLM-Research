#!/usr/bin/env python3
"""
Grounding DINO - Highest Confidence Frame Extractor (ANNOTATED VERSION)
Process videos and save the frame with the highest detection confidence WITH bounding boxes drawn
For use in paper figures.
"""

# =============================================================================
# CONFIGURATION - Modify these settings
# =============================================================================

# Detection Settings
DETECTION_PROMPTS = "military tank."
BOX_THRESHOLD = 0.6  # Confidence threshold (0.0-1.0)
TEXT_THRESHOLD = 0.6  # Text-image matching threshold (0.0-1.0)

# Performance Settings
INFERENCE_EVERY_N_FRAMES = 20  # Run inference every N frames (higher = faster, lower = more accurate)

# Path Settings
INPUT_FOLDER = "DataForResearch"
OUTPUT_FOLDER = "output_annotated"  # Different folder for annotated images

# Annotation Settings
BOX_COLOR = (0, 255, 0)  # Green in BGR
BOX_THICKNESS = 3
LABEL_FONT_SCALE = 0.8
LABEL_COLOR = (0, 255, 0)  # Green
LABEL_BG_COLOR = (0, 0, 0)  # Black background for label
LABEL_THICKNESS = 2

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import cv2
import torch
import gc
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import time
import json
from pathlib import Path


def load_model():
    """Load Grounding DINO Tiny from HuggingFace"""
    print("Loading Grounding DINO Tiny...")

    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    print("Model loaded successfully")
    return processor, model, device


def detect_objects(processor, model, device, frame, text_prompt, box_threshold, text_threshold):
    """Run detection on a frame and return results with confidence scores"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)

    # Run inference
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results with both thresholds
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]]
    )[0]

    # Normalize boxes to 0-1000 for VLM consumption
    width, height = image_pil.size
    normalized_boxes = []

    if len(results['boxes']) > 0:
        for box in results['boxes']:
            x1, y1, x2, y2 = box.tolist()
            # Clamp to image boundaries
            x1 = max(0, min(width, x1))
            y1 = max(0, min(height, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))

            # Normalize to 0-1000 range
            nx1 = int((x1 / width) * 1000)
            ny1 = int((y1 / height) * 1000)
            nx2 = int((x2 / width) * 1000)
            ny2 = int((y2 / height) * 1000)

            normalized_boxes.append([nx1, ny1, nx2, ny2])

    # Use text_labels if available (newer API), otherwise labels
    labels_key = 'text_labels' if 'text_labels' in results else 'labels'

    return {
        'boxes': results['boxes'],
        'scores': results['scores'],
        'labels': results[labels_key],
        'normalized_boxes': normalized_boxes
    }


def draw_annotations(frame, boxes, scores, labels):
    """Draw bounding boxes and labels on the frame"""
    annotated_frame = frame.copy()

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # Get box coordinates (already in pixel coordinates)
        x1, y1, x2, y2 = [int(coord) for coord in box]

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # Prepare label text
        if isinstance(label, str):
            label_text = f"{label}: {score:.2f}"
        else:
            label_text = f"tank: {score:.2f}"

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS
        )

        # Draw background rectangle for label
        label_y = max(y1 - 10, text_height + 10)
        cv2.rectangle(
            annotated_frame,
            (x1, label_y - text_height - 5),
            (x1 + text_width + 5, label_y + 5),
            LABEL_BG_COLOR,
            -1  # Filled rectangle
        )

        # Draw label text
        cv2.putText(
            annotated_frame,
            label_text,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            LABEL_FONT_SCALE,
            LABEL_COLOR,
            LABEL_THICKNESS
        )

    return annotated_frame


def process_video(video_path, output_path, processor, model, device):
    """Process a single video and save the frame with highest detection confidence (with annotations)"""
    print(f"\nProcessing: {video_path.name}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Could not open video")
        return None

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  FPS: {fps}, Total frames: {total_frames}")

    frame_count = 0
    highest_confidence = 0.0
    best_frame = None
    best_frame_number = 0
    best_timestamp = 0.0
    best_results = None
    best_boxes_pixels = None  # Store pixel coordinates for drawing

    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps

            # Run inference every N frames
            if frame_count % INFERENCE_EVERY_N_FRAMES == 0:
                results = detect_objects(
                    processor, model, device, frame,
                    DETECTION_PROMPTS, BOX_THRESHOLD, TEXT_THRESHOLD
                )

                # Check if we have detections and track highest confidence
                if len(results['scores']) > 0:
                    max_score = results['scores'].max().item()

                    if max_score > highest_confidence:
                        highest_confidence = max_score
                        best_frame = frame.copy()
                        best_frame_number = frame_count
                        best_timestamp = current_time
                        best_boxes_pixels = results['boxes'].cpu().numpy().tolist()
                        best_results = {
                            'boxes': results['boxes'].cpu().numpy().tolist(),
                            'scores': results['scores'].cpu().numpy().tolist(),
                            'labels': results['labels'],
                            'normalized_boxes': results['normalized_boxes']
                        }

            # Progress update every 100 frames
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {progress:.1f}%", end='\r')

    except Exception as e:
        print(f"  Error processing video: {e}")
        return None

    finally:
        cap.release()
        torch.cuda.empty_cache()
        gc.collect()

    elapsed_time = time.time() - start_time

    # Save the best frame with annotations if we found any detection
    if best_frame is not None and best_results is not None:
        # Draw annotations on the frame
        annotated_frame = draw_annotations(
            best_frame,
            best_boxes_pixels,
            best_results['scores'],
            best_results['labels']
        )

        cv2.imwrite(str(output_path), annotated_frame)
        print(f"  Highest confidence: {highest_confidence:.4f} at frame {best_frame_number} ({best_timestamp:.2f}s)")
        print(f"  Saved (with annotations): {output_path.name}")
        print(f"  Completed in {elapsed_time:.1f}s")

        return {
            'video_name': video_path.name,
            'video_path': str(video_path),
            'output_image': str(output_path),
            'highest_confidence': highest_confidence,
            'frame_number': best_frame_number,
            'timestamp': best_timestamp,
            'detections': best_results
        }
    else:
        print(f"  No detections found above threshold")
        print(f"  Completed in {elapsed_time:.1f}s")
        return {
            'video_name': video_path.name,
            'video_path': str(video_path),
            'output_image': None,
            'highest_confidence': 0.0,
            'frame_number': 0,
            'timestamp': 0.0,
            'detections': None
        }


def process_all_videos():
    """Process all videos in the input folder structure"""
    print("\n" + "="*70)
    print("GROUNDING DINO - ANNOTATED FRAME EXTRACTOR")
    print("="*70)
    print(f"Detection target: {DETECTION_PROMPTS.replace('.', '').strip()}")
    print(f"Inference frequency: Every {INFERENCE_EVERY_N_FRAMES} frame(s)")
    print(f"Box threshold: {BOX_THRESHOLD}")
    print(f"Text threshold: {TEXT_THRESHOLD}")
    print(f"Output: Images WITH bounding boxes drawn")
    print("="*70 + "\n")

    # Load model once
    processor, model, device = load_model()

    # Setup paths
    script_dir = Path(__file__).parent
    input_dir = script_dir / INPUT_FOLDER
    output_dir = script_dir / OUTPUT_FOLDER

    # Create main output directory
    output_dir.mkdir(exist_ok=True)

    # Find all subdirectories with videos
    video_extensions = ['.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI']

    # Get all subdirectories in the input folder
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"No subdirectories found in {input_dir}")
        return

    print(f"Found {len(subdirs)} category folder(s):")
    for sd in subdirs:
        print(f"  - {sd.name}")

    # Process each subdirectory
    all_metadata = {}
    total_videos = 0
    total_detections = 0
    no_detection_videos = []  # Track videos with no detections

    for subdir in subdirs:
        print("\n" + "="*70)
        print(f"PROCESSING CATEGORY: {subdir.name}")
        print("="*70)

        # Create matching output subdirectory
        output_subdir = output_dir / subdir.name
        output_subdir.mkdir(exist_ok=True)

        # Find all video files in this subdirectory
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(subdir.glob(f'*{ext}')))

        if not video_files:
            print(f"  No video files found")
            continue

        print(f"Found {len(video_files)} video(s)")

        category_metadata = []

        for video_file in sorted(video_files):
            # Output image path (same name as video but .png)
            output_name = video_file.stem + ".png"
            output_path = output_subdir / output_name

            result = process_video(video_file, output_path, processor, model, device)
            if result:
                category_metadata.append(result)
                total_videos += 1
                if result['highest_confidence'] > 0:
                    total_detections += 1
                else:
                    no_detection_videos.append(f"{subdir.name}/{video_file.name}")

        all_metadata[subdir.name] = category_metadata

    # Save all metadata
    metadata_path = output_dir / "detection_metadata_annotated.json"
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE (ANNOTATED)")
    print("="*70)
    print(f"Detection Results: {total_detections}/{total_videos} videos had detections")
    print(f"Output location: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"\nNOTE: All images have bounding boxes drawn on them.")

    # Report videos with no detections
    if no_detection_videos:
        print("\n" + "-"*70)
        print(f"WARNING: {len(no_detection_videos)} video(s) had NO detections:")
        for video in no_detection_videos:
            print(f"  - {video}")
        print("-"*70)
    else:
        print("\nAll videos had at least one detection.")

    print("="*70)


if __name__ == "__main__":
    process_all_videos()
