# Standard library imports for file and system operations
import os
import argparse
from pathlib import Path

# Third-party library imports for image processing, machine learning, and computer vision
import cv2
import torch
import numpy as np
from tqdm import tqdm
import supervision as sv
from torchvision.ops import box_convert

# Custom module imports for SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Custom module imports for Grounding DINO
from groundingdino.util.inference import load_model, load_image, predict, load_image_from_video

# Custom module imports for shoe-splatter
import utils

# Get the secret key from the environment variables.
SECRET_KEY = os.environ.get('AM_I_DOCKER', False)
if SECRET_KEY:
    print('I am running in a Docker container')

# Constants
TEXT_PROMPT = "shoe. shoes. boot."
SAM2_CHECKPOINT = "/app/checkpoints/sam2.1_hiera_large.pt" if SECRET_KEY else "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "/app/extensions/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
    if SECRET_KEY else "./extensions/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "/app/checkpoints/groundingdino_swint_ogc.pth" if SECRET_KEY else "./checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def detect_frame(image_source: np.array, image: torch.Tensor, sam2_predictor, grounding_model) -> None:
    """
    Detect objects in an input image using SAM 2 and Grounding DINO.

    Args:
        image_source (np.array): The original image as a numpy array.
        image (torch.Tensor): The input image as a PyTorch tensor.
        sam2_predictor (SAM2ImagePredictor): The SAM 2 predictor object.
        grounding_model (nn.Module): The Grounding DINO model object.

    Returns:
        tuple: A tuple containing the bounding boxes, masks, class IDs, and confidences.
    """
    sam2_predictor.set_image(image_source)
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )
    if boxes.size() == (0,4):
        return [], [], [], []
    
    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # FIXME: figure how does this influence the G-DINO model
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.numpy()
    class_names = labels
    class_ids = np.array(list(range(len(class_names))))

    return input_boxes, masks.astype(bool), class_ids, confidences


def filter_largest_detections(detections: sv.Detections) -> sv.Detections:
    """
    Filter the detections to keep only the largest detection.
    Args:
        detections (sv.Detections): Detections object containing bounding boxes, masks, class IDs, and confidences.
    
    Returns:
        sv.Detections: Detections object containing the largest detection.
    """
    largest_idx = 0
    largest_area = 0

    for i in range(len(detections.xyxy)):
        xmin, ymin, xmax, ymax = detections.xyxy[i]
        area = (xmax - xmin) * (ymax - ymin)
        if area > largest_area:
            largest_idx = i
    return detections[largest_idx]


def fill_holes(binary_image: np.ndarray) -> np.ndarray:
    """
    Fill holes in a binary image using morphological operations.

    Args:
        binary_image (np.ndarray): Input binary image (0 and 255 values).

    Returns:
        np.ndarray: Binary image with holes filled.
    """
    # Ensure the input image is binary (0 and 255)
    binary_image = np.where(binary_image > 0, 255, 0).astype(np.uint8)

    # Create a filled image to start with
    filled_image = np.zeros(binary_image.shape, dtype=np.uint8)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw filled contours on the empty image
    cv2.drawContours(filled_image, contours, -1, 255, cv2.FILLED)

    return filled_image


def run_image_detection(input_path: Path, masks_path: Path, annotated_path: Path, images_path: Path, sam2_predictor, grounding_model) -> None:
    """
    Run detection on the input image path using SAM 2 and Grounding DINO.

    Args:
        input_path (Path): Path to the input image files.
        masks_path (Path): Path to save the output image masks with detections.
        annotated_path (Path): Path to save the output image files with detections.
        images_path (Path): Path to save the output image files.
        sam2_predictor (SAM2ImagePredictor): The SAM 2 predictor object.
        grounding_model (nn.Module): The Grounding DINO model object.
    Returns:
        None
    """
    image_files  = utils.get_image_files(input_path)

    tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    with tqdm(total=len(image_files), desc="Processing Images", unit="image") as pbar:
        for img_id, image_file in enumerate(image_files): 
            image_source, image = load_image(image_file)
            input_boxes, masks, class_ids, confidences = detect_frame(image_source, image, sam2_predictor, grounding_model)

            detections = sv.Detections(
                xyxy=input_boxes,
                mask=masks,
                class_id=class_ids,
                confidence=confidences
            )

            detections = detections.with_nmm(threshold=0.1, class_agnostic=True)
            detections = filter_largest_detections(detections)
            detections = tracker.update_with_detections(detections)

            if len(detections) == 1:
                # confidence_mask = np.where(detections.confidence > 0.25)[0].tolist()
                # detections = detections[confidence_mask]
                # labels = [labels[i] for i in confidence_mask]

                annotated_frame = box_annotator.annotate(scene=cv2.cvtColor(image_source.copy(),
                                                                            cv2.COLOR_BGR2RGB),
                                                                            detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                cv2.imwrite(os.path.join(annotated_path, image_file.name), annotated_frame)
                mask_image = np.asarray(detections.mask[0].astype(np.uint8))
                cv2.imwrite(os.path.join(masks_path, image_file.name), fill_holes(mask_image))
                # image_output = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
                # image_output[np.where(mask_image == 0)] = (0,0,0)
                # cv2.imwrite(os.path.join(images_path, image_file.name), image_output)
            else:
                mask_image = np.zeros(image_source.shape[:2]).astype(np.uint8)
                cv2.imwrite(os.path.join(masks_path, image_file.name), mask_image)

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix({"Current File": image_file.name})

    print(f"Processed {len(image_files)} images.")


def run_video_detection(input_path: Path, masks_path: Path, annotated_path: Path, images_path: Path, sam2_predictor, grounding_model) -> None:
    """
    Run detection on the input video path using SAM 2 and Grounding DINO.

    Args:
        input_path (Path): Path to the input image files.
        masks_path (Path): Path to save the output image masks with detections.
        annotated_path (Path): Path to save the output image files with detections.
        images_path (Path): Path to save the output image files.
        sam2_predictor (SAM2ImagePredictor): The SAM 2 predictor object.
        grounding_model (nn.Module): The Grounding DINO model object.
    Returns:
        None
    """
    tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=input_path)
    video_info = sv.VideoInfo.from_video_path(video_path=input_path)

    for x, frame in enumerate(tqdm(frame_generator, desc="Processing Video", total=video_info.total_frames)):
        if x % 15 != 0:
            continue
        image_source, image = load_image_from_video(frame)
        input_boxes, masks, class_ids, confidences = detect_frame(image_source, image, sam2_predictor, grounding_model)

        if len(input_boxes) == 0:
            continue

        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks,
            class_id=class_ids,
            confidence=confidences
        )

        detections = detections.with_nmm(threshold=0.1, class_agnostic=True)
        detections = filter_largest_detections(detections)
        detections = tracker.update_with_detections(detections)

        if len(detections) == 1:
            # confidence_mask = np.where(detections.confidence > 0.25)[0].tolist()
            # detections = detections[confidence_mask]
            # labels = [labels[i] for i in confidence_mask]

            annotated_frame = box_annotator.annotate(scene=image_source.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(annotated_path, f"{x:05d}.jpg"), annotated_frame)
            mask_image = np.asarray(detections.mask[0].astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(masks_path, f"{x:05d}.jpg"), fill_holes(mask_image))
            image_output = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
            image_output[np.where(mask_image == 0)] = (0,0,0)
            cv2.imwrite(os.path.join(images_path, f"{x:05d}.jpg"), image_output)

    print(f"Processed {int(video_info.total_frames/15)} frames.")


def main(input_path: Path) -> None:
    """
    Detect and save masks from an input image or directory.

    Args:
        input_path (Path): Path to the results workspace.
    
    Returns:
        None: No return value.
    """
    # create output directory
    images_path = input_path / "images"
    masks_path = input_path / "masks"
    annotated_path = input_path / "annotated"
    utils.clear_and_create_directory(masks_path)
    utils.clear_and_create_directory(annotated_path)

    # build SAM2 image predictor
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )

    input_type, _ = utils.detect_input_type(images_path)

    if input_type == "video":
        run_video_detection(input_path, masks_path, annotated_path, images_path, sam2_predictor, grounding_model)

    elif input_type == "images":
        run_image_detection(images_path, masks_path, annotated_path, images_path, sam2_predictor, grounding_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect masks in shoe splatter images.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the results workspace")
    args = parser.parse_args()

    main(args.input)