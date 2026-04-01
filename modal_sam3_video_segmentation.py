import os
import subprocess
from pathlib import Path

import modal

app = modal.App("sam3-tennis-video-segmentation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("torch", "torchvision")
    .run_commands(
        "git clone https://github.com/facebookresearch/sam3.git /root/sam3",
        "cd /root/sam3 && pip install -e '.[notebooks]'",
        "pip install 'numpy>=1.26,<2' supervision==0.27.0.post2 opencv-python-headless pillow ipython jupyter_bbox_widget",
    
    
        #"pip install supervision opencv-python-headless pillow ipython jupyter_bbox_widget",
        # "pip uninstall -y cc_torch || true",
        # "TORCH_CUDA_ARCH_LIST='8.0 9.0' pip install git+https://github.com/ronghanghu/cc_torch",
        # "pip uninstall -y torch_generic_nms || true",
        # "TORCH_CUDA_ARCH_LIST='8.0 9.0' pip install git+https://github.com/ronghanghu/torch_generic_nms",
    )
)

vol = modal.Volume.from_name("sam3-video-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A200", # before: L40S
    timeout=60 * 60,
    volumes={"/outputs": vol},
    secrets=[modal.Secret.from_name("hf-token")],
)
def segment_video(
    video_bytes: bytes,
    filename: str = "short_tennis_sample.mp4",
    prompt_text: str = "logo",
) -> str:
    import cv2
    import numpy as np
    import supervision as sv
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    workdir = Path("/tmp/sam3_run")
    workdir.mkdir(parents=True, exist_ok=True)

    video_path = workdir / filename
    video_path.write_bytes(video_bytes)

    frames_dir = workdir / video_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(frames_dir / "%05d.jpg"),
        ],
        check=True,
    )

    devices = [torch.cuda.current_device()]
    predictor = build_sam3_video_predictor(
        bpe_path="/root/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        gpus_to_use=devices,
    )

    response = predictor.handle_request(
        request={"type": "start_session", "resource_path": video_path.as_posix()}
    )
    session_id = response["session_id"]

    predictor.handle_request(
        request={"type": "reset_session", "session_id": session_id}
    )

    response = predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt_text,
        }
    )

    initial_result = response["outputs"]

    def from_sam(result: dict) -> sv.Detections:
        return sv.Detections(
            xyxy=sv.mask_to_xyxy(result["out_binary_masks"]),
            mask=result["out_binary_masks"],
            confidence=result["out_probs"],
            tracker_id=result["out_obj_ids"],
        )

    color = sv.ColorPalette.from_hex(
        [
            "#ffff00",
            "#ff9b00",
            "#ff8080",
            "#ff66b2",
            "#ff66ff",
            "#b266ff",
            "#9999ff",
            "#3399ff",
            "#66ffff",
            "#33ff99",
            "#66ff66",
            "#99ff00",
        ]
    )

    def annotate(image: np.ndarray, detections: sv.Detections, text: str | None = None) -> np.ndarray:
        h, w, _ = image.shape
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(w, h))

        mask_annotator = sv.MaskAnnotator(
            color=color,
            color_lookup=sv.ColorLookup.TRACK,
            opacity=0.6,
        )

        annotated_image = mask_annotator.annotate(image.copy(), detections)

        if text:
            label_annotator = sv.LabelAnnotator(
                color=color,
                color_lookup=sv.ColorLookup.TRACK,
                text_scale=text_scale,
                text_color=sv.Color.BLACK,
                text_position=sv.Position.TOP_CENTER,
                text_offset=(0, -30),
            )
            labels = [f"#{tracker_id} {text}" for tracker_id in detections.tracker_id]
            annotated_image = label_annotator.annotate(annotated_image, detections, labels)

        return annotated_image

    frame_outputs = {}
    for response in predictor.handle_stream_request(
        request={"type": "propagate_in_video", "session_id": session_id}
    ):
        frame_outputs[response["frame_index"]] = response["outputs"]

    target_video = workdir / f"{video_path.stem}-result{video_path.suffix}"
    compressed_video = workdir / f"{video_path.stem}-result-compressed{video_path.suffix}"

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        output = frame_outputs[index]
        detections = from_sam(output)
        return annotate(frame, detections, prompt_text)

    sv.process_video(
        source_path=str(video_path),
        target_path=str(target_video),
        callback=callback,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(target_video),
            "-vcodec",
            "libx264",
            "-crf",
            "28",
            str(compressed_video),
        ],
        check=True,
    )

    output_remote_path = f"/{compressed_video.name}"
    out_bytes = compressed_video.read_bytes()
    Path("/outputs", compressed_video.name).write_bytes(out_bytes)
    vol.commit()
    return output_remote_path


@app.local_entrypoint()
def main(
    video_path: str = "short_tennis_sample.mp4",
    prompt_text: str = "logo",
):
    local_video = Path(video_path)
    if not local_video.exists():
        raise FileNotFoundError(f"Input video not found: {local_video}")

    remote_output = segment_video.remote(
        video_bytes=local_video.read_bytes(),
        filename=local_video.name,
        prompt_text=prompt_text,
    )
    print("Saved output in Modal Volume 'sam3-video-results' at:", remote_output)
    print(
        f"Download it with: modal volume get sam3-video-results {remote_output} ./"
    )
