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
    )
)

vol = modal.Volume.from_name("sam3-video-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB", # before: L40S
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
    import json
    import time
    import numpy as np
    import supervision as sv
    import torch
    from sam3.model_builder import build_sam3_video_predictor
    
    total_start = time.perf_counter() # start a global timer

    workdir = Path("/tmp/sam3_run")
    workdir.mkdir(parents=True, exist_ok=True)

    video_path = workdir / filename
    video_path.write_bytes(video_bytes)
    
    cap = cv2.VideoCapture(str(video_path))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()


    target_video = workdir / f"{video_path.stem}-result{video_path.suffix}"
    compressed_video = workdir / f"{video_path.stem}-result-compressed{video_path.suffix}"

    

    devices = [torch.cuda.current_device()]
    predictor = build_sam3_video_predictor(
        bpe_path="/root/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        gpus_to_use=devices,
    )
    
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    gpu_props = torch.cuda.get_device_properties(gpu_index)
    gpu_total_memory_gb = round(gpu_props.total_memory / (1024**3), 2)

    print("=" * 80)
    print("SAM3 / MODAL RUNTIME INFO")
    print(f"GPU index           : {gpu_index}")
    print(f"GPU name            : {gpu_name}")
    print(f"GPU total memory GB : {gpu_total_memory_gb}")
    print(f"Input video         : {filename}")
    print(f"Source resolution   : {source_width}x{source_height}")
    print(f"Source frame count  : {source_frame_count}")
    print(f"Source FPS          : {source_fps:.3f}")
    print("=" * 80)

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
    per_frame_inference_ms = {}      # SAM3-only streamed inference time per frame
    per_frame_latency_ms = {}        # inference + render for each frame
    frame_arrival_time = {}          # when SAM3 returns that frame

    inference_start = time.perf_counter()
    prev_yield_time = inference_start

    # Measures SAM3 streaming inference time per yielded frame
    for response in predictor.handle_stream_request(
        request={"type": "propagate_in_video", "session_id": session_id}
    ):
        now = time.perf_counter()
        frame_index = response["frame_index"]

        # SAM3-only time between this yielded frame and the previous yielded frame
        per_frame_inference_ms[frame_index] = (now - prev_yield_time) * 1000.0

        # Store when this frame became available from SAM3
        frame_arrival_time[frame_index] = now

        frame_outputs[frame_index] = response["outputs"]
        prev_yield_time = now

    inference_total_s = time.perf_counter() - inference_start
    processed_frame_count = len(frame_outputs)
    inference_fps = (
        processed_frame_count / inference_total_s if inference_total_s > 0 else 0.0
    )


    
    per_frame_render_ms = {}

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        t0 = time.perf_counter()

        output = frame_outputs[index]
        detections = from_sam(output)
        annotated = annotate(frame, detections, prompt_text)

        render_ms = (time.perf_counter() - t0) * 1000.0
        per_frame_render_ms[index] = render_ms

        # Full per-frame latency: SAM3 yielded the frame result + rendering that frame
        inference_ms = per_frame_inference_ms.get(index, 0.0)
        per_frame_latency_ms[index] = inference_ms + render_ms

        return annotated

    render_start = time.perf_counter()

    # Rendering/Annotation Time
    sv.process_video(
        source_path=str(video_path),
        target_path=str(target_video),
        callback=callback,
    )

    render_total_s = time.perf_counter() - render_start
    render_fps = (
        processed_frame_count / render_total_s if render_total_s > 0 else 0.0
    )

    compress_start = time.perf_counter()
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
    compress_total_s = time.perf_counter() - compress_start
    
    total_elapsed_s = time.perf_counter() - total_start
    end_to_end_fps = (
        processed_frame_count / total_elapsed_s if total_elapsed_s > 0 else 0.0
    )

    end_to_end_ms_per_frame = (
    (total_elapsed_s / processed_frame_count) * 1000.0
    if processed_frame_count > 0 else 0.0
    )
    
    source_frame_duration_ms = (
    1000.0 / source_fps if source_fps and source_fps > 0 else 0.0
    )
    
    # RTF = how much slower or faster than real-time your pipeline is.
    # rtf = 1.0 → real-time
    # rtf < 1.0 → faster than real-time
    #rtf > 1.0 → slower than real-time
    rtf = (
    end_to_end_ms_per_frame / source_frame_duration_ms
    if source_frame_duration_ms > 0 else 0.0
    )
    

    avg_inference_ms = (
        sum(per_frame_inference_ms.values()) / len(per_frame_inference_ms)
        if per_frame_inference_ms else 0.0
    )
    avg_render_ms = (
        sum(per_frame_render_ms.values()) / len(per_frame_render_ms)
        if per_frame_render_ms else 0.0
    )
    
    avg_latency_ms = (
    sum(per_frame_latency_ms.values()) / len(per_frame_latency_ms)
    if per_frame_latency_ms else 0.0
    )

    p50_frame_latency_ms = (
    float(np.percentile(list(per_frame_latency_ms.values()), 50))
    if per_frame_latency_ms else 0.0
    )
    
    
    metrics = {
        "modal_gpu_requested": "A100-40GB", # A100-80GB costs a bit more # before: L40S
        "gpu_index": gpu_index,
        "gpu_name": gpu_name,
        "gpu_total_memory_gb": gpu_total_memory_gb,
        "input_video": filename,
        "source_width": source_width,
        "source_height": source_height,
        "source_frame_count": source_frame_count,
        "source_fps": source_fps,
        "processed_frame_count": processed_frame_count,
        "inference_total_s": inference_total_s,
        "inference_fps": inference_fps,
        "inference_time_per_frame_ms_avg": avg_inference_ms,
        "per_frame_latency_ms_avg": avg_latency_ms,
        "p50_frame_latency_ms": p50_frame_latency_ms,
        "end_to_end_ms_per_frame": end_to_end_ms_per_frame,
        "source_frame_duration_ms": source_frame_duration_ms,
        "rtf": rtf,
        "render_total_s": render_total_s,
        "render_fps": render_fps,
        "compression_total_s": compress_total_s,
        "total_elapsed_s": total_elapsed_s,
        "end_to_end_fps": end_to_end_fps,
        "avg_inference_ms_per_frame": avg_inference_ms,
        "avg_render_ms_per_frame": avg_render_ms,
                "per_frame_latency_ms": {
            str(k): v for k, v in sorted(per_frame_latency_ms.items())
        },
        "per_frame_inference_ms": {
            str(k): v for k, v in sorted(per_frame_inference_ms.items())
        },
        "per_frame_render_ms": {
            str(k): v for k, v in sorted(per_frame_render_ms.items())
        },
    }

    print("=" * 80)
    print("SAM3 PERFORMANCE SUMMARY")
    print(f"GPU                    : {gpu_name}")
    print(f"Processed frames       : {processed_frame_count}")
    print(f"Inference total (s)    : {inference_total_s:.3f}")
    print(f"Inference FPS          : {inference_fps:.3f}")
    print(f"Inference time/frame ms: {avg_inference_ms:.3f}")
    print(f"Render total (s)       : {render_total_s:.3f}")
    print(f"Render FPS             : {render_fps:.3f}")
    print(f"Avg render/frame ms    : {avg_render_ms:.3f}")
    print(f"Per-frame latency avg ms: {avg_latency_ms:.3f}")
    print(f"P50 frame latency ms    : {p50_frame_latency_ms:.3f}")
    print(f"Compression total (s)  : {compress_total_s:.3f}")
    print(f"End-to-end total (s)   : {total_elapsed_s:.3f}")
    print(f"End-to-end FPS         : {end_to_end_fps:.3f}")
    print(f"End-to-end ms/frame    : {end_to_end_ms_per_frame:.3f}")
    print(f"RTF                    : {rtf:.3f}")
    print("=" * 80)

    metrics_path = workdir / f"{video_path.stem}-metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    output_remote_path = f"/{compressed_video.name}"
    metrics_remote_path = f"/{metrics_path.name}"

    Path("/outputs", compressed_video.name).write_bytes(compressed_video.read_bytes())
    Path("/outputs", metrics_path.name).write_text(metrics_path.read_text())

    vol.commit()

    print(f"Saved result video to Modal Volume: {output_remote_path}")
    print(f"Saved metrics JSON to Modal Volume: {metrics_remote_path}")

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
    
    # Download the output video and metrics JSON from Modal Volume to local machine
    print("Saved output in Modal Volume 'sam3-video-results' at:", remote_output)
    print(
        f"Download video with: modal volume get sam3-video-results {remote_output} ./"
    )
    print(
        f"Download metrics with: modal volume get sam3-video-results /{Path(video_path).stem}-metrics.json ./"
    )
