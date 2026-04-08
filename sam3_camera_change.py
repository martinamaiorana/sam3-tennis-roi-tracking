import os
import subprocess
from pathlib import Path

import modal

app = modal.App("sam3-tennis-video-segmentation")

GPU_TYPE = "A100-80GB" # "RTX-PRO-6000" 
VOLUME_NAME = "sam3-video-results"

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

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60,
    volumes={"/outputs": vol},
    secrets=[modal.Secret.from_name("hf-token")],
)
def segment_video(
    video_bytes: bytes,
    filename: str = "moving_camera.mp4",
    prompt_text: str = "logo",
) -> dict:
    import cv2
    import json
    import time
    import numpy as np
    import supervision as sv
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    def cuda_sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def mean_ms(values_ms: list[float]) -> float:
        return float(np.mean(values_ms)) if values_ms else 0.0

    def median_ms(values_ms: list[float]) -> float:
        return float(np.percentile(values_ms, 50)) if values_ms else 0.0

    def std_ms(values_ms: list[float]) -> float:
        return float(np.std(values_ms)) if values_ms else 0.0

    def p95_ms(values_ms: list[float]) -> float:
        return float(np.percentile(values_ms, 95)) if values_ms else 0.0

    total_start = time.perf_counter()

    workdir = Path("/tmp/sam3_run")
    workdir.mkdir(parents=True, exist_ok=True)

    # video_path = workdir / filename
    # video_path.write_bytes(video_bytes)
    # Replaced below:
    # Save raw video
    raw_video_path = workdir / filename
    raw_video_path.write_bytes(video_bytes)

    # Transcode to SAM3-safe format (H264 + yuv420p)
    video_path = workdir / f"{raw_video_path.stem}_sam3_safe.mp4"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_video_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(video_path),
        ],
        check=True,
    )
    # Replaced up to here
    
    cap = cv2.VideoCapture(str(video_path))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if source_fps <= 0:
        raise ValueError("Could not read a valid FPS from the input video.")

    video_duration_seconds = (
        source_frame_count / source_fps if source_frame_count > 0 else 0.0
    )

    target_video = workdir / f"{video_path.stem}-result{video_path.suffix}"
    compressed_video = workdir / f"{video_path.stem}-result-compressed{video_path.suffix}"
    metrics_path = workdir / f"{video_path.stem}-metrics.json"

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

    predictor.handle_request(
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

    frame_outputs: dict[int, dict] = {}

    # Per-frame timing
    frame_input_time_s: dict[int, float] = {}
    frame_output_time_s: dict[int, float] = {}

    per_frame_inference_ms: dict[int, float] = {}
    per_frame_e2e_ms: dict[int, float] = {}

    # ------------------------------------------------------------
    # INFERENCE PASS
    # inference_ms(frame) = inference_end - inference_start
    # ------------------------------------------------------------
    stream = predictor.handle_stream_request(
        request={"type": "propagate_in_video", "session_id": session_id}
    )

    while True:
        cuda_sync()
        inference_start = time.perf_counter()

        try:
            response = next(stream)
        except StopIteration:
            break

        cuda_sync()
        inference_end = time.perf_counter()

        frame_index = int(response["frame_index"])
        frame_input_time_s[frame_index] = inference_start
        per_frame_inference_ms[frame_index] = (inference_end - inference_start) * 1000.0
        frame_outputs[frame_index] = response["outputs"]

    processed_frame_count = len(frame_outputs) #source_frame_count

    # ------------------------------------------------------------
    # RENDER / SAVE PASS
    # e2e_ms(frame) = output_available_time - input_time
    # ------------------------------------------------------------
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        output = frame_outputs[index]
        detections = from_sam(output)

        annotated = annotate(frame, detections, text=None) # remove Labels in the output video
        # annotated = annotate(frame, detections, prompt_text) # with ID labels on the output video

        frame_output_time_s[index] = time.perf_counter()

        if index in frame_input_time_s:
            per_frame_e2e_ms[index] = (
                frame_output_time_s[index] - frame_input_time_s[index]
            ) * 1000.0

        return annotated

    sv.process_video(
        source_path=str(video_path),
        target_path=str(target_video),
        callback=callback,
    )
    
    total_processing_seconds = time.perf_counter() - total_start
    
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

    pipeline_throughput_fps = (
        processed_frame_count / total_processing_seconds
        if total_processing_seconds > 0
        else 0.0
    )

    realtime_factor_rtf = (
        total_processing_seconds / video_duration_seconds
        if video_duration_seconds > 0
        else 0.0
    )
    meets_realtime_constraint = realtime_factor_rtf <= 1.0

    inference_values_ms = [
        per_frame_inference_ms[k] for k in sorted(per_frame_inference_ms.keys())
    ]
    e2e_values_ms = [
        per_frame_e2e_ms[k] for k in sorted(per_frame_e2e_ms.keys())
    ]

    mean_inference_ms_per_frame = mean_ms(inference_values_ms)
    p50_inference_ms_per_frame = median_ms(inference_values_ms)
    p95_inference_ms_per_frame = p95_ms(inference_values_ms)
    std_inference_ms_per_frame = std_ms(inference_values_ms)
    mean_end_to_end_latency_ms = mean_ms(e2e_values_ms)

    metrics = {
        "input_video": filename,
        "gpu_requested": GPU_TYPE,
        "gpu_reported": gpu_name,
        #"gpu_reported": {
            # "gpu_index": gpu_index,
            #"gpu_name": gpu_name,
            #"gpu_total_memory_gb": gpu_total_memory_gb,
        #},
        "num_frames": source_frame_count,
        "resolution": f"{source_width}x{source_height}",
        "video_duration_seconds": round(float(video_duration_seconds), 3),
        "source_fps": round(source_fps, 3),
        "metrics": {
            "pipeline_throughput_fps": round(pipeline_throughput_fps, 3),
            "total_processing_seconds": round(total_processing_seconds, 3),
            "realtime_factor_rtf": round(realtime_factor_rtf, 3),
            "meets_realtime_constraint": bool(meets_realtime_constraint),
            "mean_inference_ms_per_frame": round(mean_inference_ms_per_frame, 3),
            "p50_inference_ms_per_frame": round(p50_inference_ms_per_frame, 3),
            "p95_inference_ms_per_frame": round(p95_inference_ms_per_frame, 3),
            "std_inference_ms_per_frame": round(std_inference_ms_per_frame, 3),
            "mean_end_to_end_latency_ms": round(mean_end_to_end_latency_ms, 3),
        },
    }

    print("=" * 80)
    print("SAM3 PERFORMANCE SUMMARY")
    print(f"GPU                            : {gpu_name}")
    print(f"Processed frames               : {processed_frame_count}")
    print(f"Pipeline throughput FPS        : {pipeline_throughput_fps:.3f}")
    print(f"Total processing seconds       : {total_processing_seconds:.3f}")
    print(f"Realtime factor RTF           : {realtime_factor_rtf:.3f}")
    print(f"Meets realtime constraint      : {meets_realtime_constraint}")
    print(f"Mean inference ms/frame        : {mean_inference_ms_per_frame:.3f}")
    print(f"P50 inference ms/frame         : {p50_inference_ms_per_frame:.3f}")
    print(f"P95 inference ms/frame         : {p95_inference_ms_per_frame:.3f}")
    print(f"Std inference ms/frame         : {std_inference_ms_per_frame:.3f}")
    print(f"Mean end-to-end latency ms     : {mean_end_to_end_latency_ms:.3f}")
    print("=" * 80)

    metrics_path.write_text(json.dumps(metrics, indent=2))

    output_remote_path = f"/{compressed_video.name}"
    metrics_remote_path = f"/{metrics_path.name}"

    Path("/outputs", compressed_video.name).write_bytes(compressed_video.read_bytes())
    Path("/outputs", metrics_path.name).write_text(metrics_path.read_text())

    vol.commit()

    print(f"Saved result video to Modal Volume: {output_remote_path}")
    print(f"Saved metrics JSON to Modal Volume: {metrics_remote_path}")

    return {
        "output_video": output_remote_path,
        "metrics_json": metrics_remote_path,
        **metrics,
    }


@app.local_entrypoint()
def main(
    video_path: str = "moving_camera.mp4",
    prompt_text: str = "logo",
):
    local_video = Path(video_path)
    if not local_video.exists():
        raise FileNotFoundError(f"Input video not found: {local_video}")

    result = segment_video.remote(
        video_bytes=local_video.read_bytes(),
        filename=local_video.name,
        prompt_text=prompt_text,
    )

    print("\n=== SAM3 Modal run completed ===")
    print(f"Saved output in Modal Volume '{VOLUME_NAME}' at: {result['output_video']}")
    print(f"Saved metrics in Modal Volume '{VOLUME_NAME}' at: {result['metrics_json']}")

    local_output_dir = Path("modal_outputs")
    local_output_dir.mkdir(parents=True, exist_ok=True)

    remote_video_path = result["output_video"]
    remote_metrics_path = result["metrics_json"]

    local_video_path = local_output_dir / Path(remote_video_path).name
    local_metrics_path = local_output_dir / Path(remote_metrics_path).name

    print("\nDownloading artifacts locally from Modal volume...")

    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            "--force",
            VOLUME_NAME,
            remote_video_path,
            str(local_video_path),
        ],
        check=True,
    )

    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            "--force",
            VOLUME_NAME,
            remote_metrics_path,
            str(local_metrics_path),
        ],
        check=True,
    )

    print("\nDownloaded locally to:")
    print(local_video_path)
    print(local_metrics_path)