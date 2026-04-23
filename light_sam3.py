"""
LIGHT SAM3 — versione leggera della pipeline di segmentazione loghi.

Idea:
  - frame 0 = target, SAM3 viene eseguito sul frame target.
  - per ogni frame successivo, calcolo uno score di similarita' (correlazione
    di istogrammi HSV) rispetto al frame target.
  - se score < similarity_threshold => assumo cambio di camera / zoom /
    nuovi oggetti in scena: rieseguo SAM3 e il frame corrente diventa il
    nuovo target.
  - altrimenti riutilizzo le detection dell'ultimo frame target.

La pipeline e' online: ogni frame annotato viene scritto subito su disco
(cv2.VideoWriter), senza accumulare tutti gli output in memoria.
"""

import subprocess
from pathlib import Patoh

import modal

app = modal.App("light-sam3-online")

GPU_TYPE = "B200"
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
def segment_video_light(
    video_bytes: bytes,
    filename: str = "zoom_in_camera_change.mp4",
    prompt_text: str = "logo",
    similarity_threshold: float = 0.85,
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

    def mean_ms(v):   return float(np.mean(v))            if v else 0.0
    def median_ms(v): return float(np.percentile(v, 50))  if v else 0.0
    def p95_ms(v):    return float(np.percentile(v, 95))  if v else 0.0
    def std_ms(v):    return float(np.std(v))             if v else 0.0

    total_start = time.perf_counter()

    workdir = Path("/tmp/sam3_run")
    workdir.mkdir(parents=True, exist_ok=True)

    raw_video_path = workdir / filename
    raw_video_path.write_bytes(video_bytes)

    # Transcode in formato SAM3-safe (H264 + yuv420p)
    video_path = workdir / f"{raw_video_path.stem}_sam3_safe.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(raw_video_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
            str(video_path),
        ],
        check=True,
    )

    cap = cv2.VideoCapture(str(video_path))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if source_fps <= 0:
        cap.release()
        raise ValueError("Could not read a valid FPS from the input video.")

    video_duration_seconds = (
        source_frame_count / source_fps if source_frame_count > 0 else 0.0
    )

    target_video     = workdir / f"{video_path.stem}-light-result{video_path.suffix}"
    compressed_video = workdir / f"{video_path.stem}-light-result-compressed{video_path.suffix}"
    metrics_path     = workdir / f"{video_path.stem}-light-metrics.json"

    devices = [torch.cuda.current_device()]
    predictor = build_sam3_video_predictor(
        bpe_path="/root/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        gpus_to_use=devices,
    )

    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    print("=" * 80)
    print("LIGHT SAM3 — RUNTIME INFO")
    print(f"GPU                  : {gpu_name}")
    print(f"Input video          : {filename}")
    print(f"Resolution           : {source_width}x{source_height}")
    print(f"Frames               : {source_frame_count}")
    print(f"FPS                  : {source_fps:.3f}")
    print(f"Similarity threshold : {similarity_threshold}")
    print("=" * 80)

    # ------------------------------------------------------------
    # SAM3 su un singolo frame: scriviamo un mp4 di 1 frame, apriamo
    # una sessione SAM3, aggiungiamo il prompt testuale, propaghiamo
    # una sola iterazione. Manteniamo l'API video predictor gia' usata
    # negli altri script del progetto.
    # ------------------------------------------------------------
    single_dir = workdir / "single_frame"
    single_dir.mkdir(parents=True, exist_ok=True)

    def run_sam3_on_frame(frame_bgr: np.ndarray) -> dict:
        tmp_img = single_dir / "frame.png"
        tmp_vid = single_dir / "frame.mp4"
        cv2.imwrite(str(tmp_img), frame_bgr)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-loop", "1", "-i", str(tmp_img),
                "-frames:v", "1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(tmp_vid),
            ],
            check=True,
        )

        resp = predictor.handle_request(
            request={"type": "start_session", "resource_path": tmp_vid.as_posix()}
        )
        sid = resp["session_id"]
        predictor.handle_request(request={"type": "reset_session", "session_id": sid})
        predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": sid,
                "frame_index": 0,
                "text": prompt_text,
            }
        )
        stream = predictor.handle_stream_request(
            request={"type": "propagate_in_video", "session_id": sid}
        )
        output = None
        for response in stream:
            output = response["outputs"]
            break

        if output is None:
            h, w = frame_bgr.shape[:2]
            output = {
                "out_binary_masks": np.zeros((0, h, w), dtype=bool),
                "out_probs":        np.zeros((0,),      dtype=np.float32),
                "out_obj_ids":      np.zeros((0,),      dtype=np.int32),
            }
        return output

    # ------------------------------------------------------------
    # Similarita' frame-by-frame: correlazione di istogrammi HSV.
    # range [-1, 1], 1 == identici.
    # ------------------------------------------------------------
    def hsv_hist(frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [32, 32, 32],
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def similarity(h1: np.ndarray, h2: np.ndarray) -> float:
        return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))

    # ------------------------------------------------------------
    # Annotazione (stessa estetica degli altri script del progetto)
    # ------------------------------------------------------------
    color = sv.ColorPalette.from_hex(
        [
            "#ffff00", "#ff9b00", "#ff8080", "#ff66b2",
            "#ff66ff", "#b266ff", "#9999ff", "#3399ff",
            "#66ffff", "#33ff99", "#66ff66", "#99ff00",
        ]
    )

    def from_sam(result: dict) -> sv.Detections:
        return sv.Detections(
            xyxy=sv.mask_to_xyxy(result["out_binary_masks"]),
            mask=result["out_binary_masks"],
            confidence=result["out_probs"],
            tracker_id=result["out_obj_ids"],
        )

    def annotate(img: np.ndarray, detections: sv.Detections, text: str | None) -> np.ndarray:
        h, w, _ = img.shape
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(w, h))
        out = sv.MaskAnnotator(
            color=color, color_lookup=sv.ColorLookup.TRACK, opacity=0.6
        ).annotate(img.copy(), detections)
        if text and len(detections) > 0:
            out = sv.LabelAnnotator(
                color=color,
                color_lookup=sv.ColorLookup.TRACK,
                text_scale=text_scale,
                text_color=sv.Color.BLACK,
                text_position=sv.Position.TOP_CENTER,
                text_offset=(0, -30),
            ).annotate(
                out, detections,
                [f"#{tid} {text}" for tid in detections.tracker_id],
            )
        return out

    # ------------------------------------------------------------
    # Loop ONLINE: un frame alla volta, niente buffer globale.
    # ------------------------------------------------------------
    writer = cv2.VideoWriter(
        str(target_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (source_width, source_height),
    )

    target_hist: np.ndarray | None = None
    current_output: dict | None = None

    per_frame_latency_ms: list[float] = []
    per_frame_inference_ms: list[float] = []
    similarity_scores: list[float] = []
    rerun_frame_indices: list[int] = []

    frame_idx = 0
    while True:
        cuda_sync()
        t_in = time.perf_counter()

        ok, frame_bgr = cap.read()
        if not ok:
            break

        cur_hist = hsv_hist(frame_bgr)

        if target_hist is None:
            need_rerun = True
            sim = 1.0
        else:
            sim = similarity(target_hist, cur_hist)
            need_rerun = sim < similarity_threshold
        similarity_scores.append(sim)

        inf_ms = 0.0
        if need_rerun:
            cuda_sync()
            t0 = time.perf_counter()
            current_output = run_sam3_on_frame(frame_bgr)
            cuda_sync()
            inf_ms = (time.perf_counter() - t0) * 1000.0
            target_hist = cur_hist
            rerun_frame_indices.append(frame_idx)

        per_frame_inference_ms.append(inf_ms)

        detections = from_sam(current_output)
        annotated = annotate(frame_bgr, detections, prompt_text)
        writer.write(annotated)

        cuda_sync()
        per_frame_latency_ms.append((time.perf_counter() - t_in) * 1000.0)

        if frame_idx % 25 == 0 or need_rerun:
            print(
                f"[light] frame {frame_idx:5d}  sim={sim:.3f}  "
                f"{'RERUN' if need_rerun else '     '}  "
                f"inf_ms={inf_ms:7.1f}  e2e_ms={per_frame_latency_ms[-1]:7.1f}"
            )
        frame_idx += 1

    cap.release()
    writer.release()

    processed_frame_count = frame_idx
    total_processing_seconds = time.perf_counter() - total_start

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(target_video),
            "-vcodec", "libx264", "-crf", "28",
            str(compressed_video),
        ],
        check=True,
    )

    pipeline_throughput_fps = (
        processed_frame_count / total_processing_seconds
        if total_processing_seconds > 0 else 0.0
    )
    realtime_factor_rtf = (
        total_processing_seconds / video_duration_seconds
        if video_duration_seconds > 0 else 0.0
    )
    meets_realtime_constraint = realtime_factor_rtf <= 1.0

    metrics = {
        "input_video": filename,
        "mode": "LIGHT_ONLINE",
        "gpu_requested": GPU_TYPE,
        "gpu_reported": gpu_name,
        "num_frames": source_frame_count,
        "resolution": f"{source_width}x{source_height}",
        "video_duration_seconds": round(float(video_duration_seconds), 3),
        "source_fps": round(source_fps, 3),
        "similarity_threshold": similarity_threshold,
        "num_rerun_frames": len(rerun_frame_indices),
        "rerun_frame_indices": rerun_frame_indices,
        "metrics": {
            "pipeline_throughput_fps":    round(pipeline_throughput_fps, 3),
            "total_processing_seconds":   round(total_processing_seconds, 3),
            "realtime_factor_rtf":        round(realtime_factor_rtf, 3),
            "meets_realtime_constraint":  bool(meets_realtime_constraint),
            "mean_inference_ms_per_frame": round(mean_ms(per_frame_inference_ms), 3),
            "p50_inference_ms_per_frame":  round(median_ms(per_frame_inference_ms), 3),
            "p95_inference_ms_per_frame":  round(p95_ms(per_frame_inference_ms), 3),
            "std_inference_ms_per_frame":  round(std_ms(per_frame_inference_ms), 3),
            "mean_end_to_end_latency_ms":  round(mean_ms(per_frame_latency_ms), 3),
            "p95_end_to_end_latency_ms":   round(p95_ms(per_frame_latency_ms), 3),
            "mean_similarity_score":       round(mean_ms(similarity_scores), 4),
        },
    }

    print("=" * 80)
    print("LIGHT SAM3 — SUMMARY")
    print(f"GPU                     : {gpu_name}")
    print(f"Processed frames        : {processed_frame_count}")
    print(f"SAM3 re-run frames      : {len(rerun_frame_indices)} "
          f"({100.0*len(rerun_frame_indices)/max(1,processed_frame_count):.2f}%)")
    print(f"Re-run @ frame indices  : {rerun_frame_indices}")
    print(f"Throughput FPS          : {pipeline_throughput_fps:.3f}")
    print(f"Total seconds           : {total_processing_seconds:.3f}")
    print(f"Realtime factor         : {realtime_factor_rtf:.3f}  "
          f"(meets RT: {meets_realtime_constraint})")
    print(f"Mean inference ms/frame : {metrics['metrics']['mean_inference_ms_per_frame']}")
    print(f"Mean e2e latency ms     : {metrics['metrics']['mean_end_to_end_latency_ms']}")
    print("=" * 80)

    metrics_path.write_text(json.dumps(metrics, indent=2))

    output_remote_path  = f"/{compressed_video.name}"
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
    video_path: str = "zoom_in_camera_change.mp4",
    prompt_text: str = "logo",
    similarity_threshold: float = 0.85,
):
    local_video = Path(video_path)
    if not local_video.exists():
        raise FileNotFoundError(f"Input video not found: {local_video}")

    result = segment_video_light.remote(
        video_bytes=local_video.read_bytes(),
        filename=local_video.name,
        prompt_text=prompt_text,
        similarity_threshold=similarity_threshold,
    )

    print("\n=== LIGHT SAM3 run completed ===")
    print(f"Saved output in Modal Volume '{VOLUME_NAME}' at: {result['output_video']}")
    print(f"Saved metrics in Modal Volume '{VOLUME_NAME}' at: {result['metrics_json']}")

    local_output_dir = Path("modal_outputs")
    local_output_dir.mkdir(parents=True, exist_ok=True)

    remote_video_path   = result["output_video"]
    remote_metrics_path = result["metrics_json"]
    local_video_path    = local_output_dir / Path(remote_video_path).name
    local_metrics_path  = local_output_dir / Path(remote_metrics_path).name

    print("\nDownloading artifacts locally from Modal volume...")
    subprocess.run(
        ["modal", "volume", "get", "--force",
         VOLUME_NAME, remote_video_path, str(local_video_path)],
        check=True,
    )
    subprocess.run(
        ["modal", "volume", "get", "--force",
         VOLUME_NAME, remote_metrics_path, str(local_metrics_path)],
        check=True,
    )

    print("\nDownloaded locally to:")
    print(local_video_path)
    print(local_metrics_path)
