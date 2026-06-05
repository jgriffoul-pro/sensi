
def build_output_paths(output_dir: Path, saved_count: int) -> tuple[Path, Path]:
    """Crée les chemins synchronisés .npy et .mp4."""

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    milliseconds = int((time.time() % 1) * 1000)
    base_name = f"gesture_{timestamp}_{milliseconds:03d}_{saved_count:03d}"

    npy_path = output_dir / f"{base_name}.npy"
    mp4_path = output_dir / f"{base_name}.mp4"

    return npy_path, mp4_path


def save_landmarks_sequence(sequence: list[np.ndarray], output_path: Path) -> None:
    """Sauvegarde les landmarks en .npy."""
    arr = np.array(sequence, dtype=np.float32)
    np.save(output_path, arr)


def save_video_clip(frames: list[np.ndarray], output_path: Path, fps: int = OUTPUT_FPS) -> None:
    """Sauvegarde une liste de frames OpenCV en MP4."""

    if not frames:
        raise ValueError("Aucune frame vidéo à sauvegarder.")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Impossible de créer la vidéo : {output_path}")

    for frame in frames:
        writer.write(frame)

    writer.release()


def save_hybrid_sequence(
    landmark_sequence: list[np.ndarray],
    video_sequence: list[np.ndarray],
    output_dir: Path,
    saved_count: int,
) -> tuple[Path, Path]:
    """Sauvegarde .npy + .mp4 pour le même geste."""

    npy_path, mp4_path = build_output_paths(output_dir, saved_count)

    save_landmarks_sequence(landmark_sequence, npy_path)
    save_video_clip(video_sequence, mp4_path)

    return npy_path, mp4_path

