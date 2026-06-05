from pathlib import Path
import time

import cv2
import numpy as np
import mediapipe as mp


# =========================
# CONFIGURATION
# =========================

CAMERA_INDEX = 0

OUTPUT_ROOT = Path("train_data")

MOVEMENTS = [
    "mouvement_1",
    "mouvement_2",
    "mouvement_3",
]

VIDEOS_PER_MOVEMENT = 10
SECONDS_PER_VIDEO = 3
FPS = 30

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

WINDOW_NAME = "Capture X_train - ESPACE valider - ENTREE refaire - q quitter"


# =========================
# MEDIAPIPE
# =========================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


FACE_LANDMARKS_SELECTED = [
    # bouche
    0, 13, 14, 17, 37, 39, 40, 61, 78, 81, 82, 87,
    178, 181, 185, 191, 267, 269, 270, 291, 308, 311,
    312, 317, 402, 405, 409, 415,

    # oeil gauche
    33, 133, 159, 145, 153, 154, 155, 173,

    # oeil droit
    263, 362, 386, 374, 380, 381, 382, 398,

    # sourcils
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
]


def landmarks_to_vector(results) -> np.ndarray:
    """
    Vecteur simplifié :
    - pose complète : 33 x 4 = 132
    - visage sélectionné : len(FACE_LANDMARKS_SELECTED) x 3
    - main gauche : 21 x 3 = 63
    - main droite : 21 x 3 = 63
    """

    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    if results.face_landmarks:
        face = np.array(
            [
                [
                    results.face_landmarks.landmark[i].x,
                    results.face_landmarks.landmark[i].y,
                    results.face_landmarks.landmark[i].z,
                ]
                for i in FACE_LANDMARKS_SELECTED
            ],
            dtype=np.float32,
        ).flatten()
    else:
        face = np.zeros(len(FACE_LANDMARKS_SELECTED) * 3, dtype=np.float32)

    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        left_hand = np.zeros(21 * 3, dtype=np.float32)

    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        right_hand = np.zeros(21 * 3, dtype=np.float32)

    return np.concatenate([pose, face, left_hand, right_hand])


def draw_landmarks(frame, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    return frame


def put_text(frame, text, y, color=(255, 255, 255), scale=0.8, thickness=2):
    cv2.putText(
        frame,
        text,
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def create_new_take_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)

    numbers = []

    for path in output_root.iterdir():
        if path.is_dir() and path.name.startswith("take"):
            suffix = path.name.replace("take", "")
            if suffix.isdigit():
                numbers.append(int(suffix))

    take_number = max(numbers, default=0) + 1
    take_dir = output_root / f"take{take_number:02d}"
    take_dir.mkdir(parents=True, exist_ok=False)

    return take_dir


# =========================
# INTERACTION
# =========================

def wait_for_space_before_record(cap, holistic, movement_name: str, take: int) -> bool:
    """
    Avant chaque prise :
    - ESPACE : lancer l'enregistrement
    - q : quitter
    """

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame non lue.")
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        preview = frame.copy()
        preview = draw_landmarks(preview, results)

        hands_ok = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
        face_ok = results.face_landmarks is not None

        put_text(preview, f"Mouvement : {movement_name}", 40, scale=0.9)
        put_text(preview, f"Prise : {take}/{VIDEOS_PER_MOVEMENT}", 80, scale=0.9)
        put_text(preview, "ESPACE = lancer l'enregistrement", 130, color=(0, 255, 255), scale=0.9, thickness=3)
        put_text(preview, "q = quitter", 170, color=(200, 200, 200), scale=0.7)
        put_text(
            preview,
            f"Visage: {'OK' if face_ok else 'NON'} | Mains: {'OK' if hands_ok else 'NON'}",
            FRAME_HEIGHT - 30,
            color=(0, 255, 0) if face_ok and hands_ok else (0, 165, 255),
            scale=0.7,
        )

        cv2.imshow(WINDOW_NAME, preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return False

        if key == 32:
            return True


def ask_validation(last_frame, movement_name: str, take: int) -> str:
    """
    Après chaque prise :
    - ESPACE : valider et sauvegarder
    - ENTREE : refaire sans sauvegarder
    - q : quitter
    """

    while True:
        frame = last_frame.copy()

        put_text(frame, f"Capture terminee : {movement_name} {take}/{VIDEOS_PER_MOVEMENT}", 40, scale=0.9)
        put_text(frame, "ESPACE = valider la prise", 100, color=(0, 255, 0), scale=0.9, thickness=3)
        put_text(frame, "ENTREE = refaire la prise", 145, color=(0, 165, 255), scale=0.9, thickness=3)
        put_text(frame, "q = quitter", 190, color=(200, 200, 200), scale=0.75)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return "quit"

        if key == 32:
            return "valid"

        if key in (10, 13):
            return "redo"


# =========================
# CAPTURE
# =========================

def record_take(cap, holistic, movement_name: str, take: int, movement_dir: Path) -> str:
    """
    Enregistre une prise en mémoire.
    Sauvegarde seulement si l'utilisateur valide avec ESPACE.
    """

    total_frames = int(SECONDS_PER_VIDEO * FPS)

    landmarks_sequence = []
    video_frames = []
    last_frame = None

    print(f"REC {movement_name} - prise {take}/{VIDEOS_PER_MOVEMENT}")

    start_time = time.time()

    for frame_count in range(total_frames):
        ret, frame = cap.read()

        if not ret:
            print("Frame non lue.")
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        vector = landmarks_to_vector(results)
        landmarks_sequence.append(vector)

        annotated_frame = frame.copy()
        annotated_frame = draw_landmarks(annotated_frame, results)

        elapsed = time.time() - start_time
        remaining = max(0, SECONDS_PER_VIDEO - elapsed)

        put_text(annotated_frame, f"REC {movement_name} {take}/{VIDEOS_PER_MOVEMENT}", 40, color=(0, 0, 255), scale=0.9, thickness=3)
        put_text(annotated_frame, f"Temps restant : {remaining:.1f}s", 80)
        put_text(annotated_frame, f"Frame : {frame_count + 1}/{total_frames}", 120, scale=0.7)

        video_frames.append(annotated_frame.copy())
        last_frame = annotated_frame.copy()

        cv2.imshow(WINDOW_NAME, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return "quit"

    if last_frame is None:
        return "redo"

    decision = ask_validation(last_frame, movement_name, take)

    if decision != "valid":
        print("Prise non validee.")
        return decision

    video_path = movement_dir / f"{movement_name}_{take:02d}.mp4"
    npy_path = movement_dir / f"{movement_name}_{take:02d}.npy"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    if not writer.isOpened():
        raise RuntimeError(f"Impossible de créer la vidéo : {video_path}")

    for saved_frame in video_frames:
        writer.write(saved_frame)

    writer.release()

    landmarks_array = np.array(landmarks_sequence, dtype=np.float32)
    np.save(npy_path, landmarks_array)

    print("Prise validee.")
    print(f"Sauvé : {video_path}")
    print(f"Sauvé : {npy_path}")
    print(f"Shape landmarks : {landmarks_array.shape}")

    return "valid"


def capture_dataset() -> None:
    output_dir = create_new_take_dir(OUTPUT_ROOT)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la webcam CAMERA_INDEX={CAMERA_INDEX}. "
            "Essaie CAMERA_INDEX = 1 ou 2."
        )

    print("Capture dataset lancee.")
    print("Avant prise : ESPACE pour enregistrer.")
    print("Apres prise : ESPACE pour valider, ENTREE pour refaire.")
    print("q : quitter.")
    print(f"Sortie : {output_dir.resolve()}")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as holistic:

        for movement_name in MOVEMENTS:
            movement_dir = output_dir / movement_name
            movement_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== Mouvement : {movement_name} ===")

            take = 1

            while take <= VIDEOS_PER_MOVEMENT:
                should_record = wait_for_space_before_record(
                    cap=cap,
                    holistic=holistic,
                    movement_name=movement_name,
                    take=take,
                )

                if not should_record:
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Capture interrompue.")
                    return

                result = record_take(
                    cap=cap,
                    holistic=holistic,
                    movement_name=movement_name,
                    take=take,
                    movement_dir=movement_dir,
                )

                if result == "valid":
                    take += 1

                elif result == "redo":
                    print("On refait la même prise.")

                elif result == "quit":
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Capture interrompue.")
                    return

    cap.release()
    cv2.destroyAllWindows()

    print("\nCapture terminee.")
    print(f"Dataset sauvegarde dans : {output_dir.resolve()}")


if __name__ == "__main__":
    capture_dataset()
