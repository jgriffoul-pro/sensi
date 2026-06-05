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
    "mouvement_4",
    "mouvement_5",
    "mouvement_6",
    "mouvement_7",
    "mouvement_8",
    "mouvement_9",
    "mouvement_10",
    "mouvement_11",
    "mouvement_12",
    "mouvement_13",
    "mouvement_14",
    "mouvement_15",
    "mouvement_16",
    "mouvement_17",
    "mouvement_18",
]

PROMPTEUR_DIR = Path("prompteur")

PROMPT_VIDEOS = {
    "mouvement_1": PROMPTEUR_DIR / "aider.mp4",
    "mouvement_2": PROMPTEUR_DIR / "ameliorer.mp4",
    "mouvement_3": PROMPTEUR_DIR / "ami.mp4",
    "mouvement_4": PROMPTEUR_DIR / "aujourd'hui.mp4",
    "mouvement_5": PROMPTEUR_DIR / "bonjour.mp4",
    "mouvement_6": PROMPTEUR_DIR / "communiquer.mp4",
    "mouvement_7": PROMPTEUR_DIR / "entendant.mp4",
    "mouvement_8": PROMPTEUR_DIR / "content.mp4",
    "mouvement_9": PROMPTEUR_DIR / "je_suis.mp4",
    "mouvement_10": PROMPTEUR_DIR / "je_veux.mp4",
    "mouvement_11": PROMPTEUR_DIR / "langue_des_signes.mp4",
    "mouvement_12": PROMPTEUR_DIR / "merci.mp4",
    "mouvement_13": PROMPTEUR_DIR / "outil_pointage.mp4",
    "mouvement_14": PROMPTEUR_DIR / "outil.mp4",
    "mouvement_15": PROMPTEUR_DIR / "présenter.mp4",
    "mouvement_16": PROMPTEUR_DIR / "projet.mp4",
    "mouvement_17": PROMPTEUR_DIR / "sourd_pointage.mp4",
    "mouvement_18": PROMPTEUR_DIR / "sourd.mp4",
}

VIDEOS_PER_MOVEMENT = 10
SECONDS_PER_VIDEO = 2
FPS = 30

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

WINDOW_NAME = "Capture X_train - ESPACE pour enregistrer - q pour quitter"

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

# =========================
# MEDIAPIPE
# =========================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
def play_prompt_video(video_path):
    """
    Joue la vidéo de référence.

    ESPACE = démarrer l'enregistrement
    q = quitter
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Impossible d'ouvrir {video_path}")
        return False

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.putText(
            frame,
            "ESPACE = reproduire le geste",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Prompteur", frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            cap.release()
            cv2.destroyWindow("Prompteur")
            return False

        if key == 32:
            cap.release()
            cv2.destroyWindow("Prompteur")
            return True

def landmarks_to_vector(results) -> np.ndarray:
    """
    Convertit les landmarks MediaPipe en vecteur fixe.

    Contenu :
    - pose      : 33 points x 4 valeurs = 132
    - visage    : 468 points x 3 valeurs = 1404
    - main G    : 21 points x 3 valeurs = 63
    - main D    : 21 points x 3 valeurs = 63

    Total : 1662 valeurs par frame.
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
    """Dessine pose + visage + mains sur l'image."""

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
        )

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )

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


# =========================
# CAPTURE
# =========================

def wait_for_space(cap, holistic, movement_name: str, take: int) -> bool:
    """Affiche la webcam et attend ESPACE pour démarrer."""

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

        put_text(preview, f"Mouvement : {movement_name}", 40, scale=0.9)
        put_text(preview, f"Prise : {take}/{VIDEOS_PER_MOVEMENT}", 80, scale=0.9)
        put_text(
            preview,
            "Appuie sur ESPACE pour enregistrer",
            130,
            color=(0, 255, 255),
            scale=0.9,
            thickness=3,
        )
        put_text(preview, "Appuie sur q pour quitter", 170, color=(200, 200, 200), scale=0.7)

        hands_detected = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
        face_detected = results.face_landmarks is not None

        put_text(
            preview,
            f"Visage: {'OK' if face_detected else 'NON'} | Mains: {'OK' if hands_detected else 'NON'}",
            FRAME_HEIGHT - 30,
            color=(0, 255, 0) if face_detected and hands_detected else (0, 165, 255),
            scale=0.7,
        )

        cv2.imshow(WINDOW_NAME, preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return False

        if key == 32:
            return True


def record_take(cap, holistic, movement_name: str, take: int, movement_dir: Path) -> bool:
    """Enregistre une prise : ESPACE valide, ENTREE refait."""

    total_frames = int(SECONDS_PER_VIDEO * FPS)

    video_path = movement_dir / f"{movement_name}_{take:02d}.mp4"
    npy_path = movement_dir / f"{movement_name}_{take:02d}.npy"

    video_frames = []
    landmarks_sequence = []

    print(f"REC {movement_name} - prise {take}/{VIDEOS_PER_MOVEMENT}")

    start_time = time.time()
    frame_count = 0

    while frame_count < total_frames:
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

        put_text(
            annotated_frame,
            f"REC {movement_name} {take}/{VIDEOS_PER_MOVEMENT}",
            40,
            color=(0, 0, 255),
            scale=0.9,
            thickness=3,
        )

        put_text(
            annotated_frame,
            f"Temps restant : {remaining:.1f}s",
            80,
            scale=0.8,
        )

        put_text(
            annotated_frame,
            f"Frame : {frame_count + 1}/{total_frames}",
            120,
            scale=0.7,
        )

        video_frames.append(annotated_frame.copy())

        cv2.imshow(WINDOW_NAME, annotated_frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    while True:
        preview = video_frames[-1].copy()

        put_text(
            preview,
            "ESPACE = valider la prise",
            180,
            color=(0, 255, 0),
            scale=0.9,
            thickness=3,
        )

        put_text(
            preview,
            "ENTREE = refaire la prise",
            225,
            color=(0, 165, 255),
            scale=0.9,
            thickness=3,
        )

        put_text(
            preview,
            "q = quitter",
            270,
            color=(200, 200, 200),
            scale=0.75,
        )

        cv2.imshow(WINDOW_NAME, preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return False

        if key in (10, 13):
            print("Prise refusée. On recommence.")
            return False

        if key == 32:
            break

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Impossible de créer la vidéo : {video_path}")

    for saved_frame in video_frames:
        writer.write(saved_frame)

    writer.release()

    landmarks_array = np.array(landmarks_sequence, dtype=np.float32)
    np.save(npy_path, landmarks_array)

    print(f"Sauvé : {video_path}")
    print(f"Sauvé : {npy_path}")
    print(f"Shape landmarks : {landmarks_array.shape}")

    return True

def create_new_take_dir(output_root: Path) -> Path:
    """
    Crée un nouveau dossier de session :
    train_data/take01, train_data/take02, etc.

    À chaque lancement du script, on utilise le prochain numéro disponible.
    """

    output_root.mkdir(parents=True, exist_ok=True)

    existing_takes = [
        path for path in output_root.iterdir()
        if path.is_dir() and path.name.startswith("take")
    ]

    numbers = []

    for path in existing_takes:
        suffix = path.name.replace("take", "")
        if suffix.isdigit():
            numbers.append(int(suffix))

    next_number = max(numbers, default=0) + 1
    take_dir = output_root / f"take{next_number:02d}"
    take_dir.mkdir(parents=True, exist_ok=False)

    return take_dir


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

    print("Capture dataset lancée.")
    print("Barre ESPACE : lancer une prise.")
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

            for take in range(1, VIDEOS_PER_MOVEMENT + 1):
                prompt_video = PROMPT_VIDEOS[movement_name]

                ok = play_prompt_video(prompt_video)

                if not ok:
                    return
                should_record = wait_for_space(
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

                while True:
                    completed = record_take(
                        cap=cap,
                        holistic=holistic,
                        movement_name=movement_name,
                        take=take,
                        movement_dir=movement_dir,
                    )

                    if completed:
                        break

                    print("On refait la même prise.")

    cap.release()
    cv2.destroyAllWindows()

    print("\nCapture terminée.")
    print(f"Dataset sauvegardé dans : {output_dir.resolve()}")


if __name__ == "__main__":
    capture_dataset()
