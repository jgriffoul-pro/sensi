
def choisir_source_video() -> int | str:
    """
    Propose à l'utilisateur de choisir entre :
    1) un flux caméra direct ;
    2) une vidéo .mp4.

    Retourne soit un index caméra, soit un chemin de fichier vidéo.
    """

    print("\nChoisis la source vidéo :")
    print("1 - Flux direct caméra")
    print("2 - Fichier vidéo .mp4")

    choix = input("Ton choix [1/2] : ").strip()

    if choix == "1":
        saisie_index = input(f"Index caméra [{CAMERA_INDEX}] : ").strip()
        return int(saisie_index) if saisie_index else CAMERA_INDEX

    if choix == "2":
        chemin_video = input("Chemin du fichier .mp4 : ").strip().strip('"')
        video_path = Path(chemin_video).expanduser()

        if not video_path.exists():
            raise FileNotFoundError(f"Vidéo introuvable : {video_path}")

        if video_path.suffix.lower() != ".mp4":
            print("Attention : le fichier choisi n'est pas en .mp4, mais OpenCV va quand même essayer de le lire.")

        return str(video_path)

    raise ValueError("Choix invalide. Relance le script et choisis 1 ou 2.")


