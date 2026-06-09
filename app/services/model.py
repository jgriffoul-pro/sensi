from app.services.preprocessing import read_sequence_from_file

# ============================================================
# MODIF 001 — Mapping vocabulaire BiLSTM → NLP
# Réf : modifs/001_mapping_vocabulaire.txt
# ============================================================
# Le modèle BiLSTM v6 produit des labels qui ne correspondent
# pas au vocabulaire BARThez. Ce dictionnaire corrige les
# incohérences détectées.
# ============================================================

LABEL_MAP = {
    "SOURDE": "SOURD",
    "ENTENDANT": "ENTENDANTS",
}


def get_glosses_from_sequence() -> list[str]:
    """
    Récupère les glosses depuis output/sequence.txt (écrit par test_team_live).
    Filtre le signe 'INCONNU' qui ne doit pas être transmis au NLP.
    Applique le mapping LABEL_MAP pour corriger les incohérences BiLSTM→NLP.

    Returns:
        list[str] : liste de glosses en majuscules, sans INCONNU, mappées
                    ex: ["BONJOUR", "JE_SUIS", "CONTENT"]
    """
    glosses = read_sequence_from_file()

    # Filtre le signe inconnu — ne doit pas être transmis au modèle NLP
    glosses = [g for g in glosses if g != "INCONNU"]

    # MODIF 001 — Mapping vocabulaire BiLSTM → NLP
    glosses = [LABEL_MAP.get(g, g) for g in glosses]

    return glosses


# ============================================================
# FONCTION EN ATTENTE — Inférence LSTM directe (option future)
# À décommenter si on intègre la webcam directement dans l'API
# ============================================================

# def predict_sign(sequence: np.ndarray) -> tuple[str, float]:
#     """
#     Prédit le signe depuis une séquence de 60 frames x 150 features.
#
#     Args:
#         sequence : np.ndarray de shape (60, 150)
#
#     Returns:
#         tuple : (signe prédit en minuscules, confiance)
#                 ex: ("bonjour", 0.92)
#     """
#     batch = sequence[np.newaxis, ...]   # (1, 60, 150)
#     probs = model_lstm.predict(batch, verbose=0)[0]
#     pred_idx = int(np.argmax(probs))
#     confidence = float(probs[pred_idx])
#     sign = idx_to_sign[pred_idx]
#     return sign, confidence
