from pathlib import Path

import joblib


def save_artifact(model, artifact_path):
    """Serialize a fitted model or artifact, creating its parent folder.

    Args:
        model: Python object supported by joblib, such as a fitted estimator
            or a dictionary of preprocessing metadata. Keras networks should
            use their native ``.keras`` format.
        artifact_path: Destination file path.

    Side effects:
        Creates the destination parent folder if necessary and writes the
        serialized artifact to disk.
    """
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
