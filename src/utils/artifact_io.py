import joblib
import os

def save_artifact(model, artifact_path):
    """Serialize a fitted model or artifact, creating its parent folder.

    Args:
        model: Python object supported by joblib, such as a fitted estimator
            or a dictionary containing a model and its preprocessing objects.
        artifact_path: Destination file path.

    Side effects:
        Creates the destination parent folder if necessary and writes the
        serialized artifact to disk.
    """
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(model, artifact_path)
