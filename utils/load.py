import numpy as np
import os
import pickle


def load_embeddings(filepath):
    assert os.path.isfile(filepath), f"{filepath} does not exist"
    with open(filepath, 'rb') as f:
        embeddings = pickle.load(f)
    assert isinstance(embeddings, dict), f"{filepath} should be a dictionary"

    # Assert embedding format
    for traj_id, emb in embeddings.items():
        assert isinstance(emb, np.ndarray), "embeddings should be type np.ndarray"
        if len(emb.shape) == 1:
            embeddings[traj_id] = np.expand_dims(emb, axis=0)

    return embeddings
