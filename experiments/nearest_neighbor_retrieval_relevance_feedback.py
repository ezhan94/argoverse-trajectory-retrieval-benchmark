import argparse
import numpy as np
import os
import pickle
import sys
sys.path.append(sys.path[0] + '/..')

from collections import defaultdict
from tqdm import tqdm

from data import TEST_QUERY_IDS, TEST_RETRIEVAL_IDS
from utils import load_embeddings


"""
Usage:

python experiments/nearest_neighbor_retrieval_relevance_feedback.py \
--embeddings_file embeddings/pretrain/AE.pkl \
--output_file submissions/AE_retrievals_relevance_feedback.pkl
"""


def baseb(n: int, b: int):
    """Convert n in decimal to base b."""
    e = n//b
    q = n%b
    if n == 0:
        return '0'
    elif e == 0:
        return str(q)
    else:
        return baseb(e, b) + str(q)


def get_nearest_neighbor_retrieval(query_embedding: np.ndarray, retrieval_embeddings: dict, n: int = 1):
    retrieval_embeddings_np = np.concatenate(list(retrieval_embeddings.values()), axis=0)
    query_embedding_repeat = np.repeat(query_embedding, retrieval_embeddings_np.shape[0], axis=0)

    distance = np.linalg.norm(retrieval_embeddings_np-query_embedding_repeat, axis=1) # Euclidean distance
    sorted_inds = np.argsort(distance) # get indices of increasing distance

    retrieval_ids = np.array(list(retrieval_embeddings.keys()))
    return retrieval_ids[sorted_inds[:n]].tolist()


def get_nearest_neighbor_retrieval_with_relevance_feedback(
    query_embedding: np.ndarray, retrieval_embeddings: dict, feedback_ids: list, feedback: str, n: int = 1):
    
    retrieval_embeddings_np = np.concatenate(list(retrieval_embeddings.values()), axis=0)
    query_embedding_repeat = np.repeat(query_embedding, retrieval_embeddings_np.shape[0], axis=0)

    set_A = [] # A is the relevant set
    set_B = [] # B is the non-relevant set

    distance_to_query = np.linalg.norm(retrieval_embeddings_np-query_embedding_repeat, axis=1)
    set_A.append(distance_to_query) # query is in A

    # Sort feedback trajectories in A or B
    for i, traj_id in enumerate(feedback_ids):
        traj_embedding = retrieval_embeddings[traj_id]
        traj_embedding_repeat = np.repeat(traj_embedding, retrieval_embeddings_np.shape[0], axis=0)
        distance_to_traj = np.linalg.norm(retrieval_embeddings_np-traj_embedding_repeat, axis=1)

        # Add to set A if relevance feedback label is 2, else B
        if feedback[i] == "2":
            set_A.append(distance_to_traj)
        else:
            set_B.append(distance_to_traj)

    # Compute average distances to A and B
    distance_to_A = sum(set_A)/len(set_A)
    distance_to_B = sum(set_B)/len(set_B) if len(set_B) > 0 else 0.0
    
    # Compute new ranking wrt. updated distance
    updated_distance = distance_to_A - distance_to_B
    sorted_inds = np.argsort(updated_distance) # get indices of increasing updated_distance

    retrieval_ids = np.array(list(retrieval_embeddings.keys()))
    return retrieval_ids[sorted_inds[:n]].tolist()
    

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_file', type=str,
                        required=True, default='',
                        help='embeddings pickle file')
    parser.add_argument('--output_file', type=str,
                        required=True, default='',
                        help='file to save retrievals')
    parser.add_argument('--num_retrievals', type=int,
                        required=False, default=50,
                        help='number of retrievals per query')
    parser.add_argument('--num_feedback', type=int,
                        required=False, default=5,
                        help='number of retrievals to receive feedback for')
    args = parser.parse_args()

    # Load embeddings
    embeddings = load_embeddings(args.embeddings_file)

    # Filter for test queries and retrievals
    test_query_embeddings = { traj_id: embeddings[traj_id] for traj_id in TEST_QUERY_IDS }
    test_retrieval_embeddings = { traj_id: embeddings[traj_id] for traj_id in TEST_RETRIEVAL_IDS }

    # Compute nearest neighbor retrievals
    retrievals = defaultdict(dict)
    for query_id in tqdm(TEST_QUERY_IDS):
        query_embedding = test_query_embeddings[query_id]
        retrievals[query_id]["feedback_set"] = get_nearest_neighbor_retrieval(
            query_embedding, test_retrieval_embeddings, n=args.num_feedback)

        for i in range(args.num_feedback**3):
            feedback = baseb(i,3).zfill(args.num_feedback) # str that simulates feedback for initial set of retrievals
            retrievals[query_id][feedback] = get_nearest_neighbor_retrieval_with_relevance_feedback(
                query_embedding, test_retrieval_embeddings, 
                retrievals[query_id]["feedback_set"], feedback, n=args.num_retrievals)

    # Save results
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    pickle.dump(retrievals, open(args.output_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
