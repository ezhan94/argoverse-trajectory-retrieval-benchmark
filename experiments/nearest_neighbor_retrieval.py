import argparse
import numpy as np
import os
import pickle
import sys
sys.path.append(sys.path[0] + '/..')

from data import TEST_QUERY_IDS, TEST_RETRIEVAL_IDS
from utils import load_embeddings


"""
Usage:

python experiments/nearest_neighbor_retrieval.py \
--embeddings_file embeddings/pretrain/AE.pkl \
--output_file submissions/AE_retrievals.pkl
"""


def get_nearest_neighbor_retrieval(query_embedding: np.ndarray, retrieval_embeddings: dict, n: int = 1): 
    retrieval_embeddings_np = np.concatenate(list(retrieval_embeddings.values()), axis=0)
    query_embedding_repeat = np.repeat(query_embedding, retrieval_embeddings_np.shape[0], axis=0)

    distance = np.linalg.norm(retrieval_embeddings_np-query_embedding_repeat, axis=1) # Euclidean distance
    sorted_inds = np.argsort(distance) # get indices of increasing distance

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
    args = parser.parse_args()

    # Load embeddings
    embeddings = load_embeddings(args.embeddings_file)

    # Filter for test queries and retrievals
    test_query_embeddings = { traj_id: embeddings[traj_id] for traj_id in TEST_QUERY_IDS }
    test_retrieval_embeddings = { traj_id: embeddings[traj_id] for traj_id in TEST_RETRIEVAL_IDS }

    # Compute nearest neighbor retrievals
    retrievals = {}
    for query_id in TEST_QUERY_IDS:
        query_embedding = test_query_embeddings[query_id]
        retrievals[query_id] = get_nearest_neighbor_retrieval(query_embedding, test_retrieval_embeddings, n=args.num_retrievals)

    # Save results
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    pickle.dump(retrievals, open(args.output_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
