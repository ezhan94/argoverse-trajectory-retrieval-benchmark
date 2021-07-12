import argparse
import os
import pickle
import sys
sys.path.append(sys.path[0] + '/..')


"""
Usage:

python submissions/check_submission_format.py --submission_file submissions/AE_retrievals.pkl
"""


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission_file', type=str,
                        required=True, default='',
                        help='embeddings pickle file')
    args = parser.parse_args()

    assert os.path.isfile(args.submission_file), f"{args.submission_file} does not exist"
    
    submission = pickle.load(open(args.submission_file, 'rb'))

    assert isinstance(submission, dict), f"{args.submission_file} type, expecting dict but found {type(submission)}"

    for traj_id, retrievals in submission.items():
        assert isinstance(traj_id, int), f"dict keys type, expecting int but found {type(traj_id)}"

        # single retrieval
        if isinstance(retrievals, list):
            if len(retrievals) > 0:
                assert isinstance(retrievals[0], int), f"traj_id type, expecting int but found {type(retrievals[0])}" 
        # retrieval with feedback
        elif isinstance(retrievals, dict):
            assert "feedback_set" in retrievals, f"missing key feedback_set"
            for feedback, feedback_retrievals in retrievals.items():
                assert isinstance(feedback, str), f"feedback dict keys type, expecting str but found {type(feedback)}"
                assert isinstance(feedback_retrievals, list), f"feedback dict values type, expecting list but found {type(feedback_retrievals)}"
                if len(feedback_retrievals) > 0:
                    assert isinstance(feedback_retrievals[0], int), f"traj_id type, expecting int but found {type(feedback_retrievals[0])}"
        else:
            raise AssertionError(f"dict values type, expecting list/dict but found {type(retrievals)}")
