# Argoverse Trajectory Retrieval Benchmark dataset

## Submissions

We will evaluate submissions on test queries and retrievals from the test retrieval set.

### Format

A submission should be a pickle file that contains a `dict` with test query trajectory IDs as keys.

For dict values, we will accept 2 options:

- `list` - In this case we will use the same retrieval set for all test intents. See output of `experiments/nearest_neighbor_retrieval.py`.
- `dict` - This should contain a `feedback_set` key that specifies the initial set of retrievals for which to receive relevance labels (0, 1, or 2). All other keys should be a string corresponding to potential relevance labels. For example, key `"20102"` corresponds to relevance labels of (2, 0, 1, 0, 2) for the (1st, 2nd, 3rd, 4th, 5th) trajectories in feedback set, and the value should be the full retrieval set in response to this feedback. In summary, each query should have a dict that contains `(size of feedback set)^(num label classes) + 1` entries. See output of `experiments/nearest_neighbor_retrieval_relevance_feedback.py`.

Use `check_submission_format.py` to check the format of your submission.

### Submission Link

We will host a server that accepts and evaluates submissions. 

Please check back at a later date (last updated 7/12).
