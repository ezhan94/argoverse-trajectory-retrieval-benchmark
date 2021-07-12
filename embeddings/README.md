# Argoverse Trajectory Retrieval Benchmark dataset

## Embeddings 

Each file contains embeddings for the validation set of the [Argoverse Motion Forecasting dataset](https://www.argoverse.org/data.html) (39,472 trajectories). 

`pretrain\` contains our initial embeddings pretrained with unsupervised learning.

`finetune\` contains our embeddings after finetuning with triplet loss.

### Format

Each file contains a `dict`, where keys are trajectory IDs as `int`, and values are `np.ndarray` of shape `(1, embeddings_size)`. 

