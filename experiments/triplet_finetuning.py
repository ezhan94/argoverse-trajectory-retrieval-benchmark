import argparse
import numpy as np
import os
import pickle
import random
import sys
sys.path.append(sys.path[0] + '/..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from data import LABELS
from utils import load_embeddings


"""
Usage:

python experiments/triplet_finetuning.py \
--embeddings_file embeddings/pretrain/AE.pkl \
--output_file tmp/AE_triplet.pkl
"""


class EmbeddingDataset(Dataset):

    def __init__(self, emb_dict: dict, mode: str, train_test_split: float = 0.8):
        emb_np = np.concatenate(list(emb_dict.values()), axis=0)
        split = int(train_test_split*emb_np.shape[0])
        self.data = emb_np[:split] if mode == "train" else emb_np[split:]

        self.x_dim = self.data.shape[1]
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class TripletDataset(Dataset):

    def __init__(self, emb_dict: dict, labels: dict):
        self.emb_dict = emb_dict
        self.labels = labels
        self.trajs_with_labels = list(labels.keys())
        self.sample_triplets()

    def sample_triplets(self):
        anchors = []
        positives = []
        negatives = []

        for traj_id, label in self.labels.items():
            for i, intent_label in enumerate(label):
                # Construct triplet if intent label exists
                if intent_label == 2:
                    anchors.append(self.emb_dict[traj_id])

                    # Sample positive example (also has same intent label)
                    pos_id = random.choice(self.trajs_with_labels)
                    while pos_id == traj_id or self.labels[pos_id][i] != 2:
                        pos_id = random.choice(self.trajs_with_labels)
                    positives.append(self.emb_dict[pos_id])

                    # Sample negative example (does not have the intent)
                    neg_id = random.choice(self.trajs_with_labels)
                    while neg_id == traj_id or self.labels[neg_id][i] != 0:
                        neg_id = random.choice(self.trajs_with_labels)
                    negatives.append(self.emb_dict[neg_id])

        self.anchors = np.concatenate(anchors, axis=0)
        self.positives = np.concatenate(positives, axis=0)
        self.negatives = np.concatenate(negatives, axis=0)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        return (self.anchors[index], self.positives[index], self.negatives[index])


class Autoencoder(nn.Module):

    def __init__(self, input_dim: int, h_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim))

        self.decoder = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_dim))

    def encode(self, batch):
        return self.encoder(batch)

    def decode(self, batch):
        return self.decoder(batch)

    def forward(self, batch):
        latent = self.encode(batch)
        recon = self.decode(latent)
        return batch.size(0)*F.mse_loss(recon, batch, reduction='mean')

    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        anchor_emb = self.encode(anchor)
        pos_emb = self.encode(positive)
        neg_emb = self.encode(negative)

        triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        return anchor.size(0)*triplet_loss(anchor_emb, pos_emb, neg_emb)


def run_epoch(dataloader, optimizer=None):
    total_loss = 0.0 

    for batch_idx, batch in enumerate(dataloader):
        batch_loss = model(batch)

        if optimizer:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_loss += batch_loss.item()

    return total_loss / len(dataloader.dataset)


def run_epoch_triplet(dataloader, optimizer=None):
    total_loss = 0.0 

    for batch_idx, batch in enumerate(dataloader):
        anchor, positive, negative = batch
        batch_loss = model.triplet_loss(anchor, positive, negative)

        if optimizer:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_loss += batch_loss.item()

    return total_loss / len(dataloader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_file', type=str,
                        required=True, default='',
                        help='embeddings pickle file')
    parser.add_argument('--output_file', type=str,
                        required=True, default='',
                        help='file to save retrievals')
    parser.add_argument('-b', '--batch_size', type=int,
                        required=False, default=32,
                        help='batch size for training')
    parser.add_argument('-lr', '--learning_rate', type=float,
                        required=False, default=0.001,
                        help='learning rate for training')
    parser.add_argument('-e', '--num_epochs', type=int,
                        required=False, default=30,
                        help='number of epochs for training')
    args = parser.parse_args()

    # Load embeddings
    embeddings = load_embeddings(args.embeddings_file)

    # Initialize datasets
    train_dataset = EmbeddingDataset(embeddings, mode="train")
    test_dataset = EmbeddingDataset(embeddings, mode="test")
    triplet_dataset = TripletDataset(embeddings, LABELS)

    # Initialize dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    triplet_dataloader = DataLoader(triplet_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model and optimizer
    model = Autoencoder(input_dim=train_dataset.x_dim, h_dim=train_dataset.x_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Finetune embeddings
    best_test = float('inf')
    for epoch in range(args.num_epochs):
        train_loss = run_epoch(train_dataloader, optimizer)

        triplet_dataloader.dataset.sample_triplets() # re-sample triplets
        triplet_loss = run_epoch_triplet(triplet_dataloader, optimizer)

        test_loss = run_epoch(test_dataloader)

        print(f"Epoch {epoch+1:2d}\tTrain {train_loss:.4f}\t Test {test_loss:.4f}\tTriplet {triplet_loss:.4f}")

        if test_loss < best_test:
            best_test = test_loss
        else:
            break

    # Compute new embeddings
    new_embeddings = {}
    for traj_id in embeddings.keys():
        new_embeddings[traj_id] = model.encode(torch.tensor(embeddings[traj_id])).detach().numpy()

    # Save embeddings
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    pickle.dump(new_embeddings, open(args.output_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
