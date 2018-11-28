#!/usr/bin/python2
import argparse
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

from utils import Corpus, FileLoader, make_batch


def location_of_correct(sim_mat):
    # sort code snippets based similarity for each query
    sorted_ixs = np.argsort(np.negative(sim_mat), axis=1)
    # the known answers are the "diagonal" values
    # ie. query 0 is paired with code 0 etc
    ground_truth = np.arange(0, sim_mat.shape[0]).reshape(-1, 1)
    # True for correct entries
    flag_for_correct = sorted_ixs == ground_truth
    return np.where(flag_for_correct)[1]


def get_mrr(locs):
    # start at index zero so offset by 1
    return np.mean(1.0 / (1 + locs))


def get_fraction_correct_at(locs, cutoff):
    return np.mean(locs < cutoff)


def evaluate(
        model_path,
        corpus_path,
        pairs_path,
        batch_size=100,
):

    model = torch.load(model_path, map_location="cpu")
    model = model.cpu()
    model.eval()

    corpus = Corpus([tuple([corpus_path, os.path.dirname(corpus_path)])])
    pairs_batch_loader = FileLoader([
        tuple([pairs_path, os.path.dirname(pairs_path)])
    ], batch_size)

    code = []
    nl = []

    for data in pairs_batch_loader:
        data = map(corpus.get, data)
        batch = (
            make_batch(model.embedding_layer, data[0][0]),
            make_batch(model.embedding_layer, data[1][0])
        )

        # embed code and NL
        repr_left = model(batch[0])
        repr_right = model(batch[1])
        # accumulate for evaluation
        code.extend(repr_left.numpy())
        nl.extend(repr_right.numpy())

    code = np.array(code)
    nl = np.array(nl)

    sim_mat = cosine_similarity(code, nl)
    ans_locs = location_of_correct(sim_mat)

    summary = {}
    mr = np.mean(ans_locs)
    mrr = get_mrr(ans_locs)
    summary["mrr"] = mrr

    cutoffs = [1, 5, 10]
    fracs = []

    for c in cutoffs:
        frac = get_fraction_correct_at(ans_locs, c)
        fracs.append(frac)
    print("Num obs: {}".format(code.shape[0]))
    print("Mean Rank: {}".format(mr))
    print("MRR: {}".format(mrr))

    for c, f in zip(cutoffs, fracs):
        print("Fraction Correct@{}: {}".format(c, f))
        summary["success@{}".format(c)] = f
    return summary


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model using code search metrics"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path to trained model (must be complete dump, not just weights)",
    )
    parser.add_argument(
        "-c",
        "--corpus",
        type=str,
        help="Path to QRA-style corpus file",
    )
    parser.add_argument(
        "-p",
        "--pairs",
        type=str,
        help="Path to QRA-style file with ids for corresponding code/query",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    evaluate(args.model, args.corpus, args.pairs)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
