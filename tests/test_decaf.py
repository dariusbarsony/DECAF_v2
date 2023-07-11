from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from DECAF.DECAF import DECAF
from DECAF.data import DataModule
from .utils import gen_data_nonlinear, load_adult, get_metrics


def generate_baseline(size: int = 100) -> Tuple[torch.Tensor, DataModule, list, dict]:
    # causal structure is in dag_seed
    dag_seed = [
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 5],
        [2, 0],
        [3, 0],
        [3, 6],
        [3, 7],
        [6, 9],
        [0, 8],
        [0, 9],
    ]
    # edge removal dictionary
    bias_dict = {6: [3]}  # This removes the edge into 6 from 3.

    # DATA SETUP according to dag_seed
    G = nx.DiGraph(dag_seed)
    data = gen_data_nonlinear(G, SIZE=size)
    dm = DataModule(data.values)

    return torch.Tensor(np.asarray(data)), dm, dag_seed, bias_dict


def test_sanity_params() -> None:
    _, dummy_dm, seed, _ = generate_baseline()

    model = DECAF(
        dummy_dm.dims[0],
        dag_seed=seed,
    )

    assert model.generator is not None
    assert model.discriminator is not None
    assert model.x_dim == dummy_dm.dims[0]
    assert model.z_dim == dummy_dm.dims[0]


def test_sanity_train() -> None:
    _, dummy_dm, seed, _ = generate_baseline()

    model = DECAF(
        dummy_dm.dims[0],
        dag_seed=seed,
    )
    trainer = pl.Trainer(max_epochs=2, logger=False)

    trainer.fit(model, dummy_dm)


def test_sanity_generate() -> None:
    raw_data, dummy_dm, seed, bias_dict = generate_baseline(size=10)

    model = DECAF(
        dummy_dm.dims[0],
        dag_seed=seed,
    )
    trainer = pl.Trainer(max_epochs=2, logger=False)

    trainer.fit(model, dummy_dm)

    synth_data = (
        model.gen_synthetic(
            raw_data, gen_order=model.get_gen_order(), biased_edges=bias_dict
        )
            .detach()
            .numpy()
    )
    assert synth_data.shape[0] == 10


@pytest.mark.parametrize("X,y, df", [load_adult()])
@pytest.mark.slow
def test_run_experiments(X: pd.DataFrame, y: pd.DataFrame, df: pd.DataFrame, mode='ftu') -> None:
    """Normalize X"""
    # X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(X)
    X_normalized = pd.DataFrame(x_scaled)
    baseline_clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                                 learning_rate='constant', learning_rate_init=0.001)
    baseline_clf.fit(X_normalized, y)

    y_pred = baseline_clf.predict(X_normalized)

    print(
        "baseline scores",
        precision_score(y, y_pred),
        recall_score(y, y_pred),
        roc_auc_score(y, y_pred),
    )

    dm = DataModule(X_normalized)

    # causal structure is in dag_seed
    dag_seed = [
        [0, 6],
        [0, 12],
        [0, 1],
        [0, 5],
        [0, 3],
        [3, 6],
        [3, 12],
        [3, 1],
        [3, 7],
        [5, 6],
        [5, 12],
        [5, 1],
        [5, 7],
        [5, 3],
        [8, 6],
        [8, 12],
        [8, 3],
        [8, 5],
        [9, 6],
        [9, 5],
        [9, 12],
        [9, 1],
        [9, 3],
        [9, 7],
        [13, 5],
        [13, 12],
        [13, 3],
        [13, 1],
        [13, 7],
    ]

    # edge removal dictionary
    bias_dict = {14: [9]}  # ftu
    if mode == 'cf':
        bias_dict = {14: [9],
                     14: [5],
                     14: [7]}
    if mode == 'dp':
        bias_dict = {14: [9],
                     14: [5],
                     14: [7],
                     14: [6],
                     14: [12],
                     14: [3],
                     14: [1]}

    model = DECAF(
        dm.dims[0],
        dag_seed=dag_seed,
        use_mask=True,
        grad_dag_loss=False,
        lambda_privacy=0,
        lambda_gp=10,
        weight_decay=1e-2,
        l1_g=0,
        p_gen=-1,
        batch_size=100,
    )

    trainer = pl.Trainer(max_epochs=10, logger=False)

    trainer.fit(model, dm)

    X_synth = (
        model.gen_synthetic(
            dm.dataset.x,
            gen_order=model.get_gen_order(), biased_edges=bias_dict
        )
            .detach()
            .numpy()
    )

    y_synth = baseline_clf.predict(X_synth)

    # synth_clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
    #                           learning_rate='constant', learning_rate_init=0.001)
    # synth_clf.fit(X_synth, y_synth)
    # y_pred = synth_clf.predict(X_synth)
    #
    # print(
    #     "synth scores (Precision, Recall, AUROC):",
    #     precision_score(y, y_pred),
    #     recall_score(y, y_pred),
    #     roc_auc_score(y, y_pred),
    # )

    print('Getting metrics...')

    X_synth_df = pd.DataFrame(X_synth, columns=[
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country"])
    y_synth_df = pd.DataFrame(y_synth, columns=['label'])

    get_metrics(mode, df, X_synth_df, y_synth_df)
