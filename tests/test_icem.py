import unittest

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

import torch_explain as te
from torch_explain import datasets
from torch_explain.nn.concepts import ConceptLinearLayer
from torch_explain.nn.semantics import GodelTNorm, ProductTNorm


def train_concept_bottleneck_model(x, c, y, embedding_size=10, concept_names=None):
    x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(x, c, y, test_size=0.33, random_state=42)

    y_train = F.one_hot(y_train.long().ravel()).float()
    y_test = F.one_hot(y_test.long().ravel()).float()

    # y_train = 1 - y_train
    # y_test = 1 - y_test

    # concept embedding model
    encoder = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 10),
        torch.nn.LeakyReLU(),
    )
    concept_embedder = te.nn.ConceptEmbedding(10, c.shape[1], embedding_size)
    task_predictor = ConceptLinearLayer(embedding_size, y_train.shape[1])
    model = torch.nn.Sequential(encoder, concept_embedder, task_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form_c = torch.nn.BCEWithLogitsLoss()
    loss_form_y = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(501):
        optimizer.zero_grad()

        h = encoder(x_train)
        c_emb, c_pred = concept_embedder.forward(h, [0, 1], c_train, train=True)
        y_pred, weight_attn, bias_attn = task_predictor(c_emb, c_pred, return_attn=True)

        concept_loss = loss_form_c(c_pred, c_train)
        task_loss = loss_form_y(y_pred, y_train)
        weight_attn_reg = task_predictor.entropy_reg(weight_attn)
        loss = concept_loss + 0.5*task_loss + 1e-4*weight_attn_reg + 1e-4*torch.mean(bias_attn.abs()**2)
        loss.backward()
        optimizer.step()

        # compute accuracy
        if epoch % 100 == 0:
            h = encoder(x_test)
            c_emb, c_pred = concept_embedder.forward(h)
            y_pred = task_predictor(c_emb, c_pred)

            task_accuracy = accuracy_score(y_test, y_pred > 0.5)
            concept_accuracy = accuracy_score(c_test, c_pred > 0.5)
            print(f'Epoch {epoch}: loss {loss:.4f} task accuracy: {task_accuracy:.4f} '
                  f'concept accuracy: {concept_accuracy:.4f} bias loss: {torch.mean(bias_attn.abs()):.4f} ')

    # local_explanations = task_predictor.explain(c_emb, c_pred, 'local', concept_names)
    global_explanations = task_predictor.explain(c_emb, c_pred, 'global', concept_names)
    print(global_explanations)

    return model


if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)

    x, c, y = datasets.xor(1000)
    train_concept_bottleneck_model(x, c, y, embedding_size=32, concept_names=['x1', 'x2'])

    x, c, y = datasets.trigonometry(1000)
    train_concept_bottleneck_model(x, c, y, embedding_size=16, concept_names=['x > 0', 'y > 0', 'z > 0'])

    x, c, y = datasets.dot(1000)
    train_concept_bottleneck_model(x, c, y, embedding_size=16, concept_names=['v1.v2', 'v3.v4'])


# class TestTemplateObject(unittest.TestCase):
#     def test_deep_core(self):
#         x, c, y = datasets.xor(1000)
#         train_concept_bottleneck_model(x, c, y, embedding_size=16)
#
#         x, c, y = datasets.trigonometry(1000)
#         train_concept_bottleneck_model(x, c, y, embedding_size=16)
#         #
#         x, c, y = datasets.dot(1000)
#         train_concept_bottleneck_model(x, c, y, embedding_size=16)
#
#         return
#
#
# if __name__ == '__main__':
#     unittest.main()
