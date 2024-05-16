import torch
from collections import Counter

from .semantics import Logic, GodelTNorm


def softselect(values, temperature):
    softmax_scores = torch.log_softmax(values, dim=1)
    softscores = torch.sigmoid(softmax_scores - temperature * softmax_scores.mean(dim=1, keepdim=True))
    return softscores


class ConceptLinearLayer(torch.nn.Module):
    """
    This layer implements a linear layer working over concept embedding and outputting the task prediction.
    Similarly to the ConceptReasoningLayer, it also makes an interpretable prediction. This time, however, the
    prediction is a linear combination of the concepts, where the weights are predicted for each sample by the layer.
    """
    def __init__(self, emb_size, n_classes, bias=True, attention=False):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.weight_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, 3 * n_classes),
            torch.nn.Sigmoid(),
        )
        self.pos_weight_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
            torch.nn.Sigmoid(),
        )
        self.neg_weight_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
            torch.nn.Sigmoid(),
        )
        self.bias = bias
        if self.bias:
            self.pos_bias_nn = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size),
                                               torch.nn.LeakyReLU(),
                                               torch.nn.Linear(emb_size, n_classes),
                                               torch.nn.Sigmoid()
                                               )
            self.neg_bias_nn = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.Linear(emb_size, n_classes),
                                                  torch.nn.Sigmoid()
                                                  )

        self.attention = attention
        if attention:
            self.attention_nn = torch.nn.MultiheadAttention(emb_size, 1, batch_first=True)

    def forward(self, x, c, return_attn=False, weight_attn=None):

        if self.attention:
            x, _ = self.attention_nn(x, x, x)

        if weight_attn is None:

            weight_attn = self.pos_weight_nn(x) - self.neg_weight_nn(x)

            # weight_attn = self.weight_nn(x).reshape(-1, c.shape[1], self.n_classes, 3)
            # p, n, z = weight_attn[:, :, :, 0], weight_attn[:, :, :, 1], weight_attn[:, :, :, 2]
            # weight_attn = (p - n) * (1-z)

        # sum of w_i*c_i + b for all concepts c_i
        logits = (c.unsqueeze(-1) * weight_attn).sum(dim=1).float()
        if self.bias:
            bias_attn = self.pos_bias_nn(x.mean(dim=1)) - self.neg_bias_nn(x.mean(dim=1))

            logits += bias_attn
            # weight_attn = torch.cat([weight_attn, bias_attn.unsqueeze(1)], dim=1)
        # preds = torch.sigmoid(logits)
        preds = logits
        if return_attn:
            if self.bias:
                return preds, weight_attn, bias_attn
            return preds, weight_attn
        else:
            return preds

    def explain(self, x, c, mode, concept_names=None, class_names=None, weight_attn=None):
        assert mode in ['local', 'global', 'exact']

        if concept_names is None:
            concept_names = [f'c_{i}' for i in range(c.shape[1])]
        if class_names is None:
            class_names = [f'y_{i}' for i in range(self.n_classes)]

        # make a forward pass to get predictions and attention weights
        if self.bias:
            y_preds, weight_attn_mask, _ = self.forward(x, c, return_attn=True, weight_attn=weight_attn)
        else:
            y_preds, weight_attn_mask = self.forward(x, c, return_attn=True, weight_attn=weight_attn)

        explanations = []
        all_class_explanations = {cn: [] for cn in class_names}
        for sample_idx in range(len(x)):
            prediction = y_preds[sample_idx] > 0.5
            active_classes = torch.argwhere(prediction).ravel()

            if len(active_classes) == 0:
                # if no class is active for this sample, then we cannot extract any explanation
                explanations.append({
                    'class': -1,
                    'explanation': '',
                    'attention': [],
                })
            else:
                # else we can extract an explanation for each active class!
                for target_class in active_classes:
                    attentions = []
                    minterm = []
                    for concept_idx in range(len(concept_names)):
                        c_pred = c[sample_idx, concept_idx]
                        weight_attn = weight_attn_mask[sample_idx, concept_idx, target_class]

                        # we first check if the concept was relevant
                        # a concept is relevant <-> the absolute value of the weight attention score is higher than 0.5 and the concept is active
                        if torch.abs(c_pred) > 0.5:
                            if weight_attn > 0:
                                minterm.append(f'+{weight_attn:.0f} {concept_names[concept_idx]}')
                            else:
                                minterm.append(f'{weight_attn:.0f} {concept_names[concept_idx]}')
                        # minterm.append(f'({concept_names[concept_idx]})')
                        attentions.append(weight_attn.item())

                    # then we add the bias value (last value of the attention weights)
                    # if self.bias:
                    #     bias_attn = weight_attn_mask[sample_idx, -1, target_class]
                    #     if bias_attn > 0:
                    #         minterm.append(f'+{bias_attn:.1f} bias')
                    #     else:
                    #         minterm.append(f'{bias_attn:.1f} bias')
                    #     attentions.append(bias_attn.item())

                    # add explanation to list
                    target_class_name = class_names[target_class]
                    minterm = ' '.join(minterm)
                    all_class_explanations[target_class_name].append(minterm)
                    explanations.append({
                        'sample-id': sample_idx,
                        'class': target_class_name,
                        'explanation': minterm,
                        'attention': attentions,
                    })

        if mode == 'global':
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.most_common():
                    if count > 5:
                        explanations.append({
                            'class': class_id,
                            'explanation': explanation,
                            'count': count,
                        })

        return explanations

    @staticmethod
    def entropy_reg(t: torch.Tensor):
        abs_t = torch.abs(t) + 1e-10
        entropy = abs_t * torch.log(abs_t)
        entropy_sum = - torch.sum(entropy, dim=1)
        if entropy_sum.isnan().any():
            print(t)
            raise ValueError
        return entropy_sum.mean()


class ConceptReasoningLayer(torch.nn.Module):
    def __init__(self, emb_size, n_classes, logic: Logic = GodelTNorm(), temperature: float = 100.):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.logic = logic
        self.filter_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
        )
        self.sign_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
        )
        self.temperature = temperature

    def forward(self, x, c, return_attn=False, sign_attn=None, filter_attn=None):
        values = c.unsqueeze(-1).repeat(1, 1, self.n_classes)

        if sign_attn is None:
            # compute attention scores to build logic sentence
            # each attention score will represent whether the concept should be active or not in the logic sentence
            sign_attn = torch.sigmoid(self.sign_nn(x))

        # attention scores need to be aligned with predicted concept truth values (attn <-> values)
        # (not A or V) and (A or not V) <-> (A <-> V)
        sign_terms = self.logic.iff_pair(sign_attn, values)

        if filter_attn is None:
            # compute attention scores to identify only relevant concepts for each class
            filter_attn = softselect(self.filter_nn(x), self.temperature)

        # filter value
        # filtered implemented as "or(a, not b)", corresponding to "b -> a"
        filtered_values = self.logic.disj_pair(sign_terms, self.logic.neg(filter_attn))

        # generate minterm
        preds = self.logic.conj(filtered_values, dim=1).squeeze(1).float()

        if return_attn:
            return preds, sign_attn, filter_attn
        else:
            return preds

    def explain(self, x, c, mode, concept_names=None, class_names=None, filter_attn=None):
        assert mode in ['local', 'global', 'exact']

        if concept_names is None:
            concept_names = [f'c_{i}' for i in range(c.shape[1])]
        if class_names is None:
            class_names = [f'y_{i}' for i in range(self.n_classes)]

        # make a forward pass to get predictions and attention weights
        y_preds, sign_attn_mask, filter_attn_mask = self.forward(x, c, return_attn=True, filter_attn=filter_attn)

        explanations = []
        all_class_explanations = {cn: [] for cn in class_names}
        for sample_idx in range(len(x)):
            prediction = y_preds[sample_idx] > 0.5
            active_classes = torch.argwhere(prediction).ravel()

            if len(active_classes) == 0:
                # if no class is active for this sample, then we cannot extract any explanation
                explanations.append({
                    'class': -1,
                    'explanation': '',
                    'attention': [],
                })
            else:
                # else we can extract an explanation for each active class!
                for target_class in active_classes:
                    attentions = []
                    minterm = []
                    for concept_idx in range(len(concept_names)):
                        c_pred = c[sample_idx, concept_idx]
                        sign_attn = sign_attn_mask[sample_idx, concept_idx, target_class]
                        filter_attn = filter_attn_mask[sample_idx, concept_idx, target_class]

                        # we first check if the concept was relevant
                        # a concept is relevant <-> the filter attention score is lower than the concept probability
                        at_score = 0
                        sign_terms = self.logic.iff_pair(sign_attn, c_pred).item()
                        if self.logic.neg(filter_attn) < sign_terms:
                            if sign_attn >= 0.5:
                                # if the concept is relevant and the sign is positive we just take its attention score
                                at_score = filter_attn.item()
                                if mode == 'exact':
                                    minterm.append(f'{sign_terms:.3f} ({concept_names[concept_idx]})')
                                else:
                                    minterm.append(f'{concept_names[concept_idx]}')
                            else:
                                # if the concept is relevant and the sign is positive we take (-1) * its attention score
                                at_score = -filter_attn.item()
                                if mode == 'exact':
                                    minterm.append(f'{sign_terms:.3f} (~{concept_names[concept_idx]})')
                                else:
                                    minterm.append(f'~{concept_names[concept_idx]}')
                        attentions.append(at_score)

                    # add explanation to list
                    target_class_name = class_names[target_class]
                    minterm = ' & '.join(minterm)
                    all_class_explanations[target_class_name].append(minterm)
                    explanations.append({
                        'sample-id': sample_idx,
                        'class': target_class_name,
                        'explanation': minterm,
                        'attention': attentions,
                    })

        if mode == 'global':
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.items():
                    explanations.append({
                        'class': class_id,
                        'explanation': explanation,
                        'count': count,
                    })

        return explanations


class ConceptEmbedding(torch.nn.Module):
    def __init__(
            self,
            in_features,
            n_concepts,
            emb_size,
            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_idxs=None,
            training_intervention_prob=0.25,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.intervention_idxs = intervention_idxs
        self.training_intervention_prob = training_intervention_prob
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        self.concept_context_generators = torch.nn.ModuleList()
        for i in range(n_concepts):
            self.concept_context_generators.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
        self.concept_prob_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, 1),
            torch.nn.Sigmoid(),
        )

        # And default values for interventions here
        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.zeros(n_concepts)

    def _after_interventions(
            self,
            prob,
            concept_idx,
            intervention_idxs=None,
            c_true=None,
            train=False,
    ):
        if train and (self.training_intervention_prob != 0) and (intervention_idxs is None):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(self.ones * self.training_intervention_prob)
            intervention_idxs = torch.nonzero(mask).reshape(-1)
        if (c_true is None) or (intervention_idxs is None):
            return prob
        if concept_idx not in intervention_idxs:
            return prob
        return (c_true[:, concept_idx:concept_idx + 1] * self.active_intervention_values[concept_idx]) + \
            ((c_true[:, concept_idx:concept_idx + 1] - 1) * -self.inactive_intervention_values[concept_idx])

    def forward(self, x, intervention_idxs=None, c=None, train=False):
        c_emb_list, c_pred_list = [], []
        # We give precendence to inference time interventions arguments
        used_int_idxs = intervention_idxs
        if used_int_idxs is None:
            used_int_idxs = self.intervention_idxs
        for i, context_gen in enumerate(self.concept_context_generators):
            context = context_gen(x)
            c_pred = self.concept_prob_predictor(context)
            c_pred_list.append(c_pred)
            # Time to check for interventions
            c_pred = self._after_interventions(
                prob=c_pred,
                concept_idx=i,
                intervention_idxs=used_int_idxs,
                c_true=c,
                train=train,
            )

            context_pos = context[:, :self.emb_size]
            context_neg = context[:, self.emb_size:]
            c_emb = context_pos * c_pred + context_neg * (1 - c_pred)
            c_emb_list.append(c_emb.unsqueeze(1))

        return torch.cat(c_emb_list, axis=1), torch.cat(c_pred_list, axis=1)
