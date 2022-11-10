import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

import torch.nn.functional as F


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()
    cls.bn = nn.BatchNorm1d(config.hidden_size, affine=False)


def gather_from_dist(z3):
    if dist.is_initialized():
        z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
        z3_list[dist.get_rank()] = z3
        z3 = torch.cat(z3_list, 0)
    return z3


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


import numpy as np


def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    input_ids0 = input_ids[:,0]
    attention_mask0 = attention_mask[:,0]
    if token_type_ids is not None:
        token_type_ids0 = token_type_ids[:, 0]
    else:
        token_type_ids0 = None

    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = cls.model_args.n_samples

    mlm_outputs = None
    # Flatten input for encoding
    # input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    # attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    # if token_type_ids is not None:
    #     token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)


    input_ids = torch.stack([input_ids0 for _ in range(num_sent)], dim=1).view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = torch.stack([attention_mask0 for _ in range(num_sent)],dim=1).view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = torch.stack([token_type_ids0 for _ in range(num_sent)], dim=1).view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    ori_pooler_output = cls.pooler(attention_mask, outputs)
    cls.ori_pooler_output = ori_pooler_output.view((batch_size, num_sent, ori_pooler_output.size(-1)))

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(cls.ori_pooler_output)

    # Separate representation
    z1 = pooler_output[:,0]
    pos_ratio = cls.model_args.pos_ratio
    sampling_strategy = cls.model_args.sampling_strategy
    SMALL_NUM = np.log(1e-45)
    neg_mask = torch.eye(z1.size(0), device=z1.device)*SMALL_NUM

    # sampling z2
    if sampling_strategy=="mean":
        z2 = pooler_output[:,1]

    elif sampling_strategy=="farest":
        z = pooler_output.mean(dim=1)  # (bs, d)
        distance = (pooler_output[:, 1:] - z.unsqueeze(1)).pow(2).sum(dim=-1).sqrt()  # (bs, n_samples)
        index = distance.argmax(dim=-1)  # bs
        z2 = torch.stack([pooler_output[:, 1:][i, id] for i, id in enumerate(index)], dim=0)

    elif sampling_strategy=="off_dropout":
        z2 = pooler_output[:,1]
        for name, layer in encoder.named_modules():
            if isinstance(layer, nn.Dropout):
                layer.eval()

        outputs0 = encoder(
            input_ids0,
            attention_mask=attention_mask0,
            token_type_ids=token_type_ids0,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

        z0 = cls.pooler(attention_mask0, outputs0)
        if cls.pooler_type == "cls":
            z0 = cls.mlp(z0)
    elif sampling_strategy=="None":
        z2 = pooler_output[:,1]
    else:
        raise NotImplementedError

    # CL learning
    loss = 0
    cos_sim = 0
    if not cls.model_args.sw_only:
        if sampling_strategy=="mean":
            z = (z1 + z2) / 2
            cos_sim = cls.sim(z.unsqueeze(0), z.unsqueeze(1))
            pos_sim = cls.sim(z1, z2)
            pos_loss = -pos_sim
            neg_loss = torch.logsumexp(torch.cat([cos_sim + neg_mask, pos_sim.unsqueeze(-1)], dim=-1), dim=-1)
        elif sampling_strategy=="farest":
            # ## offdropout for neg
            # for name, layer in encoder.named_modules():
            #     if isinstance(layer, nn.Dropout):
            #         layer.eval()
            #
            # outputs0 = encoder(
            #     input_ids0,
            #     attention_mask=attention_mask0,
            #     token_type_ids=token_type_ids0,
            #     position_ids=position_ids,
            #     head_mask=head_mask,
            #     inputs_embeds=inputs_embeds,
            #     output_attentions=output_attentions,
            #     output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            #     return_dict=True,
            # )
            # z0 = cls.pooler(attention_mask0, outputs0)
            # if cls.pooler_type == "cls":
            #     z0 = cls.mlp(z0)
            # pos_sim = pos_ratio * cls.sim(z1, z2)
            # pos_loss = -pos_sim
            # cos_sim = cls.sim(z0.unsqueeze(0), z0.unsqueeze(1))
            # neg_sim = torch.cat([cos_sim + neg_mask, pos_sim.unsqueeze(-1)], dim=-1)
            # neg_loss = torch.logsumexp(neg_sim, dim=-1, keepdim=False)
            #
            # # ## 10 farest for pos, mean for neg
            # # z = (z1 + z2) / 2
            # # cos_sim = cls.sim(z.unsqueeze(0), z.unsqueeze(1))
            # # pos_sim = cls.sim(z1, z2)
            # # pos_loss = -pos_sim
            # # neg_loss = torch.logsumexp(torch.cat([cos_sim + neg_mask, pos_sim.unsqueeze(-1)], dim=-1), dim=-1)

            cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            pos_norm = torch.diag(cos_sim)
            pos_loss = -pos_norm
            neg_loss = torch.logsumexp(torch.cat([cos_sim + neg_mask], dim=-1), dim=-1)
        elif sampling_strategy=="off_dropout":
            pos_sim = pos_ratio * cls.sim(z1, z2)
            pos_loss = -pos_sim
            cos_sim = cls.sim(z0.unsqueeze(0), z0.unsqueeze(1))
            neg_sim = torch.cat([cos_sim + neg_mask, pos_sim.unsqueeze(-1)], dim=-1)
            neg_loss = torch.logsumexp(neg_sim, dim=-1, keepdim=False)
        elif sampling_strategy=="None":
            cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            pos_sim = torch.diag(cos_sim)
            pos_loss = -pos_sim
            neg_loss = torch.logsumexp(cos_sim, dim=-1)
        else:
            raise NotImplementedError

        loss = (pos_loss + neg_loss).mean()

        cls.pos_sim_var = (-pos_loss).detach().var().cpu().numpy()
        cls.neg_sim_var = cos_sim.detach().flatten()[:-1].reshape(batch_size-1, batch_size+1)[:,1:].reshape(batch_size, batch_size-1).var().cpu().numpy()

    # covariance
    cov_mask = SMALL_NUM*torch.eye(z1.size(-1), device=z1.device)
    sw_pos_ratio = cls.model_args.sw_pos_ratio
    sw_sampling_strategy = cls.model_args.sw_sampling_strategy
    if cls.model_args.sw_weight>0:
        temp = cls.model_args.sw_temp
        weight = cls.model_args.sw_weight

        if sw_sampling_strategy=="mean":
            z = cls.bn((z1+z2)/2)
            z1, z2 = cls.bn(z1), cls.bn(z2)
            cov = (z.T @ z) /batch_size/temp
            nu = (z1.T.unsqueeze(1) @ z2.T.unsqueeze(-1)) / batch_size /temp
            de = torch.cat([ cov + cov_mask, nu.squeeze(-1)], dim=-1)
        elif sw_sampling_strategy=="off_dropout":
            z1, z2 = cls.bn(z1), cls.bn(z2)
            nu = (z1.T.unsqueeze(1) @ z2.T.unsqueeze(-1)) / batch_size / temp
            nu = nu*sw_pos_ratio
            cov = (z0.T @ z0) / batch_size / temp
            de = torch.cat([cov+cov_mask, nu.squeeze(-1)], dim=-1)
        elif sw_sampling_strategy=="None":
            z1, z2 = cls.bn(z1), cls.bn(z2)
            cov = z1.T @ z2 / batch_size /temp
            # cov = F.cosine_similarity(z1.T.unsqueeze(0), z2.T.unsqueeze(1), dim=-1) / temp
            nu = torch.diag(cov)
            de = cov
        else:
            raise NotImplementedError
        cov_loss = (-nu+torch.logsumexp(de, dim=-1)).mean()
        loss += weight * cov_loss
        cls.c_pos_sim_var = nu.detach().var().cpu().numpy()
        d = de.size(-1)
        cls.c_neg_sim_var = de.detach().flatten()[:-1].reshape(d-1, d+1)[:,1:].reshape(d, d-1).var().cpu().numpy()

    # for obs
    cls.ori_pooler_output.retain_grad()  # (bs, num_sent, hidden)
    with torch.no_grad():
        d = {}
        for name, layer in encoder.named_modules():
            if isinstance(layer, nn.Dropout):
                layer.train()
                d[name]=layer.p
                layer.p=0.1
        eval_output = encoder(
            torch.stack([input_ids0, input_ids0], dim=1).view(-1, input_ids.size(-1)),
            attention_mask=torch.stack([attention_mask0, attention_mask0], dim=1).view(-1, input_ids.size(-1)),
            token_type_ids=torch.stack([token_type_ids0, token_type_ids0], dim=1).view(-1, input_ids.size(-1)) if token_type_ids is not None else None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
        r = cls.pooler(torch.stack([attention_mask0, attention_mask0], dim=1).view(-1, input_ids.size(-1))
                       , eval_output).view(batch_size, 2, -1)

        cls.align = align_loss(F.normalize(r[:, 0], dim=-1), F.normalize(r[:, 1], dim=-1)).detach().cpu().numpy()

        if cls.model_args.sw_weight > 0:
            cls.c_align = align_loss(r[:, 0].T, r[:, 1].T).detach().cpu().numpy()
        for name, layer in encoder.named_modules():
            if isinstance(layer, nn.Dropout):
                layer.eval()
        eval_output0 = encoder(
            input_ids0,
            attention_mask=attention_mask0,
            token_type_ids=token_type_ids0,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
        r0 = cls.pooler(attention_mask0, eval_output0)

        cls.uni = uniform_loss(F.normalize(r0, dim=-1)).detach().cpu().numpy()
        if cls.model_args.sw_weight > 0:
            cls.c_uni = uniform_loss(r0.T).detach().cpu().numpy()
        for name, layer in encoder.named_modules():
            if isinstance(layer, nn.Dropout):
                layer.train()
                layer.p=d[name]



    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
