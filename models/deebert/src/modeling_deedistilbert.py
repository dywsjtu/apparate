import numpy as np
import os
import sys
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.bert.modeling_bert import (
    BERT_INPUTS_DOCSTRING,
    BERT_START_DOCSTRING,
    BertEmbeddings,  # Embeddings
    BertLayer,
    BertPooler,
    BertPreTrainedModel,  # DistilBertPreTrainedModel
)
# DistilBertModel has self.transformers. Is Transformer() a big encoder block?
from transformers.models.distilbert.modeling_distilbert import (
    DISTILBERT_INPUTS_DOCSTRING,
    DISTILBERT_START_DOCSTRING,
    DistilBertModel,
    DistilBertPreTrainedModel,
    Embeddings,
    TransformerBlock,
)

sys.path.insert(1, os.path.join(os.getcwd(), '../../earlyexit'))
sys.path.insert(1, os.path.join(os.getcwd(), './earlyexit'))
from modules import BranchPoint


# DONE
def entropy(x):
    """Calculate entropy of a pre-softmax logit Tensor"""

    # exp_x = torch.exp(x)
    # A = torch.sum(exp_x, dim=1)  # sum of exp(x_i)
    # B = torch.sum(x * exp_x, dim=1)  # sum of x_i * exp(x_i)
    # return torch.log(A) - B / A

    softmax = nn.Softmax(dim=1) 
    out = softmax(x)
    out, _ = torch.max(out, dim=1)
    out = 1 - out
    return out

class DeeDistilBertTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.n_layers = config.n_layers
        self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        # self.layer = None
        # self.set_up_layers(config)
    
    def set_up_layers(self, config):
        layers = []
        for _ in range(config.n_layers):
            layer = TransformerBlock(config)
            ramp = BertHighway(config)
            branchpoint = BranchPoint(layer, ramp)
            layers.append(branchpoint)

        self.layer = nn.ModuleList(layers)

    def init_highway_pooler(self, pooler):
        loaded_model = pooler.state_dict()
        all_highways = []
        for n, p in self.named_modules():
            if n.endswith("branch_net"):
                all_highways.append(p)
        for highway in all_highways:
            for name, param in highway.pooler.state_dict().items():
                """NOTE
                The following line of code is for if we want to create our own pooler for DistilBERT.
                (see https://github.com/huggingface/transformers/issues/15639)
                highway.pooler.state_dict().keys() include ['dense.weight', 'dense.bias'],
                and if we use our custom pooler, the only keys in the state dict are ['weight', 'bias'].
                Nevertheless, we are reverting to using BertPooler, so we are commenting this out.
                """
                # name = name[name.rfind(".")+1:]  # "dense.weight" -> "weight"
                param.copy_(loaded_model[name])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        sample_id=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        all_highway_exits = ()
        all_entropies = ()
        all_logits = ()
        all_predictions = ()

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # TODO(ruipan): make me compatible with earlyexitmodel
            kwargs = {
                "x": hidden_states,
                "attn_mask": attention_mask,
                "head_mask": head_mask[i],
                "output_attentions": self.output_attentions
            }
            layer_outputs = layer_module(**kwargs)
            # layer_outputs = layer_module(
            #     # hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            #     hidden_states, attention_mask, head_mask[i], self.output_attentions
            # )
            
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            current_outputs = (hidden_states,)
            if self.output_hidden_states:
                current_outputs = current_outputs + (all_hidden_states,)
            if self.output_attentions:
                current_outputs = current_outputs + (all_attentions,)

            # # if self.training or \
            # #     ((not self.training) and i in self.active_ramps):
            # highway_exit = self.highway[i](current_outputs)
            # # logits, pooled_output
            # # NOTE: type(layer_module) is TransformerBlock, not BranchPoint (because of references)
            # highway_exit = layer_module.output  

            # if not self.training:
            # if not self.training and highway_exit is not None:  # NOTE(ruipan): accommodate EarlyExitModel
            #     highway_logits = highway_exit[0]
            #     highway_entropy = entropy(highway_logits)
            #     # FIXME(ruipan): batched inference is not yet supported, highway_logits has len=bs
            #     # record each inference request's entropy evolution as it passes through ramps
            #     # all_entropies += (highway_entropy.item(),)
            #     logits_across_ramps = highway_logits.detach().cpu().numpy()
            #     predictions_across_ramps = np.argmax(logits_across_ramps, axis=1)
            #     all_logits += (logits_across_ramps,)
            #     all_predictions += (predictions_across_ramps,)
            #     highway_exit = highway_exit + (highway_entropy,)  # logits, hidden_states(?), entropy
            #     all_highway_exits = all_highway_exits + (highway_exit,)
            # else:
            #     all_highway_exits = all_highway_exits + (highway_exit,)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        if len(all_logits) != 0:
            outputs = outputs + (all_logits,)
        else:
            outputs = outputs + (None,)
        
        if len(all_predictions) != 0:
            outputs = outputs + (all_predictions,)
        else:
            outputs = outputs + (None,)
        if len(all_entropies) != 0:
            outputs = outputs + (all_entropies,)
        else:
            outputs = outputs + (None,)

        outputs = outputs + (all_highway_exits,)
        # outputs = outputs + ([],)  # XXX: temp change for testing purposes only

        # print(f"DeeDistilBertTransformer:len(outputs) {len(outputs)}")
        # NOTE: in the case of DeeDistilBertTransformer, outputs is just hidden_states and all_highway_exits

        return outputs  # last-layer hidden state, (all hidden states), (all attentions), all highway exits

@add_start_docstrings(
    "The Bert Model transformer with early exiting (DeeBERT). ",
    DISTILBERT_START_DOCSTRING,
)
class DeeDistilBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = Embeddings(config)
        # self.transformer = DeeDistilBertTransformer(config)
        self.transformer = DeeDistilBertTransformer(config)  # NOTE(ruipan): changed name for easier profiling
        # self.pooler = nn.Linear(config.dim, config.dim)
        # self.pooler = BertPooler(config)

        self.init_weights()  # DistilBertModel uses self.post_init()

    def init_highway_pooler(self):
        # self.transformer.init_highway_pooler(self.pooler)
        pass  # distilbert doesn't have pooler

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        sample_id=None,
    ):
        """
        Return:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
                Last layer hidden-state of the first token of the sequence (classification token)
                further processed by a Linear layer and a Tanh activation function. The Linear
                layer weights are trained from the next sentence prediction (classification)
                objective during pre-training.

                This output is usually *not* a good summary
                of the semantic content of the input, you're often better with averaging or pooling
                the sequence of hidden-states for the whole input sequence.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
            highway_exits (:obj:`tuple(tuple(torch.Tensor))`:
                Tuple of each early exit's results (total length: number of layers)
                Each tuple is again, a tuple of length 2 - the first entry is logits and the second entry is hidden states.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            # input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
            input_ids=input_ids
        )
        # encoder_outputs = self.transformer(
        encoder_outputs = self.transformer(  # NOTE(ruipan): changed name for consistency across models in profiling
            embedding_output,
            # attention_mask=extended_attention_mask,
            attention_mask=attention_mask,  # NOTE: effort to make deedistilbert work
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            sample_id=sample_id,
        )
        # NOTE: encoder_outputs is hidden_states and all_highway_exits
        sequence_output = encoder_outputs[0]

        # pooled_output = self.pooler(sequence_output)
        # outputs = (sequence_output, pooled_output,) + encoder_outputs[
        # NOTE: pooled_output is generated in DeeDistilBertForSequenceClassification
        # this is because DistilBERT does not have a pooled layer
        outputs = (sequence_output, None,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions), highway exits

class HighwayException(Exception):
    def __init__(self, message, exit_layer):
        self.message = message
        self.exit_layer = exit_layer  # start from 1!

class BertHighway(nn.Module):
    """A module to provide a shortcut
    from (the output of one non-final BertLayer in BertEncoder) to (cross-entropy computation in BertForSequenceClassification)
    """
    def __init__(self, config):
        super().__init__()
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.classifier = nn.Linear(config.dim, config.num_labels)

    def forward(self, encoder_outputs):
        # Pooler
        pooler_input = encoder_outputs[0]
        pooler_output = self.pooler(pooler_input)
        # "return" pooler_output

        # BertModel
        bmodel_output = (pooler_input, pooler_output) + encoder_outputs[1:]
        # "return" bmodel_output

        # Dropout and classification
        pooled_output = bmodel_output[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, pooled_output


# refer to class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
@add_start_docstrings(
    """Bert Model (with early exiting - DeeBERT) with a classifier on top,
    also takes care of multi-layer training. """,
    DISTILBERT_START_DOCSTRING,
)
class DeeDistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers
        self.config = config

        self.distilbert = DeeDistilBertModel(config)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        # bertforsequence... uses config.hidden_size, distilbertforsequence... uses config.dim
        # not sure if these are the same
        self.classifier = nn.Linear(config.dim, self.config.num_labels)
        self.pre_classifier = nn.Linear(config.dim, config.dim)

        self.init_weights()  # DistilBertForSequence... uses self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_layer=-1,
        ramp_ids=None,
        sample_id=None,
    ):
        """
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
                Classification (or regression if config.num_labels==1) loss.
            logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
            highway_exits (:obj:`tuple(tuple(torch.Tensor))`:
                Tuple of each early exit's results (total length: number of layers)
                Each tuple is again, a tuple of length 2 - the first entry is logits and the second entry is hidden states.
        """

        exit_layer = self.num_layers
        try:
            outputs = self.distilbert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                sample_id=sample_id,
            )

            # sequence_output, pooled_output, (hidden_states), (attentions), highway exits
            pooled_output = outputs[1]

            #########################################
            # reference: https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/distilbert/modeling_distilbert.py#L770-L775
            assert pooled_output is None
            sequence_output = outputs[0]  # sequence_output.shape: torch.Size([1, 128, 768]) (bs, seq_len, dim)
            pooled_output = sequence_output[:, 0]  # (bs, dim)
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)  # FIXME
            pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
            pooled_output = self.dropout(pooled_output)  # (bs, dim)
            #########################################

            # pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        except HighwayException as e:
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]

        if not self.training:
            original_entropy = entropy(logits)
            highway_entropy = []
            highway_logits_all = []
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # work with highway exits
            highway_losses = []
            for highway_exit in outputs[-1]:  # FIXME: change this logic so that outputs[-1] (None) is never iterated on
                if highway_exit:  # NOTE(ruipan): accommodate EarlyExitModel
                    highway_logits = highway_exit[0]
                    if not self.training:
                        highway_logits_all.append(highway_logits)
                        highway_entropy.append(highway_exit[2])
                    if self.num_labels == 1:
                        #  We are doing regression
                        loss_fct = MSELoss()
                        highway_loss = loss_fct(highway_logits.view(-1), labels.view(-1))
                    else:
                        loss_fct = CrossEntropyLoss()
                        highway_loss = loss_fct(highway_logits.view(-1, self.num_labels), labels.view(-1))
                    highway_losses.append(highway_loss)
            
            if ramp_ids is None:  # backbone model training
                outputs = (loss,) + outputs
            else:  # ramp training
                # outputs = (sum(highway_losses[:-1]),) + outputs
                # exclude the final highway, of course. NOTE: this is DeeBERT's design choice
                highway_loss_sum = 0
                for ramp_id, loss in enumerate(highway_losses):
                    if ramp_id in ramp_ids and ramp_id != len(highway_losses) - 1:
                        highway_loss_sum += loss
                outputs = (highway_loss_sum,) + outputs
        if not self.training:
            outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
            if output_layer >= 0:
                outputs = (
                    (outputs[0],) + (highway_logits_all[output_layer],) + outputs[2:]
                )  # use the highway of the last layer

        return outputs  # (loss), logits, (hidden_states), (attentions), (highway_exits)
