from torch import nn

from transformers.models.bert.modeling_bert import BertPooler


class BertHighway(nn.Module):
    """A module to provide a shortcut
    from (the output of one non-final BertLayer in BertEncoder) to (cross-entropy computation in BertForSequenceClassification)
    """

    def __init__(self, config):
        super().__init__()

        if hasattr(config, "hidden_dropout_prob"):  # Bert
            dropout_prob = config.hidden_dropout_prob
        elif hasattr(config, "seq_classif_dropout"):  # DistilBert
            dropout_prob = config.seq_classif_dropout
        else:
            raise NotImplementedError
        if hasattr(config, "hidden_size"):  # Bert
            in_features = config.hidden_size
        elif hasattr(config, "dim"):  # DistilBert
            in_features = config.dim
        else:
            raise NotImplementedError
        if hasattr(config, "num_labels"):
            out_features = config.num_labels
        else:
            raise NotImplementedError

        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(in_features, out_features)

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
