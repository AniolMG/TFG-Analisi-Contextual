from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn

# Define a custom class for multi-label sequence classification using BERT
class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels # Number of labels for classification

        # Load the pre-trained BERT model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # Dropout layer to prevent overfitting
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) # Linear layer for classification
        self.loss_fct = nn.BCEWithLogitsLoss() # Loss function for multi-label classification

        self.init_weights() # Initialize the weights

    # Define the forward pass (everything set to none to allow customization)
    def forward(
        self,
        input_ids=None, # Token IDs for the input text
        attention_mask=None, # Mask to avoid attention on padding tokens
        token_type_ids=None, # Segment IDs (for sentence pairs)
        position_ids=None, # Positional IDs 
        head_mask=None, # Mask for attention heads 
        inputs_embeds=None, # Precomputed embeddings
        labels=None, # True labels for the input data
        output_attentions=None, # Whether to return attention weights
        output_hidden_states=None, # Whether to return hidden states
        return_dict=None, # Whether to return a dict or a tuple
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass the inputs through the BERT model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1] # Get the pooled output (representation of [CLS] token)

        pooled_output = self.dropout(pooled_output) # Apply dropout
        logits = self.classifier(pooled_output) # Get the logits (raw scores) for each label

        loss = None
        if labels is not None:
            # Compute the loss if labels are provided
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:] # Return logits and other outputs
            return ((loss,) + output) if loss is not None else output

        # Return a structured output including loss, logits, hidden states, and attentions
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )