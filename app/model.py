# app/model.py
import torch
import torch.nn as nn

class TrajectoryTransformer(nn.Module):
    def __init__(self, feature_size, num_classes, d_model=128, nhead=8, num_layers=4, window_size=100):
        """
        Initializes the TrajectoryTransformer.

        Args:
            feature_size (int): Number of input features.
            num_classes (int): Number of target classes.
            d_model (int, optional): Embedding dimension. Defaults to 128.
            nhead (int, optional): Number of attention heads. Defaults to 8.
            num_layers (int, optional): Number of transformer encoder layers. Defaults to 4.
            window_size (int, optional): Size of the input window. Defaults to 100.
        """
        super(TrajectoryTransformer, self).__init__()
        self.embedding = nn.Linear(feature_size, d_model)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.position_embedding = nn.Embedding(window_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention_weights_layer = nn.Linear(d_model, 1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, position_ids, src_key_padding_mask=None):
        """
        Forward pass of the transformer.

        Args:
            x (torch.Tensor): Input sequences of shape [batch_size, window_size, feature_size].
            position_ids (torch.Tensor): Position indices of shape [batch_size, window_size].
            src_key_padding_mask (torch.Tensor, optional): Boolean mask of shape [batch_size, window_size].
                                                           True values in the mask will be ignored by the attention.

        Returns:
            torch.Tensor: Logits of shape [batch_size, num_classes].
        """
        # 1) Linear embedding
        x = self.embedding(x)  # [batch_size, window_size, d_model]
        x = self.activation(x)
        x = self.layer_norm(x)

        # 2) Position embedding
        pos_emb = self.position_embedding(position_ids)  # [batch_size, window_size, d_model]
        x = x + pos_emb

        # 3) Transformer expects [sequence_len, batch_size, d_model]
        x = x.transpose(0, 1)  # => [window_size, batch_size, d_model]

        # 4) Forward through transformer encoder (mask shape: [batch_size, window_size])
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        output = output.transpose(0, 1)  # => [batch_size, window_size, d_model]

        # 5) Attention pooling
        attention_scores = self.attention_weights_layer(output).squeeze(-1)  # [batch_size, window_size]
        attention_weights = torch.softmax(attention_scores, dim=-1)          # [batch_size, window_size]

        # Weighted sum
        output = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)  # => [batch_size, d_model]

        # 6) Classification
        logits = self.classifier(output)  # => [batch_size, num_classes]
        return logits
