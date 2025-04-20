import torch
import torch.nn as nn
from transformers import AutoModel


class MultimodalVQAModel(nn.Module):
    def __init__(
            self,
            num_labels: int,
            intermediate_dim: int = 512,
            pretrained_text_name: str = 'bert-base-uncased',
            pretrained_image_name: str = 'google/vit-base-patch16-224-in21k'):

        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_text_name = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name

        self.text_encoder = AutoModel.from_pretrained(
            self.pretrained_text_name,
        )
        self.image_encoder = AutoModel.from_pretrained(
            self.pretrained_image_name,
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Linear(intermediate_dim, self.num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            attention_mask: torch.LongTensor = None,
            token_type_ids: torch.LongTensor = None,
            labels: torch.LongTensor = None):

        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
        fused_output = self.fusion(
            torch.cat(
                [
                    encoded_text['pooler_output'],
                    encoded_image['pooler_output'],
                ],
                dim=1
            )
        )
        logits = self.classifier(fused_output)

        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss

        return out

    @staticmethod
    def load_from_checkpoint(checkpoint_path, num_labels, map_location="cpu"):
        """Load a model from checkpoint."""
        model = MultimodalVQAModel(num_labels=num_labels)
        
        # Try loading with safetensors first
        try:
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
            model.load_state_dict(state_dict)
        except (ImportError, RuntimeError):
            # Fall back to regular PyTorch loading
            model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
            
        return model