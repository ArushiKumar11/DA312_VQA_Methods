import torch
import torch.nn as nn
from transformers import CLIPModel, GPT2LMHeadModel
from typing import Tuple

class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(nn.Dropout(p=0.3))
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class VQAModel(nn.Module):
    def __init__(self):
        super(VQAModel, self).__init__()

        self.clip_model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("cemilcelik/distilgpt2_pubmed")
        self.mapper = MLP(sizes=(512, 768, 768))

    def forward(self, images, tokens, mask, q_len):
        image_features = self.clip_model.get_image_features(images)
        image_features = self.mapper(image_features).view(-1, 1, 768)
        embedding = self.gpt2_model.transformer.wte(tokens)

        for b in range(embedding.shape[0]):
            # insert the visual prefix after the question
            embedding[b, q_len[b]:q_len[b]+1, :] = image_features[b]

        return self.gpt2_model(inputs_embeds=embedding, attention_mask=mask).logits

    def generate(self, images, tokens, mask, q_len):
        image_features = self.clip_model.get_image_features(images)
        image_features = self.mapper(image_features).view(-1, 1, 768)
        embedding_txt = self.gpt2_model.transformer.wte(tokens)
        embedding_txt[q_len:q_len+1, :] = image_features
        return embedding_txt