import os
from dataclasses import dataclass
from typing import List, Dict
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoFeatureExtractor
from datasets import load_dataset


@dataclass
class MultimodalCollator:
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor
    images_dir: str = "data/images"
    max_length: int = 24

    def tokenize_text(self, texts: List[str]):
        encoded_text = self.tokenizer(
            text=texts,
            padding='longest',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text['input_ids'].squeeze(),
            "token_type_ids": encoded_text['token_type_ids'].squeeze(),
            "attention_mask": encoded_text['attention_mask'].squeeze(),
        }

    def preprocess_images(self, images: List[str]):
        processed_images = self.preprocessor(
            images=[Image.open(os.path.join(self.images_dir, image_id + ".png")).convert('RGB') 
                   for image_id in images],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }

    def __call__(self, raw_batch_dict):
        if isinstance(raw_batch_dict, dict):
            questions = raw_batch_dict['question']
            image_ids = raw_batch_dict['image_id']
            labels = raw_batch_dict.get('label', None)
        else:
            questions = [i['question'] for i in raw_batch_dict]
            image_ids = [i['image_id'] for i in raw_batch_dict]
            labels = [i.get('label', None) for i in raw_batch_dict] if 'label' in raw_batch_dict[0] else None
        
        batch = {
            **self.tokenize_text(questions),
            **self.preprocess_images(image_ids),
        }
        
        if labels is not None and not all(l is None for l in labels):
            batch['labels'] = torch.tensor(labels, dtype=torch.int64)
            
        return batch


def load_vqa_dataset(
    train_path: str = "data/data_train.csv",
    test_path: str = "data/data_eval.csv",
    answer_space_path: str = "data/answer_space.txt"
):
    """Load the VQA dataset from CSV files."""
    dataset = load_dataset(
        "csv",
        data_files={
            "train": train_path,
            "test": test_path
        }
    )
    
    # Load answer space vocabulary
    with open(answer_space_path) as f:
        answer_space = f.read().splitlines()
    
    # Map answers to labels
    dataset = dataset.map(
        lambda examples: {
            'label': [
                answer_space.index(ans.replace(" ", "").split(",")[0])  # Select the 1st answer if multiple answers are provided
                for ans in examples['answer']
            ]
        },
        batched=True
    )
    
    return dataset, answer_space