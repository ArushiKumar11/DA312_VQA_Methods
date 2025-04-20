import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPImageProcessor, GPT2Tokenizer

class VQARADDataset(Dataset):
    def __init__(self, csv_file, img_dir, train_setting=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            train_setting (bool): Whether this is for training or testing
        """
        self.vqa_rad_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.preprocess = CLIPImageProcessor.from_pretrained('flaviagiammarino/pubmed-clip-vit-base-patch32')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.train_setting = train_setting

    def __len__(self):
        return len(self.vqa_rad_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.vqa_rad_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = self.preprocess(image, return_tensors="pt")
        questions = self.vqa_rad_frame.iloc[idx, 2]
        answers = self.vqa_rad_frame.iloc[idx, 4]
        tokens, mask, q_len = self.pad_sequences(idx, questions, answers)
        tokens = tokens.long()
        mask = mask.long()
        
        sample = {
            'image': image['pixel_values'].squeeze(0), 
            'tokens': tokens, 
            'mask': mask, 
            'q_len': q_len, 
            'answers': answers, 
            'questions': questions
        }

        return sample

    def pad_sequences(self, index, questions, answers):
        m = [
            torch.tensor(self.gpt2_tokenizer.encode('question: ')),
            torch.tensor(self.gpt2_tokenizer.encode(' context:')),
            torch.tensor(self.gpt2_tokenizer.encode('answer ')),
            torch.tensor(self.gpt2_tokenizer.encode('<|endoftext|>'))
        ]
        
        m_mask = [
            torch.ones(len(self.gpt2_tokenizer.encode('question: '))),
            torch.ones(len(self.gpt2_tokenizer.encode(' context:'))),
            torch.ones(len(self.gpt2_tokenizer.encode('answer '))),
            torch.zeros(len(self.gpt2_tokenizer.encode('<|endoftext|>')))
        ]

        if self.train_setting:
            q = torch.tensor(self.gpt2_tokenizer.encode(str(questions)))
            a = torch.tensor(self.gpt2_tokenizer.encode(str(answers)))
            q, q_mask, leftover_tokens = self.make_padding(16, q, question=True)
            q_len = m[0].size(0) + q.size(0) + m[1].size(0)
            a, a_mask, _ = self.make_padding(6, a, leftover_tokens=leftover_tokens)
            
            if len((a == 0).nonzero()) != 0:
                pad_start = (a == 0).nonzero()[0]
            else:
                pad_start = []
                
            a = torch.cat((a, m[3])) if len(pad_start) == 0 else torch.cat((a[:pad_start], m[3], a[pad_start:]))
            q = torch.cat((m[0], q, m[1], torch.ones(1), m[2], a))
            q_mask = torch.cat((m_mask[0], q_mask, m_mask[1], torch.ones(1), m_mask[2], a_mask, m_mask[3]))

            return q, q_mask, q_len
        else:
            q = torch.tensor(self.gpt2_tokenizer.encode(str(questions)))
            q, q_mask, _ = self.make_padding_test_setting(24, q)
            q_len = m[0].size(0) + q.size(0) + m[1].size(0)
            q = torch.cat((m[0], q, m[1], torch.ones(1), m[2]))
            q_mask = torch.cat((m_mask[0], q_mask, m_mask[1], torch.ones(1), m_mask[2]))
            
            return q, q_mask, q_len

    def make_padding(self, max_len, tokens, question=False, leftover_tokens=0):
        padding = max_len - tokens.size(0)
        
        if padding > 0:
            if question:
                leftover_tokens = padding
                mask = torch.ones(tokens.size(0))
            else:
                tokens = torch.cat((tokens, torch.zeros(padding + leftover_tokens)))
                mask = torch.zeros(max_len + leftover_tokens)
        elif padding == 0:
            if question:
                mask = torch.ones(tokens.size(0))
            else:
                mask = torch.zeros(tokens.size(0) + leftover_tokens)
                tokens = torch.cat((tokens, torch.zeros(leftover_tokens)))
        elif padding < 0:
            if question:
                tokens = tokens[:max_len]
                mask = torch.ones(max_len)
            else:
                tokens = torch.cat((tokens[:max_len], torch.zeros(leftover_tokens)))
                mask = torch.zeros(max_len + leftover_tokens)
                
        return tokens, mask, leftover_tokens

    def make_padding_test_setting(self, max_len, tokens, do_padding=False):
        padding = max_len - tokens.size(0)
        padding_len = 0
        
        if padding > 0:
            if do_padding:
                mask = torch.cat((torch.ones(tokens.size(0)), torch.zeros(padding)))
                tokens = torch.cat((tokens, torch.zeros(padding)))
                padding_len = padding
            else:
                mask = torch.ones(tokens.size(0))
        elif padding == 0:
            mask = torch.ones(max_len)
        elif padding < 0:
            tokens = tokens[:max_len]
            mask = torch.ones(max_len)
            
        return tokens, mask, padding_len


def get_loaders(csv_train='vqa_rad_train.csv', csv_val='vqa_rad_valid.csv', 
                csv_test='vqa_rad_test.csv', img_dir='img', batch_size=32):
    """
    Returns training, validation, and test data loaders.
    Args:
        csv_train (string): Path to the training csv file.
        csv_val (string): Path to the validation csv file.
        csv_test (string): Path to the test csv file.
        img_dir (string): Directory with all the images.
        batch_size (int): Batch size for DataLoader.
    """
    train_dataset = VQARADDataset(csv_file=csv_train, img_dir=img_dir, train_setting=True)
    val_dataset = VQARADDataset(csv_file=csv_val, img_dir=img_dir, train_setting=False)
    test_dataset = VQARADDataset(csv_file=csv_test, img_dir=img_dir, train_setting=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader