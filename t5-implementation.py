import torch
from transformers import T5Tokenizer, T5ForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ESGT5Dataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class T5Analyzer:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForSequenceClassification.from_pretrained('t5-base', num_labels=3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def get_esg_scores(self, texts, batch_size=8):
        dataset = ESGT5Dataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        scores = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                scores.append(batch_scores)

        return np.vstack(scores)
