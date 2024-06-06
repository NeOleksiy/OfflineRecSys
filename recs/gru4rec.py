import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.functional import softmax
from tqdm import tqdm
from recs.recommend_base import BaseRecommender
import os
from typing import List


class ItemsDataset:
    def __init__(self, data, item2ind, device='cpu'):
        self.device = device
        self.data = data
        self.item2ind = item2ind
        self.unk_id = item2ind['<unk>']
        self.pad_id = item2ind['<pad>']

    def __getitem__(self, idx: int) -> List[int]:
        train_sample = [self.item2ind.get(item, self.unk_id) for item in self.data.iloc[idx]]

        return train_sample

    def __len__(self) -> int:
        return len(self.data)

    def collate_fn_with_padding(self,
                                input_batch: List[List[int]]) -> torch.Tensor:
        sessions_len = [len(x) for x in input_batch]
        max_session_len = max(sessions_len)

        new_batch = []
        for session in input_batch:
            for _ in range(max_session_len - len(session)):
                session.append(self.pad_id)
            new_batch.append(session)

        sessions = torch.LongTensor(new_batch).to(self.device)

        new_batch = {
            'input_ids': sessions[:, :-1],
            'target_ids': sessions[:, 1:]
        }

        return new_batch


class GRU(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, task):
        super().__init__()
        self.task = task
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, vocab_size)

        self.non_lin = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_batch) -> torch.Tensor:
        embeddings = self.embedding(input_batch)  # [batch_size, seq_len, hidden_dim]
        output, _ = self.rnn(embeddings)  # [batch_size, seq_len, hidden_dim]
        output = self.dropout(self.linear(self.non_lin(output)))  # [batch_size, seq_len, hidden_dim]
        projection = self.projection(self.non_lin(output))  # [batch_size, seq_len, vocab_size]

        return projection


class GRU4Rec(BaseRecommender):
    def __init__(self, vocab, task, batch_size=64, hidden_dim=256, num_epoch=10,
                 device='cpu',
                 model_path='',
                 load=False):
        self.task = task
        if os.path.isfile(model_path) and load:
            self.model = torch.load(model_path)
        else:
            self.model = GRU(hidden_dim=hidden_dim, vocab_size=len(vocab)).to(device)
        self.device = device
        self.criterion = None
        self.optimizer = None
        self.num_epoch = num_epoch
        self.vocab = vocab
        self.batch_size = batch_size
        self.task.add_parameters({'hidden_dim': hidden_dim,
                                  'batch_size': batch_size,
                                  'num_epoch': num_epoch})

    def prepare_data(self, train):

        train_sessions = train.groupby('user_id')['item_id'].apply(lambda x: np.array(x))
        item2ind = {char: i for i, char in enumerate(self.vocab)}
        ind2item = {i: char for char, i in item2ind.items()}
        self.criterion = nn.CrossEntropyLoss(ignore_index=item2ind['<pad>'])
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        train_dataset = ItemsDataset(train_sessions, item2ind=item2ind)
        train_dataloader = DataLoader(
            train_dataset, shuffle=True,
            collate_fn=train_dataset.collate_fn_with_padding,
            batch_size=self.batch_size)

        return train_dataloader, item2ind, ind2item

    def fit(self, train_data):
        losses = []

        for epoch in range(self.num_epoch):
            epoch_losses = []
            self.model.train()
            for batch in tqdm(train_data, desc=f'Training epoch {epoch}:'):
                self.optimizer.zero_grad()
                logits = self.model(batch['input_ids']).flatten(start_dim=0, end_dim=1)
                loss = self.criterion(
                    logits, batch['target_ids'].flatten())
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            losses.append(sum(epoch_losses) / len(epoch_losses))
        self.task.upload_artifact('gru_model.pth', artifact_object=self.model)
        return losses

    def generate_items(self, item2ind, ind2item, items, n=10):
        device = 'cpu'
        model = self.model.to(device)
        input_ids = [item2ind.get(char, item2ind['<unk>']) for char in items]
        input_ids = torch.LongTensor(input_ids).to(device)
        items = []
        scores = []
        model.eval()
        with torch.no_grad():
            for i in range(n):
                next_char_distribution = self.model(input_ids)[-1]
                next_char = next_char_distribution.squeeze().argmax()

                input_ids = torch.cat([input_ids, next_char.unsqueeze(0)])
                items.append(ind2item[next_char.item()])
                scores.append(float(softmax(next_char_distribution).max().numpy()))
                # pred.append((ind2item[next_char.item()],float(softmax(next_char_distribution).max().numpy())))

        return items, scores

    def make_candidates(self, sessions, user_ids,
                        item2ind, ind2item,
                        n=20,
                        cand_path=None, load=False,
                        save_path='', save=False):
        if os.path.isfile(cand_path) and load:
            return pd.read_csv(cand_path)
        else:
            preds = []
            train_sessions = sessions.groupby('user_id')['item_id'].apply(lambda x: np.array(x))
            for userid in tqdm(user_ids):
                item, score = self.generate_items(items=train_sessions.loc[userid],
                                                  n=n,
                                                  ind2item=ind2item,
                                                  item2ind=item2ind)
                preds.append((userid, item, score))
            candidates = pd.DataFrame(preds, columns=['user_id', 'item_id', 'rank'])
            candidates = candidates.explode(['item_id', 'rank'])
            candidates = candidates.sort_values(by='rank', ascending=False)
            candidates['rank'] = candidates.groupby('user_id').cumcount() + 1
        if os.path.isfile(save_path) and save:
            candidates.to_csv(save_path)
        return candidates
