#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import os, sys
from typing import Dict, Tuple, List, Optional
import re
import random
from datetime import datetime
from argparse import ArgumentParser, ArgumentTypeError

import torch
import torch.nn as nn
import torch.nn.functional as F

from performer_pytorch import PerformerLM

# from torchviz import make_dot

from adafactor import Adafactor

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

parser = ArgumentParser()
parser.add_argument('-o', '--o', default='./', dest='output', help='Location of output(s)')
parser.add_argument('-c', '--use_cuda', type=str2bool, dest='use_cuda', default=True, help="Use cuda if cuda supported")

use_cuda = parser.parse_args().use_cuda

device = 'cuda' if torch.has_cuda and use_cuda else 'cpu'

model_dir = parser.parse_args().output


# # Data Parsing
# Parse through subtitles

# In[2]:


SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
SEP_TOKEN = '<sep>'

multiplier = [60, 60 * 60, 24 * 60 * 60]
def get_time(timestr: str) -> int:
    time = timestr.split(':')
    final_time = 0
    ms = float(time[-1]) * 1000
    final_time += int(ms)
    for i in range(len(time)-2):
        t = time[-2-i]
        final_time += multiplier[i] * int(t)
    return final_time

normalize_pattern = re.compile(r'(\{[\\\*][\w\(\)\\\,\*]*|\})', re.M)
sub_space = re.compile(r'(\{|\\[nN])', re.M)
insert_space = re.compile(r'([\w\"])([\.\!\,\?])')
def normalize_text(text: str) -> str:
    text = normalize_pattern.sub('', text)
    text = sub_space.sub(' ', text)
    text = re.sub(r'([\'\"])', r' \1 ', text)
    text = re.sub(r'([\.\!\?])(\w)', r'\1 \2', text)
    text = ' '.join(text.split())
    return insert_space.sub(r'\1 \2', text)

number_match = re.compile(r'\d+')
def match_num(text: str) -> int:
    x = number_match.findall(text)
    return int(x[0] if len(x) > 0 else 0)

class ParsedVocab:
    """The parsed vocabulary."""

    def __init__(self, words: List[Tuple[str, int]], longest: int = 0):
        words.sort(key=lambda x : x[1], reverse=True)
        # words = [('<sos>', 1), ('<eos>', 1)] + words
        words = [(PAD_TOKEN, 0), (UNK_TOKEN, 0), (SEP_TOKEN, 0)] + words
        self._word2freq = words
        self._word2ind = {}
        self._words = list(map(lambda x: x[0], words))
        self._longest = longest

        for i, (w, _) in enumerate(words):
            self._word2ind[w] = i

    def __getitem__(self, i) -> str:
        return self._words[i]

    def __str__(self) -> str:
        return f'Parsed Vocabulary ({len(self._words)} words)'
    
    def get_words(self) -> List[str]:
        return self._words
    
    def get_index(self, word) -> int:
        if word in self._word2ind:
            return self._word2ind[word]
        return -1

    def sen_to_seq(self, sentence: str, seq_len: int = 0, add_tokens: bool = True, add_pad: bool = False) -> List[int]:
        sentence = normalize_text(sentence)
        if seq_len <= 0:
            seq_len = self._longest
        l = []
        if add_tokens:
            sentence = f'{SOS_TOKEN} {sentence}'
        s = sentence.split()
        for i in range(min(len(s), seq_len - add_tokens)):
            word = s[i]
            if word in self._word2ind:
                l.append(self._word2ind[word])
            else:
                l.append(self._word2ind[UNK_TOKEN])
        if add_tokens:
            l += [self._word2ind[EOS_TOKEN]]
        if len(s) < seq_len and add_pad:
            l += [self._word2ind[PAD_TOKEN]] * (seq_len - len(l))
        return l

    def conv_to_seq(self, conversation: List[Dict[str, object]], max_seq_len: int = 0) -> List[int]:
        l = [self._word2ind[SOS_TOKEN]]
        for conv in conversation:
            l.extend(self.sen_to_seq(conv['line'], add_tokens=False))
            l.append(self._word2ind[SEP_TOKEN])

        if len(l) > 0:
            l = l[:max_seq_len]
            l[-1] = self._word2ind[EOS_TOKEN]
        l += [self._word2ind[PAD_TOKEN]] * (max_seq_len - len(l))
        return l
        

    def gen_mask(self, tokenized_sentence: List[int]) -> List[bool]:
        """Creates a mask for the given tokenized sentence.

        >>> pv = ParsedVocab([('<sos>', 0), ('<eos>', 0), ('hi', 0)])
        >>> x = pv.sen_to_seq('<sos> hi <eos> <unk> <unk>', )
        >>> pv.gen_mask(x)
        [1, 1, 1, 0, 0]
        """
        l = [True] * len(tokenized_sentence)
        i = len(tokenized_sentence) - 1
        while i >= 0 and tokenized_sentence[i] == self._word2ind[PAD_TOKEN]:
            l[i] = False
            i -= 1
        return l

class Vocab:
    """Regulard vocabulary for holding the conversations and number of words."""

    DEFAULT_CONTEXT = 'default'

    def __init__(self, conversation_depth: int = 4):
        self.words = {}
        self._context = Vocab.DEFAULT_CONTEXT
        self.conversations = {}
        self._held_conversations = {}
        self.conversation_depth = conversation_depth
        self.longest = 0

    def add_word(self, word: str) -> None:
        word = word.lower()
        if not word in self.words:
            self.words[word] = 0
        self.words[word] += 1

    def add_sentence(self, sentence: str) -> None:
        sentence = f'{SOS_TOKEN} {sentence} {EOS_TOKEN}'
        [self.add_word(s) for s in sentence.split()]

    def switch_context(self, new_context: str) -> None:
        if self._context in self._held_conversations and len(self._held_conversations[self._context]) > self.conversation_depth:
            self.conversations[self._context].append(self._held_conversations[self._context][
                -self.conversation_depth:
            ])
        self._context = new_context

    def add_conversation(self, conversation: Dict[str, object]) -> None:
        if not self._context in self.conversations:
            self.conversations[self._context] = []
            self._held_conversations[self._context] = [conversation]
            return
        self.add_line(conversation)
        line = self._held_conversations[self._context][-1]['line'].split()
        if len(line) > self.longest:
            self.longest = len(line)
    
    def add_line(self, conversation: Dict[str, object]) -> bool:
        if not self._context in self._held_conversations or len(self._held_conversations[self._context]) == 0:
            self._held_conversations[self._context] = [conversation]
            return True
        hc = self._held_conversations[self._context] # Held Conversation
        lc = hc[-1] # Last conversation
        # Same speaker
        if (len(lc['speaker']) > 0 and lc['speaker'] == conversation['speaker']) or             (len(lc['speaker']) == 0 and len(conversation['speaker']) == 0 and len(conversation['line']) > 0 and conversation['line'][0].islower()) and             conversation['when'] - lc['when'] < 1000 * 60 * 1.5:
            hc[-1]['when'] = conversation['when']
            hc[-1]['line'] += f" {conversation['line']}"
            return False
        if len(self._held_conversations[self._context]) >= 2:
            self.conversations[self._context].append(self._held_conversations[self._context][
                -min(self.conversation_depth, len(hc)):
            ])
        hc.append(conversation)
        return True

    def parse_vocab(self) -> ParsedVocab:
        words = list(self.words.items())
        return ParsedVocab(words, self.longest)


# In[3]:


FOLDERS = ['ditfxx_subs', 'steins_gate_subs']
CONVERSATION_DEPTH = 4

vocab = Vocab(CONVERSATION_DEPTH)

for folder in FOLDERS:
    dir = os.listdir(os.path.join('data', folder))
    dir.sort(key=match_num)
    print(f'Parsing folder: {folder}')
    for f in dir:
        filepath = os.path.join(os.getcwd(), 'data', folder, f)
        if not os.path.isfile(filepath): continue
        print(f'  Opening file: {f}')
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as sub_file:
            is_event = False
            line = True
            while not is_event and line:
                line = sub_file.readline()
                if not line: break
                if line.rstrip() == "[Events]":
                    is_event = True
            current_format = False
            current_conversation = []
            
            vocab.switch_context(f)
            line = True
            # for line in sub_file.readlines():
            while line:
                try:
                    line = sub_file.readline()
                except UnicodeDecodeError:
                    print('    Error decoding a line, skipped.')
                if line.startswith('Format:'):
                    line = line[len('Format:'):].strip().split(', ')
                    current_format = line
                    continue
                if current_format == False or not line.startswith('Dialogue:'): continue
                line = line[len('Dialogue:'):].strip().split(',')
                line[len(current_format)-1] = ','.join(line[len(current_format)-1:])
                dialogue = dict(zip(current_format, line))
                if not dialogue['Style'] in ['main', 'Default']: continue
                # Extract variables
                speaker = dialogue['Name']
                text = normalize_text(dialogue['Text'])
                time = get_time(dialogue['Start'])

                # if len(current_conversation) > 0 and time - current_conversation[-1]['when'] > 1000 * 60 * 2:
                #     current_conversation = []
                # if len(current_conversation) > 0 and ((len(speaker) > 0 and current_conversation[-1]['speaker'] == speaker) or 
                # (len(speaker) == 0 and len(dialogue['Text']) > 0 and dialogue['Text'][0].islower())):
                #     current_conversation[-1]['line'] += f' {text}'
                #     current_conversation[-1]['when'] = time
                # else:
                #     vocab.add_conversation(current_conversation)
                #     current_conversation.append({
                #         'speaker': speaker,
                #         'line': text,
                #         'when': time
                #     })
                # if len(current_conversation) > CONVERSATION_DEPTH:
                #     current_conversation = current_conversation[CONVERSATION_DEPTH - len(current_conversation):]
                # if len(current_conversation) == 1:
                #     continue
                vocab.add_conversation({
                    'speaker': speaker,
                    'line': text,
                    'when': time
                })
                vocab.add_sentence(text)
            

pv = vocab.parse_vocab()
convos = 0
for k, c in vocab.conversations.items():
    convos += len(c)
print(f'Done! Num conversations: {convos}, num words: {len(pv.get_words())}, longest convo: {vocab.longest}')
# print(words[:100])


# In[4]:


x = list(vocab.conversations)
c = vocab.conversations[x[0]]
c[0][:]


# # Define Model
# Defining the actual AI model

# In[5]:


# 3 Sentences with 2 delims
seq_len = vocab.longest * (vocab.conversation_depth - 1) + 2

# Thanks to
# https://github.com/lucidrains/performer-pytorch
model = PerformerLM(
    num_tokens=len(pv.get_words()),
    max_seq_len=seq_len,
    dim=512,
    depth=6,
    heads=8,
    causal=False,
    nb_features=256,
    generalized_attention=False,
    kernel_fn=nn.ReLU(),
    reversible=True,
    ff_chunks=10,
    use_scalenorm=False,
    use_rezero=True
)

x = torch.randint(0, len(pv.get_words()), (1, seq_len))
mask = torch.ones_like(x).bool()

y = model(x, mask=mask)

print(y.size())

# make_dot(y.mean(), params=dict(model.named_parameters()))
print(x.shape)


# # Train model

# In[6]:


optimizer = Adafactor(model.parameters())
criterion = nn.CrossEntropyLoss()

PRINT_EVERY = 40

class ConversationIter:

    def __init__(self, vocab: Vocab, parsed_vocab: ParsedVocab, max_seq_len: int):
        self._vocab = vocab
        self._parsed_vocab = parsed_vocab
        self._context = random.choice(list(vocab.conversations))
        self.max_seq_len = max_seq_len

        self._i = 0

    def __iter__(self):
        self._context = random.choice(list(self._vocab.conversations))
        self._i = 0
        return self

    def __next__(self):
        if self._i >= len(self._vocab.conversations[self._context]):
            raise StopIteration
        x = self._vocab.conversations[self._context][self._i]
        input = torch.tensor(self._parsed_vocab.conv_to_seq(x[:-1], self.max_seq_len))
        target = torch.tensor(self._parsed_vocab.sen_to_seq(x[-1]['line'], seq_len=self.max_seq_len, add_pad=True))
        self._i += 1
        return input, target

def train(conv_iter: ConversationIter):
    model.train()
    accrued_loss = 0
    start = datetime.now()
    for i, (input, target) in enumerate(conv_iter):
        input.to(device)
        target.to(device)

        mask = torch.tensor(pv.gen_mask(input))

        input.unsqueeze_(0)
        target.unsqueeze_(0)
        mask.unsqueeze_(0)

#         input = F.one_hot(input, len(pv.get_words()))
#         input.transpose_(0, 1)

#         print(input.shape, target.shape)

        optimizer.zero_grad()
        output = model(input, mask=mask)
#         output.transpose_(1, 2)
        loss = criterion(output.squeeze(0), target.squeeze(0))
        loss.backward()
        optimizer.step()
        
        accrued_loss += loss.item()
        
        if (i + 1) % PRINT_EVERY == 0:
            print(f'  Iter {i+1} (Took {(datetime.now() - start).total_seconds():.3f}s): AverageLoss: {accrued_loss/PRINT_EVERY:.4f}')
            accrued_loss = 0
            start = datetime.now()


# In[ ]:


TRAIN_EPOCHS = 40
SAVE_EVERY = 5

conv_iter = ConversationIter(vocab, pv, seq_len)

def save_checkpoint(epoch: int):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(model_dir, f'checkpoints/amadeus-performer-{epoch}.pt'))

for epoch in range(TRAIN_EPOCHS):
    print(f'Training epoch #{epoch+1} of {TRAIN_EPOCHS}:')
    total = datetime.now()
    train(conv_iter)
    print(f'Epoch {epoch+1} took {(datetime.now()-total).total_seconds():.3f}s\n\n')
    
    if (epoch + 1) % SAVE_EVERY == 0:
        print('Saving checkpoint...')
        save_checkpoint(epoch)


# In[ ]:




