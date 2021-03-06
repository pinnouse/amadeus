"""Vocab module

handles all the vocabulary parsing and encoding, as well as preparing training
data/batches
"""

import random
from typing import Dict, Tuple, List, Optional

from tokenizers import BertWordPieceTokenizer, Encoding
from tokenizers.implementations import BaseTokenizer


class Vocab:
    """Regulard vocabulary for holding the conversations and number of words."""

    DEFAULT_CONTEXT = 'default'
    SENTENCE_CUTOFF_DURATION = 1000 * 6

    def __init__(self, max_seq_len: int, conversation_depth: int = 4):
        self.words = {}
        self._context = Vocab.DEFAULT_CONTEXT
        self.conversations = {}
        self._held_conversations = {}
        self.conversation_depth = conversation_depth
        self.longest = 0
        self.longest_tokenized = 0
        self.tokenizer = BertWordPieceTokenizer('data/bert-base-uncased-vocab.txt', lowercase=True)
        self.tokenizer.enable_truncation(max_seq_len)

    def add_word(self, word: str) -> None:
        word = word.lower()
        if not word in self.words:
            self.words[word] = 0
        self.words[word] += 1

    def add_sentence(self, sentence: str) -> None:
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
        self.add_line(conversation)
        lc = self._held_conversations[self._context][-1]
        line = lc['line'].split()
        if len(line) > self.longest:
            self.longest = len(line)
        tokenized = self.tokenizer.encode(lc['line'])
        if len(tokenized.ids) > self.longest_tokenized:
            self.longest_tokenized = len(tokenized.ids)
    
    def add_line(self, conversation: Dict[str, object]) -> bool:
        if not self._context in self._held_conversations or len(self._held_conversations[self._context]) == 0:
            self._held_conversations[self._context] = [conversation]
            return True
        hc = self._held_conversations[self._context] # Held Conversation
        lc = hc[-1] # Last conversation
        
        same_speaker = len(lc['speaker']) > 0 and lc['speaker'] == conversation['speaker'] and not lc['speaker'] in ['NTP', 'Text']
        continuing_line = (len(lc['speaker']) == 0 or lc['speaker'] in ['NTP', 'Text']) and \
            (len(conversation['speaker']) == 0 or conversation['speaker'] in ['NTP', 'Text']) \
            and len(conversation['line']) > 0 and conversation['line'][0].islower()

        if same_speaker or continuing_line and conversation['when'] - lc['when'] < Vocab.SENTENCE_CUTOFF_DURATION:
            lc['when'] = conversation['when']
            lc['line'] += f" {conversation['line']}"
            return False
        if len(self._held_conversations[self._context]) >= 2:
            self.conversations[self._context].append(hc[
                -min(self.conversation_depth, len(hc)):
            ])
        hc.append(conversation)
        if conversation['when'] - lc['when'] >= Vocab.SENTENCE_CUTOFF_DURATION:
            self._held_conversations[self._context] = [conversation]
        return True

    def get_tokenizer(self) -> BaseTokenizer:
        return self.tokenizer

    def get_conversations(self, in_seq_len: int, out_seq_len: int, \
        add_two_person: bool = True) -> List[Dict[str,List[int]]]:
        conversations = []
        for conversation in self.conversations.values():
            for dialogue in conversation:
                inputs = [self.tokenizer.encode(y['line']) for y in dialogue[:-1]][::-1]
                target = self.tokenizer.encode(dialogue[-1]['line'])
                target.pad(out_seq_len)
                target.truncate(out_seq_len)
                if add_two_person and self.conversation_depth > 2 and len(dialogue) > 2 and len(inputs) > 0:
                    inputs[0].pad(in_seq_len)
                    inputs[0].truncate(in_seq_len)
                    conversations.append({
                        'inputs': inputs[0].ids,
                        'target': target.ids,
                        'mask': inputs[0].attention_mask
                    })
                inputs = Encoding.merge(inputs)
                inputs.pad(in_seq_len)
                inputs.truncate(in_seq_len)
                conversations.append({
                    'inputs': inputs.ids,
                    'target': target.ids,
                    'mask': inputs.attention_mask
                })
        return conversations

class ConversationIter:

    def __init__(self, conversations: List[List[Encoding]], batch_size: int = 1):
        self._conversations = []
        self._batch_size = batch_size
        self._i = 0

        for conv in conversations:
            self._conversations.append((conv['inputs'], conv['target'], conv['mask']))

        if len(self._conversations) == 0:
            raise Exception("Not enough conversations in the Vocab, could not make ConversationIter")

    def __iter__(self):
        random.shuffle(self._conversations)
        self._i = 0
        return self

    def __next__(self) -> Tuple[List[int], List[int], List[int]]:
        inputs, targets, masks = [], [], []
        while min(len(inputs), len(targets), len(masks)) < self._batch_size and self._i < len(self._conversations):
            i, t, m = self._conversations[self._i]
            inputs.append(i)
            targets.append(t)
            masks.append(m)
            self._i += 1
        if min(len(inputs), len(targets)) <= 0:
            raise StopIteration
        return inputs, targets, masks

    def random_sample(self, amount: int = 0) -> Tuple[List[Encoding], List[Encoding]]:
        indices = random.sample(range(len(self._conversations)), min(amount, len(self._conversations))) if amount > 0 else range(len(self._conversations))
        inputs, targets, masks = [], [], []
        for j in indices:
            i, t, m = self._conversations[j]
            inputs.append(i)
            targets.append(t)
            masks.append(m)
        return inputs, targets, masks
