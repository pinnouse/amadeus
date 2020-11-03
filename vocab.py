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

    def __init__(self, conversation_depth: int = 4):
        self.words = {}
        self._context = Vocab.DEFAULT_CONTEXT
        self.conversations = {}
        self._held_conversations = {}
        self.conversation_depth = conversation_depth
        self.longest = 0
        self.longest_tokenized = 0
        self.tokenizer = BertWordPieceTokenizer('data/bert-base-uncased-vocab.txt', lowercase=True)

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
        # Same speaker
        if (len(lc['speaker']) > 0 and lc['speaker'] == conversation['speaker']) or \
            (len(lc['speaker']) == 0 and len(conversation['speaker']) == 0 and len(conversation['line']) > 0 and conversation['line'][0].islower()) and \
            conversation['when'] - lc['when'] < 1000 * 60 * 1.5:
            hc[-1]['when'] = conversation['when']
            hc[-1]['line'] += f" {conversation['line']}"
            return False
        if len(self._held_conversations[self._context]) >= 2:
            self.conversations[self._context].append(self._held_conversations[self._context][
                -min(self.conversation_depth, len(hc)):
            ])
        hc.append(conversation)
        return True

    def get_tokenizer(self) -> BaseTokenizer:
        return self.tokenizer

class ConversationIter:

    def __init__(self, conversations: List[List[object]], in_seq_len: int, \
        out_seq_len, tokenizer: BaseTokenizer, batch_size: int = 1):
        self._context = 0
        self._conversations = conversations
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len

        self._batch_size = batch_size

        self._tokenizer = tokenizer

        self._i = 0

        if len(self._conversations) == 0:
            raise Exception("Not enough conversations in the Vocab, could not make ConversationIter")

    def __iter__(self):
        self._context = random.randrange(0, len(self._conversations))
        self._i = 0
        return self

    def __next__(self) -> Tuple[List[Encoding], List[Encoding]]:
        conv = self._conversations[self._context]
        if self._i >= len(conv):
            raise StopIteration

        inputs, targets = [], []
        for _ in range(self._batch_size):
            try:
                x = conv[self._i]
                l = [self._tokenizer.encode(y['line']) for y in x]
                # i = Encoding.merge()
                i = Encoding.merge(l[:-1])
                t = l[-1]
                i.pad(self.in_seq_len)
                t.pad(self.out_seq_len)

                inputs.append(i)
                targets.append(t)

                self._i += 1
            except IndexError:
                if min(len(inputs), len(targets)) == 0:
                    raise StopIteration
                else:
                    return inputs, targets
        return inputs, targets

    def random_sample(self, pad_in: bool = False, pad_out: bool = False) -> Tuple[List[Encoding], List[Encoding]]:
        self._context = random.randrange(0, len(self._conversations))
        self._i = random.randrange(0, len(self._conversations[self._context]))
        conv = self._conversations[self._context][self._i]
        l = [self._tokenizer.encode(y['line']) for y in conv]
        i = Encoding.merge(l[:-1])
        t = l[-1]
        if pad_in:
            i.pad(self.in_seq_len)
        if pad_out:
            t.pad(self.out_seq_len)
        return [i], [t]