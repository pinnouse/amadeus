import torch
from tokenizers import BertWordPieceTokenizer

from amadeus_model import Amadeus

tokenizer = BertWordPieceTokenizer('data/bert-base-uncased-vocab.txt', lowercase=True)

model = Amadeus(num_tokens=tokenizer.get_vocab_size(), enc_seq_len=4096, dec_seq_len=1024)
model.load_state_dict(torch.load('models/amadeus-performer-2020-11-03-16.54.13.pt'))
model.eval(fix_proj_matrices=True)

in_seq = torch.randint(0, tokenizer.get_vocab_size(), (1, model.in_seq_len))
out_seq = torch.randint(0, tokenizer.get_vocab_size(), (1, model.out_seq_len))

traced_script_model = torch.jit.trace(model, (in_seq, out_seq), check_trace=False)
traced_script_model.save('traced.pt')