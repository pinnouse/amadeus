import torch
from tokenizers import BertWordPieceTokenizer, Encoding

from amadeus_model import Amadeus

tokenizer = BertWordPieceTokenizer('data/bert-base-uncased-vocab.txt', lowercase=True)

model = Amadeus(num_tokens=tokenizer.get_vocab_size(), enc_seq_len=2048, dec_seq_len=512)
model.load_state_dict(torch.load('models/amadeus-performer-2020-11-01-22.37.50.pt'))
model.cuda()
model.eval()

run = True

sentences = []

while run:
    try:
        sentence = input('> ')
        if sentence in ['quit', 'exit']:
            run = False
            continue
        sentences.append(tokenizer.encode(sentence))
        if len(sentences) > 3:
            sentences = sentences[-3:]
        input_seq = torch.tensor(Encoding.merge(sentences[:]).ids).cuda()
        start_tokens = torch.tensor([tokenizer.token_to_id('[CLS]')]).cuda()
        out = model.generate(input_seq=input_seq, start_tokens=start_tokens, eos_token=tokenizer.token_to_id('[SEP]'))
        response = tokenizer.decode(out.tolist())
        sentences.append(tokenizer.encode(response))
        print(response)
    except KeyboardInterrupt:
        run = False