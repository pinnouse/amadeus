import torch
from tokenizers import BertWordPieceTokenizer, Encoding

from amadeus_model import Amadeus

tokenizer = BertWordPieceTokenizer('data/bert-base-uncased-vocab.txt', lowercase=True)

model = Amadeus(num_tokens=tokenizer.get_vocab_size(), enc_seq_len=1024, dec_seq_len=512)
checkpoint = torch.load('checkpoints/amadeus-performer-2020-11-25-00.20.57-300.pt')
model.eval(True)
# model.load_state_dict(torch.load('models/amadeus-performer-2020-11-06-12.47.52.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
model.cuda()

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