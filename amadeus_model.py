"""The full net for Amadeus"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from performer_pytorch import PerformerLM, AutoregressiveWrapper

def top_k(logits, thresh = 0.9):
    k = int((1 - thresh) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class Amadeus(nn.Module):
    """Wrapper containing the Amadeus model
    
    Attributes:
        enc: encoder module
        dec: decoder module
        in_seq_len: maximum possible input sequence length
        out_seq_len: maximum possible generated sequence length
        ignore_index: token id to ignore during training
    """


    def __init__(self, num_tokens: int, dims: int = 512, \
        enc_seq_len: int = 1024, enc_layers: int = 1, \
        dec_seq_len: int = 1024, dec_layers: int = 10, \
        heads: int = 8, nb_features: int = 64, \
        pad_token_id: int = 0):
        """Initializer for Amadeus model

        Args:
            num_tokens: number of tokens in the model
            dims: optional; number of dimensions for both encoder and decoder
            max_seq_len: optional; max producable sequence length (should be max length
                of encoder + decoder)
            enc_layers: optional; number of encoder layers
            dec_layers: optional; number of decoder layers
            heads: optional; number of attention heads
            nb_features: optional; number of random features
            kernel_fn: optional; the underlying kernel function to use
            pad_token_id: optional; the id of the [PAD] token to ignore during training
        """
        super().__init__()

        self.ignore_index = pad_token_id

        enc = PerformerLM(num_tokens=num_tokens, max_seq_len=enc_seq_len, \
            dim=dims, depth=enc_layers, heads=heads, nb_features=nb_features, \
            reversible=True, emb_dropout=0.1, ff_dropout=0., attn_dropout=0.1)
        # dec = ReformerLM(num_tokens=num_tokens, max_seq_len=dec_seq_len, \
        #     n_hashes=4, dim=dims, depth=dec_layers, heads=heads, causal=True)
        dec = PerformerLM(num_tokens=num_tokens, max_seq_len=dec_seq_len, \
            dim=dims, depth=dec_layers, heads=heads, nb_features=nb_features, \
            reversible=True, causal=True, cross_attend=True, \
            emb_dropout=0.1, ff_dropout=0.1, attn_dropout=0.1, ff_glu=True)

        enc.token_emb = dec.token_emb

        # self.enc = AutoregressiveWrapper(enc, pad_value=0)
        self.enc = enc
        # self.dec = Autopadder(dec)
        self.dec = AutoregressiveWrapper(dec, ignore_index=0, pad_value=0)
        
        self.in_seq_len = enc.max_seq_len
        self.out_seq_len = dec.max_seq_len

    def eval(self, fix_proj_matrices: bool = False):
        """Set to eval mode"""
        super().eval()
        self.enc.eval()
        self.dec.eval()
        
        if fix_proj_matrices:
            self.enc.fix_projection_matrices_() # As of performer-pytorch 0.5.1

        # https://github.com/lucidrains/performer-pytorch/issues/10
        # for layer in self.dec.performer.performer.net.layers:
        #     layer[0].fn.fast_attention.redraw_projection = False
        #     projection_matrix = layer[0].fn.fast_attention.create_projection()
        #     layer[0].fn.fast_attention.register_buffer('projection_matrix', projection_matrix)

    @torch.no_grad()
    def generate(self, input_seq: torch.Tensor, start_tokens: torch.Tensor, \
        eos_token: int = -1, mask: Optional[torch.Tensor] = None, \
        temperature: float = 1., filter_thresh: float = 0.9) -> torch.Tensor:
        """Generate a sequence based on the given input

        Args:
            start_tokens: starting token(s) to begin the decode
            eos_token: id of the [EOS] token to terminate sentence, if -1,
                this method will generate to seq_len
            mask: optional; input mask for the output sequence
            temperature: temperature for the topk softmax
            filter_thresh: threshold for the topk filter
        Returns:
            The resulting, generated sequence
        """
        assert len(input_seq.shape) == len(start_tokens.shape)
        num_dims = len(input_seq.shape)
        if num_dims == 1:
            input_seq = input_seq[None, :]
            start_tokens = start_tokens[None, :]

        encodings = self.enc(input_seq, mask=mask, return_encodings=True)
        dec = self.dec.generate(start_tokens, self.out_seq_len, \
            eos_token=eos_token, temperature=temperature, filter_thres=filter_thresh, \
            context=encodings, context_mask=mask)
        if num_dims == 1:
            dec = dec.squeeze(0)
        return dec


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs):
        """Forward pass calculates loss for model
        
        Args:
            x: input tensor for forward pass, otherwise, a tensor holding both
                input and targets, where x has the shape
                (2, batch_size, seq_len)
        """

        mask = kwargs.pop('mask', torch.ones_like(inputs, dtype=bool))
        encodings = self.enc(inputs, mask=mask, return_encodings=True)
        if self.out_seq_len < self.in_seq_len:
            encodings = encodings[:, :self.out_seq_len]

        return self.dec(targets, context=encodings)
