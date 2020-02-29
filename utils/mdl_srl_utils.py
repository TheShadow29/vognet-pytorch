"""
Some helpful functions/classes are defined
"""
import torch
from torch import nn
# from fairseq.models import FairseqEncoder
from torch.nn import functional as F
from fairseq import utils


def combine_first_ax(inp_tensor, keepdim=False):
    inp_shape = inp_tensor.shape
    if keepdim:
        return inp_tensor.view(
            1, inp_shape[0] * inp_shape[1], *inp_shape[2:])
    return inp_tensor.view(
        inp_shape[0] * inp_shape[1], *inp_shape[2:])


def uncombine_first_ax(inp_tensor, s0):
    "s0 is the size(0) intended, usually B"
    inp_shape = inp_tensor.shape
    size0 = inp_tensor.size(0)
    assert size0 % s0 == 0
    s1 = size0 // s0
    return inp_tensor.view(
        s0, s1, *inp_shape[1:])


def do_cross(x1, x2=None, dim1=1, op='add'):
    """
    if x2 is none do x1(row) + x1(col)
    else x1(row) + x2(col)
    dim1, dim2 are first and second dimension
    to be used for crossing.
    Both x1, x2 should have same shape except
    for at most one dimension

    if input is B x C x D x E with dim1=1
    B x C x D x E ->
    B x C x 1 x D x E -> B x C x C x D x E;
    B x 1 x C x D x E -> B x C x C x D x E;
    and then add

    op = 'add', 'subt', 'mult' or 'concat'
    """
    x1_shape = x1.shape
    if x2 is None:
        x2 = x1
    assert x1.shape == x2.shape
    x1_dim = len(x1_shape)
    out_shape = tuple((*x1_shape[:dim1], x1_shape[dim1], *x1_shape[dim1:]))
    if dim1 < x1_dim:
        x1_row = x1.view(*x1_shape[:dim1+1], 1, *
                         x1_shape[dim1+1:]).expand(out_shape)
        x2_col = x2.view(*x1_shape[:dim1], 1, *
                         x1_shape[dim1:]).expand(out_shape)
    else:
        x1_row = x1.view(*x1_shape[:dim1+1], 1)
        x2_col = x2.view(*x1_shape[:dim1], 1, x1_shape[dim1])

    if op == 'add':
        return (x1_row + x2_col) / 2
    elif op == 'mult':
        return (x1_row * x2_col)
    elif op == 'concat':
        return torch.cat([x1_row, x2_col], dim=-1)
    elif op == 'subtract':
        return (x1_row - x2_col)


class LSTMEncoder(nn.Module):
    """LSTM encoder."""

    def __init__(
            self, cfg, comm, embed_dim=512, hidden_size=512, num_layers=1,
            dropout_in=0.1, dropout_out=0.1, bidirectional=False,
            left_pad=True, pretrained_embed=None, padding_value=0.,
            num_embeddings=0, pad_idx=0
    ):
        super().__init__()
        self.cfg = cfg
        self.comm = comm
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = num_embeddings
        self.padding_idx = pad_idx
        embed_dim1 = embed_dim
        if pretrained_embed is None:
            self.embed_tokens = nn.Embedding(
                num_embeddings, embed_dim1, self.padding_idx
            )
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()
        # embed tokens
        x = self.embed_tokens(src_tokens)

        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths.data.tolist(), enforce_sorted=False)

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(
            packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_value)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -
                                1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': (encoder_padding_mask
                                     if encoder_padding_mask.any() else None)
        }

    def reorder_only_outputs(self, outputs):
        """
        outputs of shape : T x B x C -> B x T x C
        """
        return outputs.transpose(0, 1).contiguous()

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class SimpleAttn(nn.Module):
    def __init__(self, qdim, hdim):
        super().__init__()
        self.lin1 = nn.Linear(qdim, hdim)
        self.lin2 = nn.Linear(qdim, hdim)
        self.lin3 = nn.Linear(hdim, 1)

    def forward(self, qvec, qlast, inp):
        """
        qvec: B x nsrl x qdim
        qlast: B x 1 x qdim
        """
        # B x nv x  nsrl x hdim
        B, num_verbs, nsrl, qd = qvec.shape
        qvec_enc = self.lin1(qvec)
        # B x nv x 1 x hdim
        qlast_enc = self.lin2(qlast)

        hdim = qlast_enc.size(-1)

        # B x nv x nsrl x hdim
        q1_enc = torch.tanh(
            qvec_enc +
            qlast_enc.view(
                B, num_verbs, 1, hdim
            ).expand(
                B, num_verbs, nsrl, hdim
            )
        )
        # B x nv x nsrl
        u1 = self.lin3(q1_enc).squeeze(-1)
        # B x nv x nsrl
        attns = F.softmax(u1, dim=-1)

        # B x nv x nsrl x qdim
        qvec_out = attns.unsqueeze(-1).expand_as(qvec) * qvec

        return qvec_out
