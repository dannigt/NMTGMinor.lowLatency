import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer, PositionalEncoding, variational_dropout, PrePostProcessing
from onmt.modules.BaseModel import NMTModel, Reconstructor, DecoderState
import onmt
from onmt.modules.WordDrop import embedded_dropout
from torch.utils.checkpoint import checkpoint
from collections import defaultdict


def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward



class MixedEncoder(nn.Module):


    def __init__(self,text_encoder,audio_encoder):

        super(MixedEncoder, self).__init__()


        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder

    def forward(self, input, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x len_src (wanna tranpose)

        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src

        """

        """ Embedding: batch_size x len_src x d_model """

        if(input.dim() == 2):
            return self.text_encoder.forward(input)
        else:
            return self.audio_encoder.forward(input)




class TransformerEncoder(nn.Module):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)
        
    """
    
    def __init__(self, opt, dicts, positional_encoder,encoder_type):
    
        super(TransformerEncoder, self).__init__()
        
        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        if hasattr(opt,'encoder_layers') and opt.encoder_layers != -1:
            self.layers = opt.encoder_layers
        else:
            self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.time = opt.time
        self.version = opt.version
        self.input_type = encoder_type

        # input lookup table
        if encoder_type != "text":
            self.audio_trans = nn.Linear(dicts, self.model_size)
        else:
            self.word_lut = nn.Embedding(dicts.size(),
                                         self.model_size,
                                         padding_idx=onmt.Constants.PAD)

        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        elif opt.time == 'gru':
            self.time_transformer = nn.GRU(self.model_size, self.model_size, 1, batch_first=True)
        elif opt.time == 'lstm':
            self.time_transformer = nn.LSTM(self.model_size, self.model_size, 1, batch_first=True)
        
        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)
        
        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')
        
        self.positional_encoder = positional_encoder

        self.limit_rhs_steps = opt.limit_rhs_steps

        self.build_modules(limit_rhs_steps=opt.limit_rhs_steps)
        if self.limit_rhs_steps is not None:
            largest_rhs_mask = positional_encoder.len_max + self.limit_rhs_steps
            rhs_mask = torch.BoolTensor(np.triu(np.ones((largest_rhs_mask, largest_rhs_mask)),
                                                k=1 + self.limit_rhs_steps).astype('uint8'))
            self.register_buffer('rhs_mask', rhs_mask)

        if opt.freeze_encoder:
            for p in self.parameters():
                p.requires_grad = False
                print(p.requires_grad)

    def build_modules(self, limit_rhs_steps=None):
        self.layer_modules = nn.ModuleList([EncoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size,
                                                         self.attn_dropout, limit_rhs_steps=limit_rhs_steps) for _ in range(self.layers)])

    def forward(self, input, **kwargs):
        """
        Inputs Shapes: 
            input: batch_size x len_src (wanna tranpose)
        
        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src 
            
        """

        """ Embedding: batch_size x len_src x d_model """
        if self.input_type == "text":
            mask_src = input.data.eq(onmt.Constants.PAD).unsqueeze(1)  # batch_size x len_src x 1 for broadcasting
            emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        else:

            mask_src = input.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).unsqueeze(1)
            input = input.narrow(2, 1, input.size(2) - 1)
            emb = self.audio_trans(input.contiguous().view(-1, input.size(2))).view(input.size(0),
                                                                                    input.size(1), -1)

        """ Scale the emb by sqrt(d_model) """
        
        emb = emb * math.sqrt(self.model_size)

        """ Adding positional encoding """
        emb = self.time_transformer(emb)

        emb = self.preprocess_layer(emb)
        
        context = emb.transpose(0, 1).contiguous()

        for i, layer in enumerate(self.layer_modules):
            
            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:
                if self.limit_rhs_steps is not None:
                    # This also needs the mask! (when resuming from checkpoint)
                    context = checkpoint(custom_layer(layer), context, mask_src, rhs_mask=self.rhs_mask, layer_idx=i)
                else:
                    context = checkpoint(custom_layer(layer), context, mask_src)
            else:
                if self.limit_rhs_steps is not None:
                    # add RHS mask
                    context = layer(context, mask_src, rhs_mask=self.rhs_mask, layer_idx=i)
                else:
                    context = layer(context, mask_src)      # batch_size x len_src x d_model

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        context = self.postprocess_layer(context)

        output_dict = { 'context': context, 'src_mask': mask_src }

        # return context, mask_src
        return output_dict


class TransformerDecoder(nn.Module):
    """Encoder in 'Attention is all you need'

    Args:
        opt
        dicts


    """

    def __init__(self, opt, dicts, positional_encoder, ignore_source=False, feature_embedding=None):

        super(TransformerDecoder, self).__init__()

        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.time = opt.time
        self.version = opt.version
        self.encoder_type = opt.encoder_type
        self.ignore_source = ignore_source

        self.fixed_target_length = 0

        if hasattr(opt, 'fixed_target_length'):
            if opt.fixed_target_length == "int":
                self.fixed_target_length = 1
                print('Embedding')
            elif opt.fixed_target_length == "encoding":
                self.fixed_target_length = 2
                print('Encoding')
            elif opt.fixed_target_length == "forward_backward_encoding":
                self.fixed_target_length = 3
                print('Forward backward encoding')
            elif opt.fixed_target_length == "no":
                print('No fixed target len.')
            else:
                raise NotImplementedError

        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        else:
            raise NotImplementedError
        # elif opt.time == 'gru':
        #     self.time_transformer = nn.GRU(self.model_size, self.model_size, 1, batch_first=True)
        # elif opt.time == 'lstm':
        #     self.time_transformer = nn.LSTM(self.model_size, self.model_size, 1, batch_first=True)

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.word_lut = nn.Embedding(dicts.size(),
                                     self.model_size,
                                     padding_idx=onmt.Constants.PAD)

        # self.feat_lut = feature_embedding

        # if self.feat_lut is not None:
        #     self.enable_feature = True
        #     self.feature_projector = nn.Linear(opt.model_size * 2, opt.model_size)
        # else:
        self.enable_feature = False

        self.positional_encoder = positional_encoder

        if self.fixed_target_length == 1:
            self.length_lut = nn.Embedding(8192, opt.model_size, padding_idx=onmt.Constants.PAD)
            self.length_projector = nn.Linear(opt.model_size * 2, opt.model_size)

        len_max = self.positional_encoder.len_max
        mask = torch.ByteTensor(np.triu(np.ones((len_max,len_max)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

        self.build_modules()

    def build_modules(self):
        self.layer_modules = nn.ModuleList([DecoderLayer(self.n_heads, self.model_size,
                                                         self.dropout, self.inner_size,
                                                         self.attn_dropout,
                                                         ignore_source=self.ignore_source) for _ in range(self.layers)])

    def renew_buffer(self, new_len):

        print(new_len)
        self.positional_encoder.renew(new_len)
        mask = torch.ByteTensor(np.triu(np.ones((new_len,new_len)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

    def embedding_processing(self, input, input_attbs, freeze_embeddings=False):

        # len_tgt = input.size(1)  # target length
        # input_attbs = input_attbs.unsqueeze(1).repeat(1, len_tgt)  # make into same lenth as target len

        # if self.switchout > 0 and self.training:
        #     vocab_size = self.word_lut.weight.size(0)
        #     input = switchout(input, vocab_size, self.switchout)

        # if freeze_embeddings:
        #     with torch.no_grad:
        #         emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        #         if self.feat_lut is not None:
        #             attb_emb = self.feat_lut(input_attbs)
        #         else:
        #             attb_emb = []
        # else:
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)

        # if self.feat_lut is not None:
        #     attb_emb = self.feat_lut(input_attbs)
        # else:
        attb_emb = []

        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        if self.fixed_target_length == 2 or self.fixed_target_length == 3:

            if self.fixed_target_length == 3:
                emb = self.time_transformer(emb)
                emb = emb * math.sqrt(self.model_size)

            # add target length encoding
            tgt_length = input.data.ne(onmt.Constants.PAD).sum(1).unsqueeze(1).expand_as(input.data)
            index = torch.arange(input.data.size(1)).unsqueeze(0).expand_as(tgt_length).type_as(tgt_length)
            tgt_length = (tgt_length - index) * input.data.ne(onmt.Constants.PAD).long()

            num_timescales = self.model_size // 2
            log_timescale_increment = math.log(10000) / (num_timescales - 1)
            inv_timescales = torch.exp(torch.arange(0, num_timescales).float() * -log_timescale_increment)
            scaled_time = tgt_length.float().unsqueeze(2) * inv_timescales.unsqueeze(0).unsqueeze(0).type_as(emb)
            pos_emb = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), 2)
            emb = emb + pos_emb

        else:
            emb = self.time_transformer(emb)
        if isinstance(emb, tuple):
            emb = emb[0]

        # now emb should have size B x T x H

        # expand B to B x T
        if self.enable_feature:
            emb = torch.cat([emb, attb_emb], dim=-1)
            emb = torch.relu(self.feature_projector(emb))

        if self.fixed_target_length == 1:
            tgt_length = input.data.ne(onmt.Constants.PAD).sum(1).unsqueeze(1).expand_as(input.data)
            index = torch.arange(input.data.size(1)).unsqueeze(0).expand_as(tgt_length).type_as(tgt_length)
            tgt_length = (tgt_length - index) * input.data.ne(onmt.Constants.PAD).long()
            tgt_emb = self.length_lut(tgt_length);
            emb = torch.cat([emb, tgt_emb], dim=-1)

            emb = torch.relu(self.length_projector(emb))

        return emb



    def forward(self, input, context, src, src_type, input_attbs=None, **kwargs):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """

        """ Embedding: batch_size x len_tgt x d_model """


        # emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        # if self.time == 'positional_encoding':
        #     emb = emb * math.sqrt(self.model_size)
        # """ Adding positional encoding """
        # emb = self.time_transformer(emb)
        # if isinstance(emb, tuple):
        #     emb = emb[0]

        emb = self.embedding_processing(input, input_attbs, freeze_embeddings=False)
        # emb = self.preprocess_layer(emb)  # done later

        if context is not None:
            if src_type == "audio":
                mask_src = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).unsqueeze(1)
                pad_mask_src = src.data.narrow(2, 0, 1).squeeze(2).ne(onmt.Constants.PAD)  # batch_size x len_src
            else:

                mask_src = src.data.eq(onmt.Constants.PAD).unsqueeze(1)
                pad_mask_src = src.data.ne(onmt.Constants.PAD)
        else:
            mask_src = None
            pad_mask_src = None

        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).byte().unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)

        # transpose to T x B x H
        output = emb.transpose(0, 1).contiguous()

        # add dropout to initial embedding
        output = self.preprocess_layer(output)

        for i, layer in enumerate(self.layer_modules):

            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:

                output, coverage = checkpoint(custom_layer(layer), output, context, mask_tgt, mask_src)
                                                                              # batch_size x len_src x d_model

            else:
                output, coverage = layer(output, context, mask_tgt, mask_src) # batch_size x len_src x d_model

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        output_dict = { 'hidden': output, 'coverage': coverage }

        # return output, None
        return output_dict

    def step(self, input, decoder_state, current_step=-1):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """
        context = decoder_state.context
        buffers = decoder_state.attention_buffers
        src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None
        # input_attbs = decoder_state.tgt_attbs

        if self.fixed_target_length > 0:
            tgt_length = decoder_state.tgt_length

        if decoder_state.input_seq is None:
            decoder_state.input_seq = input
        else:
            # concatenate the last input to the previous input sequence
            decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
        input = decoder_state.input_seq.transpose(0, 1)
        input_ = input[:,-1].unsqueeze(1)

        # output_buffer = list()

        # batch_size = input_.size(0)

        """ Embedding: batch_size x 1 x d_model """
        emb = self.word_lut(input_)

        emb = emb * math.sqrt(self.model_size)

        """ Adding positional encoding """
        if self.fixed_target_length == 2 or self.fixed_target_length == 3:

            if self.fixed_target_length == 3:
                emb = self.time_transformer(emb, t=input.size(1))
                emb = emb * math.sqrt(self.model_size)

            # add target length encoding
            tgt_length = tgt_length - current_step + 1
            tgt_length = tgt_length.unsqueeze(1)
            num_timescales = self.model_size // 2
            log_timescale_increment = math.log(10000) / (num_timescales - 1)
            inv_timescales = torch.exp(torch.arange(0, num_timescales).type_as(emb) * -log_timescale_increment)
            scaled_time = tgt_length.float().unsqueeze(2) * inv_timescales.unsqueeze(0).unsqueeze(0)
            pos_emb = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), 2)
            emb = emb + pos_emb

        else:
            emb = self.time_transformer(emb, t=input.size(1))

        # """ Adding positional encoding """ Already done above
        # if self.time == 'positional_encoding':
        #     emb = emb * math.sqrt(self.model_size)
        #     emb = self.time_transformer(emb, t=input.size(1))
        # else:
            # prev_h = buffer[0] if buffer is None else None
            # emb = self.time_transformer(emb, prev_h)
            # buffer[0] = emb[1]
            # raise NotImplementedError

        if isinstance(emb, tuple):
            emb = emb[0]
        # emb should be batch_size x 1 x dim

        # Preprocess layer: adding dropout
        emb = self.preprocess_layer(emb)

        # if self.enable_feature:
        #     input_attbs = input_attbs.unsqueeze(1)
        #     attb_emb = self.feat_lut(input_attbs)
        #
        #     emb = torch.cat([emb, attb_emb], dim=-1)
        #
        #     emb = torch.relu(self.feature_projector(emb))

        if self.fixed_target_length == 1:  # int encoding?
            tgt_length = tgt_length - current_step + 1
            tgt_length = tgt_length.unsqueeze(1)
            tgt_emb = self.length_lut(tgt_length)
            emb = torch.cat([emb, tgt_emb], dim=-1)

            emb = torch.relu(self.length_projector(emb))

        emb = emb.transpose(0, 1)

        # batch_size x 1 x len_src

        if context is not None:
            if self.encoder_type == "audio" and src.data.dim() == 3:
                mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).unsqueeze(1)
            else:
                mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).byte().unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        mask_tgt = mask_tgt[:, -1, :].unsqueeze(1)

        output = emb.contiguous()

        for i, layer in enumerate(self.layer_modules):

            buffer = buffers[i] if i in buffers else None
            assert(output.size(0) == 1)

            output, coverage, buffer = layer.step(output, context, mask_tgt, mask_src, buffer=buffer) # self.parameters():

            decoder_state.update_attention_buffer(buffer, i)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        return output, coverage


class Transformer(NMTModel):
    """Main model in 'Attention is all you need' """

    def __init__(self, encoder, decoder, generator=None):
        super().__init__( encoder, decoder, generator)
        self.model_size = self.decoder.model_size

    def reset_states(self):
        return

    def forward(self, batch, **kwargs):
        """
        The forward function served in training (for back propagation)

        Inputs Shapes: 
            src: len_src x batch_size
            tgt: len_tgt x batch_size
        
        Outputs Shapes:
            out:      batch_size*len_tgt x model_size
            
            
        """
        src = batch.get('source')
        tgt = batch.get('target_input')
        src_type = batch.src_type

        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        tgt_attbs = batch.get('tgt_attbs')  # vector of length B

        encoder_output = self.encoder(src)
        context = encoder_output['context']
        
        decoder_output = self.decoder(tgt, context, src,src_type,  input_attbs=tgt_attbs)  # tgt_attbs,
        output = decoder_output['hidden']

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['encoder'] = context
        output_dict['src_mask'] = encoder_output['src_mask']

        logprobs = self.generator[0](output)
        output_dict['logprobs'] = logprobs

        return output_dict

    def decode(self, batch):
        """
        :param batch: (onmt.Dataset.Batch) an object containing tensors needed for training
        :return: gold_scores (torch.Tensor) log probs for each sentence
                 gold_words  (Int) the total number of non-padded tokens
                 allgold_scores (list of Tensors) log probs for each word in the sentence
        """

        src = batch.get('source')
        tgt_input = batch.get('target_input')
        tgt_output = batch.get('target_output')
        src_type = batch.src_type

        # transpose to have batch first
        src = src.transpose(0, 1)
        tgt_input = tgt_input.transpose(0, 1)
        batch_size = tgt_input.size(0)

        context = self.encoder(src)['context']

        if hasattr(self,'autoencoder') and self.autoencoder \
                and self.autoencoder.representation == "EncoderHiddenState":
            context = self.autoencoder.autocode(context)

        gold_scores = context.new(batch_size).zero_()
        gold_words = 0
        allgold_scores = list()

        decoder_output = self.decoder(tgt_input, context, src,src_type)['hidden']

        output = decoder_output

        if hasattr(self, 'autoencoder')  and self.autoencoder and \
                self.autoencoder.representation == "DecoderHiddenState":
            output = self.autoencoder.autocode(output)

        for dec_t, tgt_t in zip(output, tgt_output):
            if(isinstance(self.generator,nn.ModuleList)):
                gen_t = self.generator[0](dec_t)
            else:
                gen_t = self.generator(dec_t)
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_words += tgt_t.ne(onmt.Constants.PAD).sum().item()
            allgold_scores.append(scores.squeeze(1).type_as(gold_scores))

        return gold_words, gold_scores, allgold_scores

    def renew_buffer(self, new_len):
        self.decoder.renew_buffer(new_len)

    def step(self, input_t, decoder_state, current_step=-1):
        """
        Decoding function:
        generate new decoder output based on the current input and current decoder state
        the decoder state is updated in the process
        :param input_t: the input word index at time t
        :param decoder_state: object DecoderState containing the buffers required for decoding
        :return: a dictionary containing: log-prob output and the attention coverage
        """


        hidden, coverage = self.decoder.step(input_t, decoder_state, current_step)
        # squeeze to remove the time step dimension
        log_prob = self.generator[0](hidden.squeeze(0))

        last_coverage = coverage[:, -1, :].squeeze(1)

        output_dict = defaultdict(lambda: None)

        output_dict['log_prob'] = log_prob
        output_dict['coverage'] = last_coverage

        
        return output_dict

    def create_decoder_state(self, batch, beam_size=1, length_batch=None):
        """
        Generate a new decoder state based on the batch input
        :param batch: Batch object (may not contain target during decoding)
        :param beam_size: Size of beam used in beam search
        :return:
        """
        src = batch.get('source')
        # tgt_attbs = batch.get('tgt_attbs')  # vector of length B

        src_transposed = src.transpose(0, 1)
        encoder_output = self.encoder(src_transposed)

        decoder_state = TransformerDecodingState(src, encoder_output['context'],
                                                 beam_size=beam_size, model_size=self.model_size,
                                                 length_batch=length_batch)

        return decoder_state


class TransformerDecodingState(DecoderState):
    
    def __init__(self, src, context, beam_size=1, model_size=512, tgt_attbs=None, length_batch=None):

        # if audio only take one dimension since only used for mask
        self.original_src = src
        if src is not None:
            if src.dim() == 3:
                self.src = src.narrow(2, 0, 1).squeeze(2).repeat(1, beam_size)
            else:
                self.src = src.repeat(1, beam_size)
        else:
            self.src = None

        if context is not None:
            self.context = context.repeat(1, beam_size, 1)
        else:
            self.context = None

        self.beam_size = beam_size

        self.input_seq = None
        self.attention_buffers = dict()
        self.model_size = model_size
        ### added for len decoding
        # self.tgt_attbs = tgt_attbs.repeat(beam_size)  # size: Bxb
        if length_batch:
            self.use_tgt_length = True
            self.tgt_length = torch.tensor(length_batch).repeat(beam_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) #.type_as(self.tgt_attbs)
        else:
            self.use_tgt_length = False

    def update_attention_buffer(self, buffer, layer):

        self.attention_buffers[layer] = buffer  # dict of 2 keys (k, v) : T x B x H

    def update_beam(self, beam, b, remaining_sents, idx):

        for tensor in [self.src, self.input_seq]  :

            if tensor is None:
                continue

            t_, br = tensor.size()
            sent_states = tensor.view(t_, self.beam_size, remaining_sents)[:, :, idx]

            sent_states.copy_(sent_states.index_select(
                1, beam[b].getCurrentOrigin()))

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]

            if buffer_ is None:
                continue

            for k in buffer_:
                t_, br_, d_ = buffer_[k].size()
                sent_states = buffer_[k].view(t_, self.beam_size, remaining_sents, d_)[:, :, idx, :]

                sent_states.data.copy_(sent_states.data.index_select(
                            1, beam[b].getCurrentOrigin()))

        # state_ = self.tgt_attbs.view(self.beam_size, remaining_sents)[:, idx]

        # state_.copy_(state_.index_select(0, beam[b].getCurrentOrigin()))

        if self.use_tgt_length:
            state_ = self.tgt_length.view(self.beam_size, remaining_sents)[:, idx]
            state_.copy_(state_.index_select(0, beam[b].getCurrentOrigin()))

    # in this section, the sentences that are still active are
    # compacted so that the decoder is not run on completed sentences
    def prune_complete_beam(self, active_idx, remaining_sents):

        model_size = self.model_size

        def update_active(t):
            if t is None:
                return t
            # select only the remaining active sentences
            view = t.data.view(-1, remaining_sents, model_size)
            new_size = list(t.size())
            new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
            return view.index_select(1, active_idx).view(*new_size)

        def update_active_2d(t):
            if t is None:
                return t
            view = t.view(-1, remaining_sents)
            new_size = list(t.size())
            new_size[-1] = new_size[-1] * len(active_idx) // remaining_sents
            new_t = view.index_select(1, active_idx).view(*new_size)
            return new_t

        self.context = update_active(self.context)

        self.input_seq = update_active_2d(self.input_seq)

        self.src = update_active_2d(self.src)

        # self.tgt_attbs = update_active_2d(self.tgt_attbs)

        if self.use_tgt_length:
            self.tgt_length = update_active_2d(self.tgt_length)

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]

            for k in buffer_:
                buffer_[k] = update_active(buffer_[k])

