import torch, copy
import torch.nn as nn
from torch.autograd import Variable
import onmt
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder, Transformer, MixedEncoder
from onmt.modules.Transformer.Layers import PositionalEncoding

init = torch.nn.init

MAX_LEN = onmt.Constants.max_position_length  # This should be the longest sentence from the dataset


def build_model(opt, dicts):

    model = None
    
    if not hasattr(opt, 'model'):
        opt.model = 'recurrent'
        
    if not hasattr(opt, 'layer_norm'):
        opt.layer_norm = 'slow'
        
    if not hasattr(opt, 'attention_out'):
        opt.attention_out = 'default'
    
    if not hasattr(opt, 'residual_type'):
        opt.residual_type = 'regular'

    if not hasattr(opt, 'input_size'):
        opt.input_size = 40

    if not hasattr(opt, 'init_embedding'):
        opt.init_embedding = 'xavier'

    if not hasattr(opt, 'ctc_loss'):
        opt.ctc_loss = 0

    if not hasattr(opt, 'encoder_layers'):
        opt.encoder_layers = -1

    if not hasattr(opt, 'fusion'):
        opt.fusion = False
   
    if not hasattr(opt, 'freeze_encoder'):
       opt.freeze_encoder = False
    
    if not hasattr(opt, 'limit_rhs_steps'):
        opt.limit_rhs_steps = None
    
    if opt.limit_rhs_steps is not None:
        try:
            opt.limit_rhs_steps = int(opt.limit_rhs_steps)
        except Exception:
            raise Exception

    if not hasattr(opt, 'fixed_target_length'):
        opt.fixed_target_length = 'no'

    onmt.Constants.layer_norm = opt.layer_norm
    onmt.Constants.weight_norm = opt.weight_norm
    onmt.Constants.activation_layer = opt.activation_layer
    onmt.Constants.version = 1.0
    onmt.Constants.attention_out = opt.attention_out
    onmt.Constants.residual_type = opt.residual_type

    if not opt.fusion:
        model = build_tm_model(opt, dicts)
    else:
        model = build_fusion(opt, dicts)

    return model


def build_tm_model(opt, dicts):


    # BUILD POSITIONAL ENCODING
    if opt.time == 'positional_encoding':
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
    else:
        raise NotImplementedError

    # if dicts['atb'].size() > 0:
    #     feat_embedding = nn.Embedding(dicts['atb'].size(), opt.model_size)
    # else:
    feat_embedding = None

    # BUILD GENERATOR
    generators = [onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())]

    if opt.ctc_loss != 0:
        generators.append(onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size() + 1))
    
    if opt.model == 'transformer':
        # raise NotImplementedError

        onmt.Constants.init_value = opt.param_init

        if opt.encoder_type == "text":
            encoder = TransformerEncoder(opt, dicts['src'], positional_encoder,opt.encoder_type)
        elif opt.encoder_type == "audio":
            encoder = TransformerEncoder(opt, opt.input_size, positional_encoder,opt.encoder_type)
        elif opt.encoder_type == "mix":
            text_encoder = TransformerEncoder(opt, dicts['src'], positional_encoder, "text")
            audio_encoder = TransformerEncoder(opt, opt.input_size, positional_encoder, "audio")
            encoder = MixedEncoder(text_encoder,audio_encoder)
        else:
            print ("Unkown encoder type:",opt.encoder_type)
            exit(-1)


        decoder = TransformerDecoder(opt, dicts['tgt'], positional_encoder, feature_embedding=feat_embedding)

        model = Transformer(encoder, decoder, nn.ModuleList(generators))

    elif opt.model == 'stochastic_transformer':
        """
        The stochastic implementation of the Transformer as in 
        "Very Deep Self-Attention Networks for End-to-End Speech Recognition"
        """
        from onmt.modules.StochasticTransformer.Models import StochasticTransformerEncoder, StochasticTransformerDecoder

        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        if opt.encoder_type == "text":
            encoder = StochasticTransformerEncoder(opt, dicts['src'], positional_encoder,opt.encoder_type)
        elif opt.encoder_type == "audio":
            encoder = StochasticTransformerEncoder(opt, opt.input_size, positional_encoder,opt.encoder_type)
        elif opt.encoder_type == "mix":
            text_encoder = StochasticTransformerEncoder(opt, dicts['src'], positional_encoder, "text")
            audio_encoder = StochasticTransformerEncoder(opt, opt.input_size, positional_encoder, "audio")
            encoder = MixedEncoder(text_encoder,audio_encoder)
        else:
            print ("Unknown encoder type:",opt.encoder_type)
            exit(-1)

        decoder = StochasticTransformerDecoder(opt, dicts['tgt'], positional_encoder)

        model = Transformer(encoder, decoder, nn.ModuleList(generators))


    elif opt.model in ['universal_transformer', 'utransformer'] :

        from onmt.modules.UniversalTransformer.Models import UniversalTransformerDecoder, UniversalTransformerEncoder
        from onmt.modules.UniversalTransformer.Layers import TimeEncoding

        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init

        time_encoder = TimeEncoding(opt.model_size, len_max=32)

        encoder = UniversalTransformerEncoder(opt, dicts['src'], positional_encoder, time_encoder)
        decoder = UniversalTransformerDecoder(opt, dicts['tgt'], positional_encoder, time_encoder)

        model = Transformer(encoder, decoder, nn.ModuleList(generators))

    else:
        raise NotImplementedError

    if opt.tie_weights:  
        print("Joining the weights of decoder input and output embeddings")
        model.tie_weights()
       
    if opt.join_embedding:
        print("Joining the weights of encoder and decoder word embeddings")
        model.share_enc_dec_embedding()

    for g in model.generator:
        init.xavier_uniform_(g.linear.weight)

        

    if opt.encoder_type == "audio":
        init.xavier_uniform_(model.encoder.audio_trans.weight.data)
        if opt.init_embedding == 'xavier':
            init.xavier_uniform_(model.decoder.word_lut.weight)
        elif opt.init_embedding == 'normal':
            init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
    elif opt.encoder_type == "text":
        if opt.init_embedding == 'xavier':
            init.xavier_uniform_(model.encoder.word_lut.weight)
            init.xavier_uniform_(model.decoder.word_lut.weight)
        elif opt.init_embedding == 'normal':
            init.normal_(model.encoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
            init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
    elif opt.encoder_type == "mix":
        init.xavier_uniform_(model.encoder.audio_encoder.audio_trans.weight.data)
        if opt.init_embedding == 'xavier':
            init.xavier_uniform_(model.encoder.text_encodedr.word_lut.weight)
            init.xavier_uniform_(model.decoder.word_lut.weight)
        elif opt.init_embedding == 'normal':
            init.normal_(model.encoder.text_encoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
            init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)

    else:
        print ("Unkown encoder type:",opt.encoder_type)
        exit(-1)


    return model


def init_model_parameters(model, opt):

    # currently this function does not do anything
    # because the parameters are locally initialized
    pass


def build_language_model(opt, dicts):

    onmt.Constants.layer_norm = opt.layer_norm
    onmt.Constants.weight_norm = opt.weight_norm
    onmt.Constants.activation_layer = opt.activation_layer
    onmt.Constants.version = 1.0
    onmt.Constants.attention_out = opt.attention_out
    onmt.Constants.residual_type = opt.residual_type

    # from onmt.modules.TransformerLM.Models import TransformerLM, TransformerLMDecoder
    #
    # positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
    #
    # decoder = TransformerLMDecoder(opt, dicts['tgt'], positional_encoder)
    #
    #
    #
    # model = TransformerLM(None, decoder, )

    from onmt.modules.LSTMLM.Models import LSTMLMDecoder, LSTMLM

    decoder = LSTMLMDecoder(opt, dicts['tgt'])

    generators = [onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())]

    model = LSTMLM(None, decoder, nn.ModuleList(generators))

    if opt.tie_weights:
        print("Joining the weights of decoder input and output embeddings")
        model.tie_weights()

    for g in model.generator:
        init.xavier_uniform_(g.linear.weight)

    init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)

    return model


def build_fusion(opt, dicts):

    # the fusion model requires a pretrained language model
    print("Loading pre-trained language model from %s" % opt.lm_checkpoint)
    lm_checkpoint = torch.load(opt.lm_checkpoint, map_location=lambda storage, loc: storage)

    # first we build the lm model and lm checkpoint
    lm_opt = lm_checkpoint['opt']

    lm_model = build_language_model(lm_opt, dicts)

    # load parameter for pretrained model
    lm_model.load_state_dict(lm_checkpoint['model'])

    # main model for seq2seq (translation, asr)
    tm_model = build_tm_model(opt, dicts)

    from onmt.modules.FusionNetwork.Models import FusionNetwork
    model = FusionNetwork(tm_model, lm_model)

    return model
