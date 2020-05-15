#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import onmt
import onmt.Markdown
import torch
import argparse
import math
import numpy
import sys
import h5py as h5
import numpy as np
import os
import pickle
import warnings
import string

parser = argparse.ArgumentParser(description='translate.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-lm', required=False,
                    help='Path to language model .pt file. Used for cold fusion')
parser.add_argument('-autoencoder', required=False,
                    help='Path to autoencoder .pt file')
parser.add_argument('-input_type', default="word",
                    help="Input type: word/char")
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-src_img_dir',   default="",
                    help='Source image directory')
parser.add_argument('-stride', type=int, default=1,
                    help="Stride on input features")
parser.add_argument('-concat', type=int, default=1,
                    help="Concate sequential audio features to decrease sequence length")
parser.add_argument('-asr_format', default="h5", required=False,
                    help="Format of asr data h5 or scp")
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img|audio].")
parser.add_argument('-previous_context', type=int, default=0,
                    help="Number of previous sentence for context")

parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=2048,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
parser.add_argument('-start_with_bos', action="store_true",
                    help="""Add BOS token to the top of the source sentence""")
# parser.add_argument('-phrase_table',
#                     help="""Path to source-target dictionary to replace UNK
#                     tokens. See README.md for the format of this file.""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')

parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('-alpha', type=float, default=0.6,
                    help="""Length Penalty coefficient""")
parser.add_argument('-beta', type=float, default=0.0,
                    help="""Coverage penalty coefficient""")
parser.add_argument('-print_nbest', action='store_true',
                    help='Output the n-best list instead of a single sentence')
parser.add_argument('-ensemble_op', default='mean', help="""Ensembling operator""")
parser.add_argument('-normalize', action='store_true',
                    help='To normalize the scores based on output length')
parser.add_argument('-fp16', action='store_true',
                    help='To use floating point 16 in decoding')
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")
parser.add_argument('-limit_rhs_steps', type=str, default=None,
                    help='How many RHS steps are visible in encoder. Default is to use full RHS context. )')

parser.add_argument('-start_prefixing_from', type=int, default=0,   # if > 0, do incremetal decoding
                    help='If non-zero, condition with prefix. From which partial input onwards to apply prefix')
parser.add_argument('-require_prefix_agreements', type=int, default=0,   # agreement
                    help='If non-zero, only output tokens that are agreed upon by the previous n partial utterances.')
parser.add_argument('-remove_last_n', type=int, default=0,
                    help='If non-zero, remove the last n tokens of partial hypotheses.')
parser.add_argument('-max_out_per_segment', type=int, default=0,
                    help='If non-zero, transcribe at a fixed rate, by maximally allowing n tokens per segment.')
parser.add_argument('-min_confidence', type=float, default=0,
                    help='If non-zero, only output tokens with likelihood > than threshold.')
parser.add_argument('-confidence_mode', type=int, default=0,
                    help='1: avg, 2: min, 3: last, 4: delete till last element > min_conf')
parser.add_argument('-wait_if_worse', action='store_true',
                    help='Wait if worse.')

parser.add_argument('-output_latency', default='latency.txt',
                    help='Path to output latency info.')
parser.add_argument('-output_confidence', default='confidence.txt',
                    help='Path to output confidence scores.')

parser.add_argument('-tgt_length',
                    help='Wanted length of the output (optional)')
parser.add_argument('-force_target_length', action="store_true",
                    help='Force output target length as in the length file')



def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))


def addone(f):
    for line in f:
        yield line
    yield None


def lenPenalty(s, l, alpha):
    
    l_term = math.pow(l, alpha)
    return s / l_term


def get_sentence_from_tokens(tokens, input_type):

    if input_type == 'word':
        sent = " ".join(tokens)
    elif input_type == 'char':
        sent = "".join(tokens)
    else:
        raise NotImplementedError
    return sent


def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    
    # Always pick n_best
    opt.n_best = opt.beam_size

    if opt.output == "stdout":
        outF = sys.stdout  # out file
    else:
        outF = open(opt.output, 'w')

    if opt.start_prefixing_from:
        try:
            # data_dir = os.path.dirname(opt.src)
            with open('.'.join(opt.src.split('.')[:-1]) + '.num.partial.seqs.pickle', 'rb') as f:
                num_partial_seqs = pickle.load(f)
        except:
            raise Exception('Failed to open partial sequence counter file.')

    # when using prefix, also write out latency report
    out_latency_f = open(opt.output_latency, 'w')
    out_confidence_f = open(opt.output_confidence, 'w')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, tgtBatch, tgt_length_batch = [], [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None
    tgt_lengthF = open(opt.tgt_length) if opt.tgt_length else None

    # here we are trying to
    inFile = None

    if opt.start_prefixing_from:
        translator = onmt.EnsembleTranslatorOnlineSim(opt)
    else:
        translator = onmt.EnsembleTranslator(opt)
        warnings.warn('Not doing online decoding!')

    if opt.dump_beam != "":
        import json
        translator.init_beam_accum()

    if opt.src == "stdin":
        inFile = sys.stdin
        opt.batch_size = 1
    elif opt.encoder_type == "audio" and opt.asr_format == "h5":
        inFile = h5.File(opt.src,'r')
    elif opt.encoder_type == "audio" and opt.asr_format == "scp":
        import kaldiio
        from kaldiio import ReadHelper
        audio_data = iter(ReadHelper('scp:' + opt.src))
    else:
      inFile = open(opt.src)

    if opt.encoder_type == "audio":

        s_prev_context = []
        t_prev_context = []

        i = 0

        latency = []
        latency_no_punct = []
        partial_hyps = []
        partial_scores = []  # this will hold likelihood of each token
        while True:  # keep reading from scp for new utterances
            if opt.asr_format == "h5":
                if i == len(inFile):
                    break
                line = np.array(inFile[str(i)])
                i += 1
            elif opt.asr_format == "scp":
                try:
                    utt_id, line = next(audio_data)
                except StopIteration:
                    break

            if opt.stride != 1:
                line = line[0::opt.stride]
            
            # line = line[:, :40]
            line = line[:, :40]
            
            line = torch.from_numpy(line)

            if opt.concat != 1:
                add = (opt.concat-line.size()[0] % opt.concat) % opt.concat
                z = torch.FloatTensor(add, line.size()[1]).zero_()
                line = torch.cat((line,z),0)
                line = line.reshape((line.size()[0]//opt.concat,line.size()[1]*opt.concat))

            #~ srcTokens = line.split()
            if opt.previous_context > 0:
                s_prev_context.append(line)
                for i in range(1, opt.previous_context+1):
                    if i < len(s_prev_context):
                        line = torch.cat((torch.cat((s_prev_context[-i-1], torch.zeros(1, line.size()[1]))), line))
                if len(s_prev_context) > opt.previous_context:
                    s_prev_context = s_prev_context[-1 * opt.previous_context:]
            srcBatch += [line]  # make batch

            if tgtF:
                #~ tgtTokens = tgtF.readline().split() if tgtF else None
                tline = tgtF.readline().strip()
                if opt.previous_context > 0:
                    t_prev_context.append(tline)
                    for i in range(1,opt.previous_context+1):
                        if i < len(s_prev_context):
                            tline = t_prev_context[-i-1]+" # "+tline
                    if len(t_prev_context) > opt.previous_context:
                        t_prev_context = t_prev_context[-1*opt.previous_context:]

                if opt.input_type == 'word':
                    tgtTokens = tline.split() if tgtF else None
                elif opt.input_type == 'char':
                    tgtTokens = list(tline.strip()) if tgtF else None
                else:
                    raise NotImplementedError("Input type unknown")

                tgtBatch += [tgtTokens]

            if tgt_lengthF:
                tgt_length_int = int(tgt_lengthF.readline())
                tgt_length_batch = [tgt_length_int]

            if len(srcBatch) < opt.batch_size:
                continue  # fetch next instance

            if opt.start_prefixing_from:  # online decoding
                # if I should use prefix
                partial_utt_idx = int(utt_id.split('_')[-1])
                orig_utt_idx = '_'.join(utt_id.split('_')[:-1])
                is_full_utt = (partial_utt_idx == (num_partial_seqs[orig_utt_idx] - 1))
            else:
                is_full_utt = True  # always write out

            # modify tgtBatch, if needed
            if opt.start_prefixing_from:
                # modify prefix based on agreement
                if opt.require_prefix_agreements:  # require prefix agree
                    if len(partial_hyps) >= 2:
                        idx_agree = 0
                        for idx_agree in range(min(len(partial_hyps[-2]), len(partial_hyps[-1]))):
                            if partial_hyps[-2][idx_agree] != partial_hyps[-1][idx_agree]:
                                # print('tokens disagreement, stopped at ', idx_agree)
                                idx_agree -= 1
                                break
                    if partial_utt_idx >= opt.start_prefixing_from:
                        tok_removed = len(partial_hyps[-1]) - (idx_agree + 1)
                        tgtBatch += [partial_hyps[-1][:idx_agree+1]]
                        partial_scores = partial_scores[:len(partial_scores) - tok_removed]
                        latency.append(len(tgtBatch[0]))
                        tok_no_punct = [tok for tok in tgtBatch[0] if not all(c in string.punctuation for c in tok)]
                        latency_no_punct.append(len(tok_no_punct))
                    else:
                        latency.append(0)
                        latency_no_punct.append(0)

                else:  # normal prefix
                    if partial_utt_idx >= opt.start_prefixing_from:
                        # prev_len = latency[-1]  # should not cut further than previous prefix
                        # actual_prefix_len = max(prev_len, len(partial_hyps[-1]) - opt.remove_last_n)

                        if not opt.wait_if_worse or partial_scores[-1] > partial_scores[-2]:
                            tgtBatch += [partial_hyps[-1]]

                        else:
                            print('waiting, {0} < {1}'.format(partial_scores[-1], partial_scores[-2]))
                            del partial_scores[-1]  #= partial_scores[-2]  # overwrite score
                            del partial_hyps[-1]  #= partial_hyps[-2]
                            tgtBatch += [partial_hyps[-1]]  # find index of closet partial_hyp that is les than me

                            # latency.append(latency[-1])  # no output
                        latency.append(len(tgtBatch[0]))
                        tok_no_punct = [tok for tok in tgtBatch[0] if not all(c in string.punctuation for c in tok)]
                        latency_no_punct.append(len(tok_no_punct))
                    else:
                        latency.append(0)
                        latency_no_punct.append(0)

            print("Batch size:", len(srcBatch), len(tgtBatch))
            predBatch, predScore, predLength, goldScore, numGoldWords, allGoldScores, all_lk = translator.translate_asr(srcBatch, tgtBatch, tgt_length_batch)
            print("Result:", len(predBatch))

            count,predScore,predWords,goldScore,goldWords,reordered_pred_words, best_conf = translateBatch(opt,tgtF,count,outF,translator,srcBatch,tgtBatch,predBatch, predScore, predLength, goldScore, numGoldWords, allGoldScores,opt.input_type,
                                                                                            all_lk, write_out=is_full_utt)
            _partial_hyp = reordered_pred_words[0][0]

            # best_lk = []
            # for my_lk in reordered_all_lk:
            #     best_lk += my_lk[0]
            #     print(my_lk[0])

            if is_full_utt:  # full seq
                latency.append(len(_partial_hyp))
                latency_no_punct.append(len([tok for tok in _partial_hyp
                                             if not all(c in string.punctuation for c in tok)]))

            else:
                # strip until last token of partial seq is not only punctuation
                while _partial_hyp and all(c in string.punctuation for c in _partial_hyp[-1]):
                    # print(_partial_hyp)
                    # print('---------------removing', _partial_hyp[-1])
                    del _partial_hyp[-1]
                    # print(_partial_hyp)
                    if best_conf:  # when continuously deleting punctuations
                        del best_conf[-1]
                    else:
                        del partial_scores[-1]

                if opt.max_out_per_segment:
                    allowed_len = latency[-1] + opt.max_out_per_segment
                    # print('-----------------', allowed_len)
                    _partial_hyp = _partial_hyp[:allowed_len]
                    best_conf = best_conf[:opt.max_out_per_segment]

                if opt.remove_last_n:
                    prev_len = latency[-1]
                    actual_removed = min(opt.remove_last_n, len(_partial_hyp) - prev_len)
                    # actual_prefix_len = max(prev_len, len(_partial_hyp) - opt.remove_last_n)
                    # _partial_hyp = _partial_hyp[:actual_prefix_len]
                    _partial_hyp = _partial_hyp[:(len(_partial_hyp)-actual_removed)]
                    # print(_partial_hyp)
                    best_conf = best_conf[:(len(best_conf)-actual_removed)]

                if opt.confidence_mode and best_conf:  # new output non empty
                    if opt.confidence_mode == 4:
                        i = -1
                        for i in range(len(best_conf) - 1, -1, -1):
                            if best_conf[i] > opt.min_confidence:
                                print(best_conf[i], '>', opt.min_confidence, 'breaking at', i)
                                i += 1
                                break

                        _partial_hyp = _partial_hyp[:latency[-1] + i]
                        best_conf = best_conf[:i]

                    # elif opt.confidence_mode == 1:
                    #     conf_val = sum(best_conf) / len(best_conf)
                    # elif opt.confidence_mode == 2:
                    #     conf_val = min(best_conf)
                    # elif opt.confidence_mode == 3:
                    #     conf_val = best_conf[-1]
                    else:
                        raise ValueError('Invalid confidence_mode {0}'.format(opt.confidence_mode))

                    # if conf_val < opt.min_confidence:
                    #     _partial_hyp = _partial_hyp[:latency[-1]]
                    #     best_conf = []

            partial_hyps.append(_partial_hyp)
            if is_full_utt or partial_utt_idx+1 >= opt.start_prefixing_from:
                partial_scores.extend(best_conf)
                #print('==============extended', len(best_conf))
                print('best conf', best_conf)

            predScoreTotal += predScore
            predWordsTotal += predWords
            goldScoreTotal += goldScore
            goldWordsTotal += goldWords
            srcBatch, tgtBatch, tgt_length_int = [], [], []

            if is_full_utt:  # don't apply prefix anymore. Clear previous ones
                if latency[-1] != len(partial_scores):
                    raise ValueError('{0} tokens vs {1} confidence scores!!'.format(latency[-1], len(partial_scores)))
                out_latency_f.write(','.join([str(x) for x in latency]) + '\n')
                out_latency_f.write(','.join([str(x) for x in latency_no_punct]) + '\n')
                out_latency_f.flush()
                out_confidence_f.write(','.join(['{0:.2f}'.format(x) for x in partial_scores]) + '\n')
                out_confidence_f.flush()
                partial_hyps = []
                partial_scores = []
                latency = []
                latency_no_punct = []

        # after all utterances are done.

        if len(srcBatch) != 0:
            print("Batch size:", len(srcBatch), len(tgtBatch))
            predBatch, predScore, predLength, goldScore, numGoldWords,allGoldScores, all_lk = translator.translate_asr(srcBatch,
                                                                                    tgtBatch)
            print("Result:", len(predBatch))

            count,predScore,predWords,goldScore,goldWords,reordered_pred_words, best_conf = translateBatch(opt,tgtF,count,outF,translator,srcBatch,tgtBatch,predBatch, predScore, predLength, goldScore, numGoldWords,allGoldScores,opt.input_type,
                                                                           all_lk, write_out=is_full_utt)
            predScoreTotal += predScore
            predWordsTotal += predWords
            goldScoreTotal += goldScore
            goldWordsTotal += goldWords
            srcBatch, tgtBatch = [], []


    else:
        
        for line in addone(inFile):
            if line is not None:
                if opt.input_type == 'word':
                    srcTokens = line.split()
                elif opt.input_type == 'char':
                    srcTokens = list(line.strip())
                else:
                    raise NotImplementedError("Input type unknown")
                srcBatch += [srcTokens]
                if tgtF:
                    #~ tgtTokens = tgtF.readline().split() if tgtF else None
                    if opt.input_type == 'word':
                        tgtTokens = tgtF.readline().split() if tgtF else None
                    elif opt.input_type == 'char':
                        tgtTokens = list(tgtF.readline().strip()) if tgtF else None
                    else:
                        raise NotImplementedError("Input type unknown")
                    tgtBatch += [tgtTokens]

                if len(srcBatch) < opt.batch_size:
                    continue
            else:
                # at the end of file, check last batch
                if len(srcBatch) == 0:
                    break

            # actually done beam search from the model
            predBatch, predScore, predLength, goldScore, numGoldWords,allGoldScores, all_lk = translator.translate(srcBatch,
                                                                                    tgtBatch)

            # convert output tensor to words
            count,predScore,predWords,goldScore,goldWords = translateBatch(opt,tgtF,count,outF,translator,
                                                                           srcBatch,tgtBatch,
                                                                           predBatch, predScore, predLength,
                                                                           goldScore, numGoldWords,
                                                                           allGoldScores,opt.input_type, all_lk)
            predScoreTotal += predScore
            predWordsTotal += predWords
            goldScoreTotal += goldScore
            goldWordsTotal += goldWords
            srcBatch, tgtBatch = [], []

    if opt.verbose:
        reportScore('PRED', predScoreTotal, predWordsTotal)
        if tgtF: reportScore('GOLD', goldScoreTotal, goldWordsTotal)


    if tgtF:
        tgtF.close()

    if out_latency_f:
        out_latency_f.close()

    if out_confidence_f:
        out_confidence_f.close()

    if opt.dump_beam:
        json.dump(translator.beam_accum, open(opt.dump_beam, 'w'))


def translateBatch(opt,tgtF,count,outF,translator,srcBatch,tgtBatch,predBatch, predScore, predLength, goldScore, numGoldWords,allGoldScores,input_type,all_lk, write_out=True):

    if opt.normalize:
        predBatch_ = []
        predScore_ = []
        all_lk_ = []
        for bb, ss, ll, llkk in zip(predBatch, predScore, predLength, all_lk):  # normalize scores within beam, for everyone in batch
            #~ ss_ = [s_/numpy.maximum(1.,len(b_)) for b_,s_,l_ in zip(bb,ss,ll)]
            length = [len(i) for i in bb] #[len(i) for i in [''.join(b_)  for b_ in bb]]
            ss_ = [lenPenalty(s_, max(l_,1), opt.alpha) for b_,s_,l_ in zip(bb,ss,length)]
            ss_origin = [(s_, len(b_)) for b_,s_,l_ in zip(bb,ss,ll)]
            sidx = numpy.argsort(ss_)[::-1]
            #~ print(ss_, sidx, ss_origin)
            predBatch_.append([bb[s] for s in sidx])
            predScore_.append([ss_[s] for s in sidx])
            all_lk_.append([[line[s] for s in sidx] for line in llkk])
        predBatch = predBatch_
        predScore = predScore_
        all_lk = all_lk_

    # print('\n'.join([str(stuff) for stuff in all_lk_]))
    predScoreTotal = sum(score[0].item() for score in predScore)
    predWordsTotal = sum(len(x[0]) for x in predBatch)
    goldScoreTotal = 0
    goldWordsTotal = 0
    if tgtF is not None:
        goldScoreTotal = sum(goldScore).item()
        goldWordsTotal = numGoldWords
            
    for b in range(len(predBatch)):
                        
        count += 1

        if (not opt.print_nbest) and write_out:
            outF.write(get_sentence_from_tokens(predBatch[b][0], input_type) + '\n')
            outF.flush()

        if opt.verbose:
            print('PRED %d: %s' % (count, get_sentence_from_tokens(predBatch[b][0], input_type)))
            best_lk = torch.tensor([lk[0] for lk in all_lk[b]])
            best_lk = torch.exp(best_lk[1:] - best_lk[:-1])
            num_out_tokens = len(predBatch[b][0]) - max(0, numGoldWords-1)
            best_conf = best_lk[:num_out_tokens].tolist()  # take actual length, exclude EOS
            print("PRED SCORE: %.4f" % predScore[b][0])

            if tgtF is not None:
                tgtSent = get_sentence_from_tokens(tgtBatch[b], input_type)
                if translator.tgt_dict.lower:
                    tgtSent = tgtSent.lower()
                print('GOLD %d: %s ' % (count, tgtSent))
                print("GOLD SCORE: %.4f" % goldScore[b])
                print("Single GOLD Scores:",end=" ")
                for j in range(len(tgtBatch[b])):
                    print(allGoldScores[j][b].item(),end =" ")
                print()
            if opt.print_nbest:
                print('\n BEST HYP:')
                for n in range(opt.n_best):
                    idx = n
                    print("[%.4f] %s" % (predScore[b][idx],
                                         " ".join(predBatch[b][idx])))

            print('')

    return count,predScoreTotal,predWordsTotal,goldScoreTotal,goldWordsTotal, predBatch, best_conf
    # also return normalization-reordered pred and likelihood
    

if __name__ == "__main__":
    main()

