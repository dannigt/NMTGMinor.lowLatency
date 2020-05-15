from __future__ import division
import torch
import onmt

"""
 Class for managing the internals of the beam search process.


         hyp1-hyp1---hyp1 -hyp1
                 \             /
         hyp2 \-hyp2 /-hyp2hyp2
                               /      \
         hyp3-hyp3---hyp3 -hyp3
         ========================

 Takes care of beams, back pointers, and scores.
"""


class Beam(object):
    def __init__(self, size, cuda=False, prefix=None, prefix_score=None):

        self.size = size
        self.done = False

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(onmt.Constants.PAD)]
        self.nextYs[0][0] = onmt.Constants.BOS

        # The attentions (matrix) for each time.
        self.attn = []

        # self.commit_depth = -1

        if prefix is not None:
            # initializing beam w/ prefix

            if prefix_score is None:
                raise ValueError('Prefix given without scores!')

            # print('trying to give prefix {0} with score {1}'.format(prefix, prefix_score))
            # prefix_scores_no_eos = prefix_score[:-1]
            # prefix_scores_no_eos.insert(0, torch.tensor([0]))  # WHY INSERT 0 HERE?

            # assign score, allScores
            # first row
            self.allScores.append(self.tt.FloatTensor(size).fill_(onmt.Constants.PAD))
            for i, item in enumerate(prefix):
                # assign nextYs
                y = self.tt.LongTensor(size).fill_(onmt.Constants.PAD)
                y[0] = item
                self.nextYs.append(y)

                # assign prevKs (always 0-th element)
                self.prevKs.append(torch.tensor(0).repeat(self.size))

                s = self.tt.FloatTensor(size).fill_(onmt.Constants.PAD)
                s[0] = sum(prefix_score[:i+1])  # prefix_scores_no_eos[i]
                self.allScores.append(s)
                # print('~~~~~~~~~~~~~~~appended prefix,', i, s)

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def get_all_states(self):
        return self.nextYs

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut, start_from_prefix=False):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)
        # wordLk, (beamsize, 5008)
        # Sum the previous scores.
        if len(self.prevKs) > 0 and not start_from_prefix:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)  # expand dim 1, becomes (beam size, 1)
        else:  # beginning of sentence
            beamLk = wordLk[0]

        flatBeamLk = beamLk.view(-1)  # beam size * vocab size

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.allScores.append(self.scores)
        # print('appended', self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords  # divide to get beam ID
        self.prevKs.append(prevK)
        self.nextYs.append(bestScoresId - prevK * numWords)
        self.attn.append(attnOut.index_select(0, prevK))

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == onmt.Constants.EOS:
            self.done = True
            self.allScores.append(self.scores)

        return self.done

    def commit(self, buffer=2):
        current_depth = len(self.allScores) - 1

        # print(len(self.allScores), len(self.prevKs), len(self.nextYs))

        if current_depth >= buffer:
            # commit to current best path, which has been spitted out
            self.allScores = [self.allScores[i][0].repeat(self.size) for i in range(len(self.allScores))]
            self.prevKs = [self.prevKs[i][0].repeat(self.size) for i in range(len(self.prevKs))]
            self.nextYs = [self.nextYs[i][0].repeat(self.size) for i in range(len(self.nextYs))]  # -2 to remove EOS and punctuation
            # self.done = False
            # store commit depth
            # self.commit_depth = current_depth
            # print('Committed at depth {0}, Ys: {1}, Ks: {2}, scores: {3}'.format(self.commit_depth, len(self.nextYs), len(self.prevKs), len(self.allScores)))

    def sortBest(self):
        return torch.sort(self.scores, 0, True)

    def getBest(self):
        "Get the score of the best in the beam."
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    def getHyp(self, k, return_att=True):
        """
        Walk back to construct the full hypothesis.

        Parameters.

             * `k` - the position in the beam to construct.

         Returns.

            1. The hypothesis
            2. The attention at each time step.
        """
        hyp, attn = [], []
        lengths = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            if return_att:
                attn.append(self.attn[j][k])
            k = self.prevKs[j][k]
        
        length = len(hyp)

        if return_att:
            return hyp[::-1], torch.stack(attn[::-1]), length
        else:
            return hyp[::-1], None, length

    def advanceEOS(self, wordLk, attnOut):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search. Are are at the desired length -> select eos

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
        else:
            beamLk = wordLk[0]

        flatBeamLk = beamLk.view(-1)

        bestScoresId = torch.arange(0,flatBeamLk.size()[0],numWords) + onmt.Constants.EOS
        bestScoresId = bestScoresId.type_as(flatBeamLk).long()
        bestScores = torch.index_select(flatBeamLk,0,bestScoresId)

        #self.allScores.append(self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append(bestScoresId - prevK * numWords)
        self.attn.append(attnOut.index_select(0, prevK))

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == onmt.Constants.EOS:
            self.done = True
            self.allScores.append(self.scores)

        return self.done
