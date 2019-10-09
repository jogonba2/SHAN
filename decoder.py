from util import Utils as ut
import numpy as np

class Decoder:

    def __init__(self, max_len_doc_sents, max_len_doc_sent_words,
                 w2v, d, all_att_model, sent_separator="\n",
                 word_separator=" ", padding_val=0., topk_sentences=3,
                 topk_words=40):

        self.max_len_doc_sents = max_len_doc_sents
        self.max_len_doc_sent_words = max_len_doc_sent_words
        self.w2v = w2v
        self.d = d
        self.all_att_model = all_att_model
        self.sent_separator = sent_separator
        self.word_separator = word_separator
        self.padding_val = padding_val
        self.topk_sentences = topk_sentences
        self.topk_words = topk_words

    def _sentence_decoder(self, x):
        rx = ut.repr_as_sents(x, self.w2v, self.max_len_doc_sents,
                              self.max_len_doc_sent_words,
                              self.sent_separator,
                              self.word_separator,
                              self.padding_val)


        sx = x.split(self.sent_separator)
        lsx = len(sx)
        padding_req = max(0, self.max_len_doc_sents - lsx)
        sent_weights = self.all_att_model.predict(np.array([rx]))[1][0][padding_req:]
        relevant_weights = sorted(sent_weights.argsort()[-self.topk_sentences:][::-1])
        sent_summary = []
        for k in relevant_weights:
            sent_summary.append(sx[k].strip())
        return "\n".join(sent_summary)

    def _sentences_decoder(self, sents):
        summaries = []
        for i in range(len(sents)):
            summaries.append(self._sentence_decoder(sents[i]))
            if i%50==0: print(i)
        return summaries

    def _word_decoder(self, x):
        rx = ut.repr_as_sents(x, self.w2v, self.max_len_doc_sents,
                              self.max_len_doc_sent_words,
                              self.sent_separator,
                              self.word_separator,
                              self.padding_val)

        sents_x = x.split(self.sent_separator)
        words_per_sent_x = [l.split() for l in sents_x]
        words_x = self.word_separator.join(sents_x).split()
        lsx = len(sents_x)
        padding_sents = max(0, self.max_len_doc_sents - lsx)
        padding_words = [max(0, self.max_len_doc_sent_words - len(l)) for l in words_per_sent_x]
        sent_weights = self.all_att_model.predict(np.array([rx]))[1][0][padding_sents:]
        word_weights = self.all_att_model.predict(np.array([rx]))[0][0]
        word_weights = [word_weights[i][padding_words[i]:] for i in range(len(word_weights))]
        word_weights = [word_weights[i] * sent_weights[i] for i in range(len(sent_weights))]
        word_summary = []
        top_pos_words = ut.get_topk_maxs(word_weights, self.topk_words)
        top_pos_words = sorted(top_pos_words)
        h = {}
        for i in range(len(top_pos_words)):
            if top_pos_words[i][0] not in h:
                h[top_pos_words[i][0]] = [top_pos_words[i][1]]
            else:
                h[top_pos_words[i][0]].append(top_pos_words[i][1])

        for k in h:
            for j in h[k]:
                word_summary.append(words_per_sent_x[k][j])
            word_summary.append(".")

        return " ".join(word_summary)

    def _words_decoder(self, sents):
        summaries = []
        for i in range(len(sents)):
            summaries.append(self._word_decoder(sents[i]))
            if i%50==0: print(i)
        return summaries

