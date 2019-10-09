from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from nltk import word_tokenize
from gensim.models import Word2Vec, KeyedVectors
from random import randint
from ast import literal_eval
import numpy as np
import pandas as pd
import os
import re


class Utils:

	@staticmethod
	def load_csv_samples(csv_file):
		aux_samples = []
		for chunk in  pd.read_csv(csv_file, sep='\s*\t\s*', lineterminator="\n", chunksize=20000, engine="python"):
			aux_samples.append(chunk)

		csv_samples = pd.concat(aux_samples, axis=0)
		del aux_samples
		texts = csv_samples["TEXT"]
		summaries = csv_samples["SUMMARY"]
		return texts.apply(literal_eval), summaries.apply(literal_eval)

	@staticmethod
	def load_bin_word2vec(fname):
		#return Word2Vec.load_word2vec_format(fname, binary=True, unicode_errors='ignore')
		return KeyedVectors.load_word2vec_format(fname, binary=True, unicode_errors='ignore')

	@staticmethod
	def load_word2vec(fname):
		return Word2Vec.load(fname)

	@staticmethod
	def repr_as_sents(x, w2v, max_len_sents,
					  max_len_sent_words,
					  sent_separator="\n",
					  word_separator=" ",
					  padding_val=0.):

		repr_doc_as_sents = []
		x = x.split(sent_separator)
		lx = len(x)
		for i in range(min(lx, max_len_sents)):
			aux_sent = x[i].split(word_separator)
			l_aux_sent = len(aux_sent)
			repr_sent = []

			for j in range(min(l_aux_sent, max_len_sent_words)):
				if aux_sent[j] in w2v:
					repr_sent.append(w2v[aux_sent[j]])
				else:
					repr_sent.append(w2v["unknown"])
			repr_doc_as_sents.append(repr_sent)

		repr_doc_as_sents = pad_sequences(repr_doc_as_sents, 
										  maxlen=max_len_sent_words, 
										  value=padding_val, 
										  dtype="float32")
		
		repr_doc_as_sents = pad_sequences([repr_doc_as_sents],
										  maxlen=max_len_sents, 
										  value=padding_val, 
										  dtype="float32")[0]
		return repr_doc_as_sents
		
	@staticmethod
	def generator(x, y, max_len_doc_sents, max_len_summ_sents,
				  max_len_doc_sent_words, max_len_summ_sent_words,
                  padding_val, pos_pairs, neg_pairs, w2v, d, 
                  similar_val=0.9999, non_similar_val=0.0001):
    
		
		while 1:
			batch_x1, batch_x2, batch_y = [], [], []

			for cpos in range(pos_pairs):
				index = randint(0, len(x)-1)
				doc  = x.iloc[index]
				summ = y.iloc[index]
				if not doc or not summ:
					cpos -= 1
					continue

				doc_sents = Utils.repr_as_sents(doc, w2v, max_len_doc_sents,
										        max_len_doc_sent_words)

				summ_sents = Utils.repr_as_sents(summ, w2v, max_len_summ_sents,
										         max_len_summ_sent_words)
				batch_x1.append(doc_sents)
				batch_x2.append(summ_sents)
				batch_y.append([0, 1])
			
			for cneg in range(neg_pairs):
				doc_index  = randint(0, len(x)-1)
				summ_index = randint(0, len(y)-1)
				while doc_index==summ_index: summ_index = randint(0, len(y) - 1) 
				doc  = x.iloc[doc_index]
				summ = y.iloc[summ_index]
				if not doc or not summ:
                                    cneg -= 1
                                    continue

				doc_sents = Utils.repr_as_sents(doc, w2v, max_len_doc_sents,
										        max_len_doc_sent_words)

				summ_sents = Utils.repr_as_sents(summ, w2v, max_len_summ_sents,
										         max_len_summ_sent_words)

				batch_x1.append(doc_sents)
				batch_x2.append(summ_sents)
				batch_y.append([1, 0])
			
			batch_x1 = np.array(batch_x1)
			batch_x2 = np.array(batch_x2)
			batch_y  = np.array(batch_y)
			
			yield [batch_x1, batch_x2], batch_y
	

	@staticmethod
	def amin(x):
		pmin, vmin = 0, float("inf")
		for i in range(len(x)): 
			if x[i] < vmin:
				vmin = x[i]
				pmin = i
		return pmin, vmin

	@staticmethod
	def get_topk_maxs(x, topk):
            top_probs_list = []
            top_pos_list = []
            for i in range(len(x)):
                for j in range(len(x[i])):
                    if len(top_probs_list) < topk:
                        top_probs_list.append(x[i][j])
                        top_pos_list.append((i, j))
                    else:
                        pmin, vmin = Utils.amin(top_probs_list)
                        if x[i][j] > vmin:
                            top_probs_list[pmin] = x[i][j]
                            top_pos_list[pmin] = (i, j)
            return top_pos_list

	@staticmethod
	def generator_2(x, y, max_len_doc_sents, max_len_summ_sents,
				  max_len_doc_sent_words, max_len_summ_sent_words,
				  padding_val, pos_pairs, neg_pairs, w2v, d,
				  similar_val=0.9999, non_similar_val=0.0001):


		while 1:
			batch_x1, batch_x2, batch_y = [], [], []
			indexes = [0 for i in range(pos_pairs)]

			for cpos in range(pos_pairs):
				index = randint(0, len(x)-1)
				indexes[cpos] = index
				doc  = x.iloc[index]
				summ = y.iloc[index]
				if not doc or not summ:
					cpos -= 1
					continue

				doc_sents = Utils.repr_as_sents(doc, w2v, max_len_doc_sents,
												max_len_doc_sent_words)

				summ_sents = Utils.repr_as_sents(summ, w2v, max_len_summ_sents,
												 max_len_summ_sent_words)
				batch_x1.append(doc_sents)
				batch_x2.append(summ_sents)
				batch_y.append([0, 1])

			for cneg in range(neg_pairs):
				doc_index  = indexes[cneg]
				summ_index = randint(0, len(y)-1)
				while doc_index==summ_index: summ_index = randint(0, len(y) - 1)
				doc  = x.iloc[doc_index]
				summ = y.iloc[summ_index]
				if not doc or not summ:
					cneg -= 1
					continue

				doc_sents = Utils.repr_as_sents(doc, w2v, max_len_doc_sents,
												max_len_doc_sent_words)
				summ_sents = Utils.repr_as_sents(summ, w2v, max_len_summ_sents,
												 max_len_summ_sent_words)

				batch_x1.append(doc_sents)
				batch_x2.append(summ_sents)
				batch_y.append([1, 0])

			batch_x1 = np.array(batch_x1)
			batch_x2 = np.array(batch_x2)
			batch_y  = np.array(batch_y)

			yield [batch_x1, batch_x2], batch_y




	@staticmethod
	def generator_triplet(x, y, max_len_doc_sents, max_len_summ_sents,
	                          max_len_doc_sent_words, max_len_summ_sent_words,
	                          padding_val, batch_size, w2v, d, model,
	                          similar_val=0.9999, non_similar_val=0.0001):

		while 1:
			batch_x1, batch_x2, batch_x3, batch_y = [], [], [], []

			for i in range(batch_size):
				doc_index = randint(0, len(x) - 1)
				summ_good_index = doc_index
				summ_neg_index = randint(0, len(y) - 1)
				while doc_index == summ_neg_index: summ_neg_index = randint(0, len(y) - 1)
				doc = x.iloc[doc_index]
				summ_good = y.iloc[summ_good_index]
				summ_neg = y.iloc[summ_neg_index]

				doc_sents = Utils.repr_as_sents(doc, w2v, max_len_doc_sents, max_len_doc_sent_words)
				summ_good_sents = Utils.repr_as_sents(summ_good, w2v, max_len_summ_sents, max_len_summ_sent_words)
				summ_neg_sents = Utils.repr_as_sents(summ_neg, w2v, max_len_summ_sents, max_len_summ_sent_words)

				batch_x1.append(doc_sents)
				batch_x2.append(summ_good_sents)
				batch_x3.append(summ_neg_sents)
				batch_y.append(np.ones(1536,))

			batch_x1 = np.array(batch_x1)
			batch_x2 = np.array(batch_x2)
			batch_x3 = np.array(batch_x3)
			batch_y  = np.array(batch_y)

#			y_preds = model.predict_distance(batch_x1, batch_x2, batch_x3)
#			print(model.triplet_loss_gen(y_preds))
#			exit()
			yield [batch_x1, batch_x2, batch_x3], batch_y

