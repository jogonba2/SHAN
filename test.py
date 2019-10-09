import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from shann import SHANN
from decoder import Decoder
from util import Utils as ut
from pythonrouge.pythonrouge import Pythonrouge
import numpy as np


"""
{'ROUGE-1-P': 0.33197, 'ROUGE-1': 0.39987, 'ROUGE-2-P': 0.14745, 'ROUGE-2': 0.17743, 'ROUGE-L-P': 0.30133, 'ROUGE-L': 0.36273}
./chkp_models//model_-00082-0.00764.hdf5
"""

max_len_doc_sents = 20 # avg 27 # MEJOR 10
max_len_summ_sents = 4
max_len_doc_sent_words = 25 # avg 29 # MEJOR 25
max_len_summ_sent_words = 15
padding_val = 0.
pos_pairs = 32
neg_pairs = 32
path_models = "./best_model/"
name_models = "model_"
output_file_sents = "./cnndaily_shann_3_sents.out"
output_file_words = "./cnndaily_shann_3_words.out"
path_weights = "./best_model//model_-00082-0.00764.hdf5"
w2v = ut.load_word2vec("../../Embeddings/cnndaily_w2v.model")
d = w2v.vector_size
topk_sentences = 3

shann_obj = SHANN(max_len_doc_sents, max_len_doc_sent_words,
		  max_len_summ_sents, max_len_summ_sent_words,
		  d, path_models, name_models)

shann_obj._set_model()
shann_obj.load_weights(path_weights)

decoder = Decoder(max_len_doc_sents, max_len_doc_sent_words,
                  w2v, d, shann_obj.get_all_att_model(),
                  topk_sentences=topk_sentences)

test_file = "../../Corpora/CNNDM/test.csv"

x_ts, y_ts = ut.load_csv_samples(test_file)

"""
# Word Level #
print(len(x_ts))
summaries = decoder._word_decoder(x_ts[15])
print(summaries)
print(y_ts[15])
exit()
with open(output_file_words, "w", encoding="utf8") as fw:
    for i in range(len(summaries)):
        fw.write(summaries[i].strip() + "\t" + y_ts.iloc[i].strip() + "\n")
"""

# Sentence Level #
print(len(x_ts))
summaries = decoder._sentences_decoder(x_ts)
gen_summaries = []
spl_references = []

for i in range(len(summaries)):
    gen_summaries.append(summaries[i].split("\n"))
    spl_references.append([y_ts[i].split("\n")])

rouge = Pythonrouge(summary_file_exist=False, delete_xml = True,
                    summary=gen_summaries, reference=spl_references,
                    n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                    f_measure_only=True, stemming=True, stopwords=False,
                    word_level=True, length_limit=False)

sc = rouge.calc_score()
print(sc)
