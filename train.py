from util import Utils as ut
from shann import SHANN

max_len_doc_sents = 20 # avg 29
max_len_summ_sents = 4
max_len_doc_sent_words = 25 # avg 28
max_len_summ_sent_words = 15
padding_val = 0.
pos_pairs = 32
neg_pairs = 32
path_models = "./chkp_models/"
name_models = "model_"
w2v = ut.load_word2vec("../../Embeddings/cnndaily_w2v.model")
d = w2v.vector_size
similar_val = 0.9999
non_similar_val = 0.0001
steps_per_epoch = 500 # 196961 muestras
epochs = 200
validation_steps = 150

shann_obj = SHANN(max_len_doc_sents, max_len_doc_sent_words,
				  max_len_summ_sents, max_len_summ_sent_words,
				  d, path_models, name_models)

shann_obj._set_model()
train_file = "../../Corpora/CNNDM/dev.csv"
dev_file = "../../Corpora/CNNDM/dev.csv"

x_tr, y_tr = ut.load_csv_samples(train_file)
x_dv, y_dv = ut.load_csv_samples(dev_file)


generator_train = ut.generator_2(x_tr, y_tr, max_len_doc_sents, max_len_summ_sents,
							   max_len_doc_sent_words, max_len_summ_sent_words,
							   padding_val, pos_pairs, neg_pairs, w2v, d, 
							   similar_val=similar_val, non_similar_val=non_similar_val)

generator_dev = ut.generator_2(x_dv, y_dv, max_len_doc_sents, max_len_summ_sents,
							 max_len_doc_sent_words, max_len_summ_sent_words,
							 padding_val, pos_pairs, neg_pairs, w2v, d, 
							 similar_val=similar_val, non_similar_val=non_similar_val)


shann_obj.fit_generator(generator_train, generator_dev, 
			steps_per_epoch, epochs, validation_steps)
