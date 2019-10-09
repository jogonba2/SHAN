from keras.layers import (Input, LSTM, TimeDistributed, \
						  Dense, RepeatVector, Permute, BatchNormalization, \
						  Lambda, GaussianNoise, Bidirectional, Flatten, \
						  Activation, Multiply, Masking, Concatenate, Subtract)
						  
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint

class SHANN:
	
	def __init__(self, max_len_doc_sents, max_len_doc_sent_words,
				 max_len_summ_sents, max_len_summ_sent_words, d,
				 path_models, name_models):
		
		self.max_len_doc_sents = max_len_doc_sents
		self.max_len_doc_sent_words = max_len_doc_sent_words
		self.max_len_summ_sents = max_len_summ_sents
		self.max_len_summ_sent_words = max_len_summ_sent_words
		self.d = d
		self.loss_train = []
		self.loss_dev = []
		self.chkpath = path_models + "/" + name_models + "-{epoch:05d}-{val_loss:.5f}.hdf5"
		self.checkpoint = ModelCheckpoint(self.chkpath, monitor='val_loss', 
						  verbose=1, save_best_only=False, 
						  mode='min')
		self.callbacks = [self.checkpoint]
		
	def get_max_len_doc_sents(self): 
		return self.max_len_doc_sents
	
	def get_max_len_doc_sent_words(self): 
		return self.max_len_doc_sent_words
	
	def get_max_len_summ_sents(self): 
		return self.max_len_summ_sents
	
	def get_max_len_summ_sent_words(self): 
		return self.max_len_summ_sent_words

	def fit_generator(self, generator_train, generator_dev, 
					  steps_per_epoch, epochs, validation_steps):
						  
		self.shann_model.fit_generator(generator_train, 
									   steps_per_epoch=steps_per_epoch,
									   epochs=epochs, 
									   validation_data=generator_dev,
									   validation_steps=validation_steps,
									   callbacks=self.callbacks, verbose = 2)
									   
	
	def predict_distance(self, x): 
		return self.shann_model.predict(x)

	def predict_attention(self, x):
                return self.all_att_model.predict(x)

	def save_model(self, name):
		model_json = self.shann_model.to_json()
		with open(name + ".json", "w") as json_file:
			json_file.write(model_json)
		shann_model.save_weights(name + ".h5")
	
	def load_model(self, f_arch_name, f_weights_name): pass
	
	def load_weights(self, fname):
		self.shann_model.load_weights(fname)
		self.shann_model.compile(optimizer="adam", loss="categorical_crossentropy")

	def get_shann_model(self): 
		return self.shann_model
	
	def get_all_att_model(self):
		return self.all_att_model

	def _set_model(self):
                inp_doc  = Input(shape=(self.max_len_doc_sents,
                                        self.max_len_doc_sent_words,
                                        self.d))
                inp_summ = Input(shape=(self.max_len_summ_sents,
                                        self.max_len_summ_sent_words,
                                        self.d))

                lstm_words  = LSTM(512, activation="tanh",
                                   return_sequences=True,
                                   dropout=0.3)

                lstm_sents  = LSTM(512, activation="tanh",
                                   return_sequences=True,
                                   dropout=0.3)

                # Word Level Doc #
                w_doc        = TimeDistributed(lstm_words)(inp_doc)
                alpha1_doc   = TimeDistributed(Dense(1))(w_doc)
                alpha1_doc   = TimeDistributed(Flatten())(alpha1_doc)
                alpha1_doc_o = TimeDistributed(Activation("softmax"))(alpha1_doc)
                alpha1_doc   = TimeDistributed(RepeatVector(512))(alpha1_doc_o)
                alpha1_doc   = TimeDistributed(Permute([2, 1]))(alpha1_doc)
                s1_doc   = Multiply()([w_doc, alpha1_doc])
                s1_doc   = TimeDistributed(Lambda(lambda x: K.sum(x, axis=-2)))(s1_doc)

                # Word Level Summ #
                w_summ    = TimeDistributed(lstm_words)(inp_summ)
                alpha1_summ   = TimeDistributed(Dense(1))(w_summ)
                alpha1_summ   = TimeDistributed(Flatten())(alpha1_summ)
                alpha1_summ = TimeDistributed(Activation("softmax"))(alpha1_summ)
                alpha1_summ   = TimeDistributed(RepeatVector(512))(alpha1_summ)
                alpha1_summ   = TimeDistributed(Permute([2, 1]))(alpha1_summ)
                s1_summ   = Multiply()([w_summ, alpha1_summ])
                s1_summ   = TimeDistributed(Lambda(lambda x: K.sum(x, axis=-2)))(s1_summ)

                # Sentence Level Doc #
                h_doc        = lstm_sents(s1_doc)
                alpha2_doc   = Dense(1)(h_doc)
                alpha2_doc   = Flatten()(alpha2_doc)
                alpha2_doc_o = Activation("softmax")(alpha2_doc)
                alpha2_doc   = RepeatVector(512)(alpha2_doc_o)
                alpha2_doc   = Permute([2, 1])(alpha2_doc)
                s2_doc    = Multiply()([h_doc, alpha2_doc])
                s2_doc    = Lambda(lambda x: K.sum(x, axis=-2))(s2_doc)

                # Sentence Level Summ #
                h_summ    = lstm_sents(s1_summ)
                alpha2_summ   = Dense(1)(h_summ)
                alpha2_summ   = Flatten()(alpha2_summ)
                alpha2_summ = Activation("softmax")(alpha2_summ)
                alpha2_summ   = RepeatVector(512)(alpha2_summ)
                alpha2_summ   = Permute([2, 1])(alpha2_summ)
                s2_summ    = Multiply()([h_summ, alpha2_summ])
                s2_summ    = Lambda(lambda x: K.sum(x, axis=-2))(s2_summ)

                diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(512,))([s2_doc, s2_summ])
                h_merged = Concatenate()([s2_doc, s2_summ, diff])
                output = Dense(2, activation="softmax")(h_merged)

                self.shann_model = Model(inputs=[inp_doc, inp_summ], outputs=output)
                self.shann_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                self.all_att_model = Model(inputs=inp_doc, outputs=[alpha1_doc_o, alpha2_doc_o])
