#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Embedding

# Start: Word based approach

# For the word based approach more preprocessing is necessary, because only words and no special characters like '()' or '%' are allowed in the stings to avoid different perception of the same word, because of neighboring special characters.\
# For this a list of all characters is created.

# In[20]:


data = pd.read_csv("deu_prep.txt", sep=";")
data.head()


# In[21]:


all_characters = set()
for seq in data.Eng:
    for chr in seq:
        all_characters.add(chr)
for seq in data.Deu:
    for chr in seq:
        all_characters.add(chr)
print(all_characters)


# Legal characters are filtered out.

# In[22]:


import string
spechial_characters = all_characters - set(string.ascii_letters)
spechial_characters -= set([str(x) for x in range(10)])
spechial_characters -= set(['ä','Ä','ö','Ö','ü','Ü','ß',' ','\n','\t'])
print(spechial_characters)
print(string.punctuation)


# Only there more illegal letters than in the 'sting.punctuation' set are discoverd

# In[23]:


print(spechial_characters-set(string.punctuation))


# In[24]:


strip = set(string.punctuation).union({'’', '“', '„'})


# All characters in 'strip' are removed from the Stings

# In[25]:


eng = []
deu = []
for seq in data.Eng:
    sentence = ""
    for chr in seq:
        if chr not in strip:
            sentence += chr
    words = map(lambda x:x.lower(), sentence.split(" "))
    eng.append(list(words))
for seq in data.Deu:
    sentence = ""
    for chr in seq:
        if chr not in strip:
            sentence += chr
    words = map(lambda x:x.lower(), sentence.split(" "))
    deu.append(list(words))
print(eng[:10])


# eng and deu are lists containing the lists of words of each sentence.

# In[26]:


eng_max = max(len(sentence) for sentence in eng)
deu_max = max(len(sentence) for sentence in deu)


# To turn the cleaned stings into vectors the 'Tokenizer' module of keras is used.

# In[27]:


eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(eng)
eng_word_index = eng_tokenizer.word_index

deu_tokenizer = Tokenizer()
deu_tokenizer.fit_on_texts(deu)
deu_word_index = deu_tokenizer.word_index


# To make sure all vectors have the same length, a padding is created.

# In[28]:


eng_sequences = eng_tokenizer.texts_to_sequences(eng)
eng_padded = pad_sequences(eng_sequences, maxlen=eng_max, padding="post")

deu_sequences = deu_tokenizer.texts_to_sequences(deu)
deu_padded = pad_sequences(deu_sequences, maxlen=deu_max, padding="post")


# Three vectors are created: \
# 'encoder_input_data' represents the input sentence as a word vector \
# 'decoder_target_data' respresents the target senctence as a word vector \
# 'decoder_input_data' is the same as 'decoder_target_data' shifted the the right on step and therfore representing the previous word vector

# In[29]:


encoder_input_data = eng_padded
doc_len = encoder_input_data.shape[1]
decoder_input_data = deu_padded[:,:-1]
decoder_target_data = deu_padded[:,1:]


# Dictionary of all words and their index in the tokenizer dictionary

# In[ ]:


#deu_tokenizer.word_index


# In[30]:


num_encoder_tokens = max(eng_tokenizer.word_index.values()) + 1 
num_decoder_tokens = max(deu_tokenizer.word_index.values()) + 1
rev_target_token_index = {i:chr for chr, i in deu_tokenizer.word_index.items()}


# In[31]:


batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 1000  # Number of samples to train on.


# The model costructed in the following is similar to the model used in the character based example. However, between the Input and the LSTM layers an Embeding layer is added.

# In[ ]:


encoder_inputs = Input(shape=(None,), name="Encoder_Input")
x = Embedding(num_encoder_tokens, latent_dim, name="Encoder_Embedding")(encoder_inputs)
x, state_h, state_c = LSTM(latent_dim,return_state=True,name="Encoder_LSTM")(x)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,), name="Decoder_Input")
x = Embedding(num_decoder_tokens, latent_dim, name="Decoder_Embedding")(decoder_inputs)
x,_,_ = LSTM(latent_dim, return_state=True, return_sequences=True, name="Decoder_LSTM")(x, initial_state=encoder_states)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax',name="Decoder_Dense")(x)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# In[ ]:


#model.summary()


# The model is trained with the encoder and decoder input data and the decoder target data.\
# Because treining takes time, the model can be loaded in the next step.

# In[ ]:

"""
model.compile(optimizer="nadam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit([encoder_input_data,decoder_input_data], decoder_target_data,batch_size=batch_size,epochs=epochs,validation_split=0.2)
model.save("s2s_word_model.keras")
"""

# In[32]:


model = load_model("s2s_word_model.keras")


# In[ ]:


#print(model.input)


# To be able to generate translated sentences, an encoder and decoder model have to be constructed based on the previously trained layers.

# In[33]:


encoder_inputs = model.get_layer("Encoder_Input").input  
encoder_embedding = model.get_layer("Encoder_Embedding")(encoder_inputs)
encoder_outputs, state_h_enc, state_c_enc = model.get_layer("Encoder_LSTM")(encoder_embedding) 
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.get_layer("Decoder_Input").input
decoder_embedding = model.get_layer("Decoder_Embedding")(decoder_inputs)
decoder_input_h = Input(shape=(latent_dim,))
decoder_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_input_h, decoder_input_c]
decoder_outputs, h_dec, c_dec = model.get_layer("Decoder_LSTM")(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [h_dec,c_dec]
decoder_dense = model.get_layer("Decoder_Dense")
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


# In[ ]:


decoder_model.summary()


# The translate function takes in the vector representation of an (english) input sentences. \
# With each iteration a new word is decoded and added to the 'decoded_sentence' list until either the stop token '\n' or the maximin length is reached.

# In[34]:


def translate(input_sentence):
    states_value = encoder_model.predict(input_sentence, verbose=0)
    target_seq = np.zeros((1,1))
    target_seq[0,0] = deu_tokenizer.word_index["\t"]
    #print(target_seq)
    #print(states_value)

    stop = False
    decoded_sentence = ["\t"]
    while not stop:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0,0,:])
        #print(sampled_token_index)

        sampled_word = rev_target_token_index[sampled_token_index]
        decoded_sentence.append(sampled_word)

        if sampled_word == "\n" or len(decoded_sentence) > deu_max:
            stop = True
        
        target_seq[0,0] = sampled_token_index
        
        states_value = [h,c]
        #print(states_value)
    return decoded_sentence


# Ten random sentences from the data set are selected passed to the 'translate' function. \
# Index 0:800 are part of the training data: \
# Index 800:1000 are new to the models and can therfore be used as test data. \
# For some reason only the first word gets translated properly, when the states (h,c) get updated. Afterwards 0 gets sampled as token index which breaks the loop.\
# If the states do not get updated and remain initial, it is possible to decode several words. However, in most cases it leads to an alternating of the first two words.

# In[36]:


max_decoded_sentence_length = deu_max
start = np.random.randint(0,1000)
#start = 795
for i in range(start,start+10):
    input_sentence = encoder_input_data[i:i+1]
    decoded_sentence = translate(input_sentence)
    print(f"Input Sentence (English): {' '.join(eng[i])}")
    print(f"Decoded Sentence (German): {' '.join(decoded_sentence)}")
    print(f"Actual Sentence (German): {' '.join(deu[i])}")
    print("-"*40 + "\n")