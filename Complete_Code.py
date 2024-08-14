#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Embedding


# Data set source: https://www.manythings.org/anki/ German - English deu-eng.zip (271774)

# Data is read and loaded into a pandas dataframe.
"""
# In[ ]:


data = pd.read_csv("deu.txt", sep="\t", header=None)
data.drop(2, axis=1, inplace=True)
data.columns = ["Eng", "Deu"]
print(data.shape)
data.head()


# In[ ]:


data.drop_duplicates(subset="Eng",inplace=True)
print(data.shape)
data.head()


# After removing duplicates 1000 rows are selected.

# In[ ]:


whole_data = data
data = whole_data.iloc[:1000]
data


# Start token '\t' and and end token '\n' are added to the Strings

# In[ ]:


data.index = [x for x in range(data.shape[0])]
data["Start"] = pd.Series(["\t " for _ in range(data.shape[0])])
data["End"] = pd.Series([" \n" for _ in range(data.shape[0])])
data["Eng"] = data["Start"] + data["Eng"] + data["End"]
data["Deu"] = data["Start"] + data["Deu"] + data["End"]

data.head()


# In[ ]:


data.drop(["Start", "End"], axis=1, inplace=True)
print(data.iloc[0,0])
data.head()


# Shuffle the data to avoid any dependencies between rows n and n+1.

# In[ ]:


data = data.sample(frac=1).reset_index(drop=True)
data.to_csv("deu_prep.txt", sep=";", index=False)
data.head()

"""
# Start: Letter based approach

# First of all sets of all possible characters occuring in the input and output data

# In[2]:


data = pd.read_csv("deu_prep.txt", sep=";")
data.head()


# In[3]:


input_characters = set()
target_characters = set()

for seq in data.Eng:
    for chr in seq:
        input_characters.add(chr)
for seq in data.Deu:
    for chr in seq:
        target_characters.add(chr)

print(input_characters)
print(target_characters)


# The model in this example was trained based on the following character sets

# In[3]:


input_characters = ['d', 'a', '.', '\t', ':', 'm', 'u', 'j', 'k', 'B', 'H', 'r', 'K', 'o', 'x', '\n', 'S', 'V', 'n', '0', 'w', 'F', 'O', "'", 'A', 'P', 'v', 'p', 'D', 'c', 'L', 'g', 'e', 'Y', 'f', ',', 'W', '+', '?', 'b', '!', 'T', 'y', '5', 't', '3', 'N', 'C', 'J', 'R', 'z', '2', 'E', ' ', 'M', 'U', 'l', 's', 'i', 'q', 'h', 'I', 'G']
target_characters = ['d', 'ß', 'a', '.', '\t', 'Ä', ':', 'Ö', 'm', 'u', 'j', 'k', 'B', 'H', 'r', 'Ü', 'K', 'o', 'q', 'x', '\n', 'S', 'V', 'n', 'F', 'w', 'O', 'Z', 'A', 'ä', 'P', 'v', 'D', 'L', 'c', 'p', "'", 'g', 'e', ',', 'f', '-', 'W', '+', '?', 'b', '!', 'T', 'y', 'ü', 't', 'N', '3', '’', 'C', 'J', 'Q', 'R', 'z', '2', 'E', ' ', 'M', 'U', 'ö', 'l', 's', 'i', '0', 'h', 'I', 'G']


# In[4]:


num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

max_encoder_seq_length = max(len(txt) for txt in data.Eng) 
max_decoder_seq_length = max(len(txt) for txt in data.Deu) 


# In[6]:


print(f"Input Characters: {input_characters}")
print(f"Target Characters: {target_characters}")
print(f"Number of encoder tokens {num_encoder_tokens}")
print(f"Number of decoder tokens {num_decoder_tokens}")
print(f"Max length of encoder sequence {max_encoder_seq_length}")
print(f"Max length of decoder sequence {max_decoder_seq_length}")


# Based on the sets dictionarys are created providing an index for each character for lookup

# In[5]:


input_token_index  = {chr:i for i,chr in enumerate(input_characters)}
target_token_index = {chr:i for i,chr in enumerate(target_characters)}


# We initialize three empty vectors: \
# 'encoder_input_data' represents the input sentence as a vector of its character vectors \
# 'decoder_target_data' respresents the target senctence as a vector of its character vectors \
# 'decoder_input_data' is the same as 'decoder_target_data' shifted the the tight on step and therfore representing the previous character vectors

# In[6]:


encoder_input_data = np.zeros((data.shape[0], max_encoder_seq_length, num_encoder_tokens),dtype="float32")
decoder_input_data = np.zeros((data.shape[0], max_decoder_seq_length, num_decoder_tokens),dtype="float32")
decoder_target_data = np.zeros((data.shape[0], max_decoder_seq_length, num_decoder_tokens),dtype="float32")


# In[7]:


for i, (input_text, target_text) in enumerate(zip(data.Eng,data.Deu)):
    for t, chr in enumerate(input_text):
        encoder_input_data[i,t,input_token_index[chr]] = 1.0
    encoder_input_data[i,t+1 :,input_token_index[" "]] = 1.0
    for t, chr in enumerate(target_text):
        decoder_input_data[i,t,target_token_index[chr]] = 1.0
        if t > 0:
            decoder_target_data[i,t-1,target_token_index[chr]] = 1.0
    decoder_input_data[i, t+1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0


# These are some examples of characters and their vectorial representation

# In[21]:


for i,chr in enumerate(data.Eng.iloc[0]):
    print(f"{chr} -> {encoder_input_data[0,i]}")
    print(f"1.0 at position {input_token_index[chr]}")


# Construct the model

# In[8]:


batch_size = 64  # Batch size for training.
epochs = 500  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 1000  # Number of samples to train on.


# The model consits of two Input layers for encoder and decoder input data. Each Followed by a LSTM layer. 
# The hidden state and the cell state returned from the encoder LSTM are passed to the decoder LSTM as initial states.
# Finally the output of the decoder LSTM is passed to a Dense layer before returning the decoder output.

# In[11]:


encoder_inputs = Input(shape=(None,num_encoder_tokens))
encoder_lstm = LSTM(latent_dim,return_state=True)
_, h, c = encoder_lstm(encoder_inputs)
encoder_states = [h,c]

decoder_inputs = Input(shape=(None,num_decoder_tokens))
decoder_lstm = LSTM(latent_dim,return_sequences=True,return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens,activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs,decoder_inputs], decoder_outputs)


# 'model.summary()' give an overview over the model layer architecture.

# In[14]:


#model.summary()


# Train the model with the vectors created previously. \
# Because training takes time the ready model can be directly loaded in the next step!

# In[12]:

"""
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit([encoder_input_data,decoder_input_data], decoder_target_data,batch_size=batch_size,epochs=epochs,validation_split=0.2)
model.save("s2s_letter_model.keras")
"""

# In[9]:


model = load_model("s2s_letter_model.keras")


# In[11]:


#print(model.input)


# To be able to generate translated sentences, an encoder and decoder model have to be constructed based on the previously trained layers.

# In[12]:


encoder_inputs = model.input[0]
encoder_outputs, h_enc, c_enc = model.layers[2].output 
encoder_states = [h_enc, c_enc]
encoder_model = Model(encoder_inputs,encoder_states)

decoder_inputs = model.input[1]
decoder_input_h = Input(shape=(latent_dim,))
decoder_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_input_h, decoder_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, h_dec, c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [h_dec,c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


# In[13]:


rev_input_token_index  = {i:chr for chr,i in input_token_index.items()}
rev_targer_token_index = {i:chr for chr,i in target_token_index.items()}


# The translate function takes in the vector representation of an (english) input sentences. \
# With each iteration a new character is decoded and added to the 'decoded_sentence' until either the stop token '\n' or the maximin length is reached.

# In[14]:


def translate(input_sentence):
    states_value = encoder_model.predict(input_sentence, verbose=0)
    target_seq = np.zeros((1,1,num_decoder_tokens))
    target_seq[0,0,target_token_index["\t"]] = 1.0
    #print([target_seq] + states_value)

    stop = False
    decoded_sentence = "\t"

    while not stop:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        sampled_token_index = np.argmax(output_tokens[0,-1,:])
        sampled_char = rev_targer_token_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop = True
        
        target_seq = np.zeros((1,1,num_decoder_tokens))
        target_seq[0,0,sampled_token_index] = 1.0

        states_value = [h,c]
    
    return decoded_sentence


# Ten random sentences from the data set are selected passed to the 'translate' function. \
# Index 0:800 are part of the training data: \
# Index 800:1000 are new to the models and can therfore be used as test data.

# In[19]:


start = np.random.randint(0,1000)
#start = 795
for i in range(start,start+5):
    input_sentence = encoder_input_data[i:i+1]
    decoded_sentence = translate(input_sentence)
    print(f"Input sentence (English): {data.iloc[i,0]}")
    print(f"Decoded sentence (German): {decoded_sentence}")
    print(f"Actual sentence (German): {data.iloc[i,1]}")
    print("-"*40 + "\n")


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


# References: \
# Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." Advances in neural information processing systems 27 (2014). \
# \
# The character level approached is based this blog article of Francois Chollet, the creatior of Keras: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html 
# 
