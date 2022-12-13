

# Using RNN

### Imports
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
import pandas as pd
import string
from string import digits
import matplotlib.pyplot as plt
# %matplotlib inline
import re
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense, Flatten, Bidirectional, Concatenate, Layer
from keras.utils.vis_utils import plot_model
from keras.models import Model
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf

"""### Load Data"""

#drive.mount('/content/drive/')
#en_hi_data_path = '/content/drive/Shareddrives/DA225-O/project/data/Hindi_English_Truncated_Corpus.csv'
en_hi_data_path = '' ### Provide location of data. The file is uploaded on repo.
lines = pd.read_csv(en_hi_data_path)
lines=lines[lines['source']=='ted']
#lines = lines[:10]

"""### Text Pre-processing"""

# Lowercase all characters
lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.lower())
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.lower())

# Remove quotes
lines['english_sentence']=lines['english_sentence'].apply(lambda x: re.sub("'", '', x))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: re.sub("'", '', x))

exclude = set(string.punctuation) # Set of all special characters
# Remove all the special characters
lines['english_sentence']=lines['english_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

remove_digits = str.maketrans('', '', digits)
lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.translate(remove_digits))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.translate(remove_digits))

lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

# Remove extra spaces
lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.strip())
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.strip())
lines['english_sentence']=lines['english_sentence'].apply(lambda x: re.sub(" +", " ", x))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: re.sub(" +", " ", x))

lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x : 'START_ '+ x + ' _END')

lines.head()

"""### Sizes"""

# Vocabulary of English
all_eng_words=set()
for eng in lines.english_sentence:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

# Vocabulary of Hindi 
all_hindi_words=set()
for mar in lines.hindi_sentence:
    for word in mar.split():
        if word not in all_hindi_words:
            all_hindi_words.add(word)

lines['length_eng_sentence']=lines['english_sentence'].apply(lambda x:len(x.split(" ")))
lines['length_hin_sentence']=lines['hindi_sentence'].apply(lambda x:len(x.split(" ")))

lines=lines[lines['length_eng_sentence']<=20]
lines=lines[lines['length_hin_sentence']<=20]

# Max Length of source sequence
lenght_list=[]
for l in lines.english_sentence:
    lenght_list.append(len(l.split(' ')))
max_length_src = int(np.max(lenght_list))

# Max Length of target sequence
lenght_list=[]
for l in lines.hindi_sentence:
    lenght_list.append(len(l.split(' ')))
max_length_tar = int(np.max(lenght_list))



input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_hindi_words))

# Calculate Vocab size for both source and target
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_hindi_words)
num_decoder_tokens += 1 # For zero padding

# Create word to token dictionary for both source and target
input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])

# Create token to word dictionary for both source and target
reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())

"""### Train-Test Split"""

X, y = lines['english_sentence'], lines['hindi_sentence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2,random_state=42)
X_train.shape, X_test.shape

sequence_length = max(max_length_src, max_length_tar)
vocab = max(num_encoder_tokens, num_decoder_tokens)

print("Sequence length: ", sequence_length)
print("vocab: ", vocab)

train_samples = len(X_train)
val_samples = len(X_test)
batch_size = 128
epochs = 30

"""### Divide data into batch"""

def generate_batch(X = X_train, y = y_train, batch_size = 128):
    ''' Generate a batch of data '''
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = np.zeros((batch_size, sequence_length),dtype='float32')
            decoder_input_data = np.zeros((batch_size, sequence_length),dtype='float32')
            decoder_target_data = np.zeros((batch_size, sequence_length, vocab),dtype='float32')
            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text.split()):
                    #print("encoder_input_data: ", word)
                    #print(input_token_index[word])
                    encoder_input_data[i, t] = input_token_index[word] # encoder input seq
                for t, word in enumerate(target_text.split()):
                    if t<len(target_text.split())-1:
                        #print("decoder_input_data: ", word)
                        #print(target_token_index[word])
                        #print(i)
                        #print(t)
                        decoder_input_data[i, t] = target_token_index[word] # decoder input seq
                    if t>0:
                        #print("decoder_target_data: ", word)
                        decoder_target_data[i, t - 1, target_token_index[word]] = 1.
            #print(decoder_input_data)
            yield([encoder_input_data, decoder_input_data], decoder_target_data)

"""### Positional Embedding"""

class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        # input_dim = (token) vocabulary size,  output_dim = embedding size
        super().__init__(**kwargs)

        self.token_embeddings = Embedding(       # Q: what is input_dim and output_dim?
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = Embedding(    # Q: Why input_dim = seq_length? 
            input_dim=sequence_length, output_dim=output_dim)   # Q: What is the vocab for this Embedding layer
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):   # inputs will be a batch of sequences (batch, seq_len)
        length = tf.shape(inputs)[-1]     # lenght will just be sequence length
        positions = tf.range(start=0, limit=length, delta=1) # indices for input to positional embedding 
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        output = embedded_tokens + embedded_positions
        #print("Embdeding input shape: ", inputs.shape)
        #print("Embedding output shape: ", output.shape)
        return output     # ADD the embeddings

    def compute_mask(self, inputs, mask=None):  # makes this layer a mask-generating layer
        return tf.math.not_equal(inputs, 0)     #mask will get propagated to the next layer.

    # When using custom layers, this enables the layer to be reinstantiated from its config dict, 
    # which is useful during model saving and loading.
    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config

"""## LSTM

### Set up encoder decoder

***Encoder***
"""

latent_dim = 256
encoder_inputs = Input(shape=(None,))
enc_emb =  PositionalEmbedding(sequence_length, vocab, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True, recurrent_dropout=0.5)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

encoder_states = [state_h, state_c]

"""***Decoder***"""

decoder_inputs = Input(shape=(None,))
dec_emb_layer = PositionalEmbedding(sequence_length, vocab, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, recurrent_dropout=0.5)
decoder_outputs, _, _ = decoder_lstm(dec_emb_layer, initial_state=encoder_states)
decoder_dense = Dense(vocab, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

"""### Compile Model"""

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
plot_model(model, show_shapes=True, expand_nested=True)
#model.summary()

"""### Train Model"""

model.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batch_size),
                    steps_per_epoch = train_samples//batch_size,
                    epochs=epochs,
                    validation_data = generate_batch(X_valid, y_valid, batch_size = batch_size),
                    validation_steps = val_samples//batch_size
                    )

"""### Inference Setup"""

encoder_model = Model(encoder_inputs, encoder_states)

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Get the embeddings of the decoder sequence
#dec_emb2= dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb_layer, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2)

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

plot_model(encoder_model, show_shapes=True, expand_nested=True)

plot_model(decoder_model, show_shapes=True, expand_nested=True)

#print(reverse_target_char_index)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    #print(input_seq)
    states_value = encoder_model.predict(input_seq)
    #print(states_value)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        #print("output_tokens: ",output_tokens)
        #print("h: ", h)
        #print("c: ",c)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #print(sampled_token_index)
        sampled_char = reverse_target_char_index[sampled_token_index]
        #print(sampled_char)
        decoded_sentence += ' '+sampled_char
        
        # Exit condition: either hit max length or find stop token.
        if (sampled_char == '_END' or len(decoded_sentence) > 50):
            stop_condition = True
        
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    
    return decoded_sentence

"""## Bidirectional LSTM

### Set up encoder decoder

***Encoder***
"""

latent_dim = 256
encoder_bi_inputs = Input(shape=(None,))
enc_bi_emb =  PositionalEmbedding(sequence_length, vocab, latent_dim)(encoder_bi_inputs)
encoder_bi_lstm = Bidirectional(LSTM(latent_dim, return_state=True, recurrent_dropout=0.5))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bi_lstm(enc_bi_emb)
state_bi_h = Concatenate()([forward_h, backward_h])
state_bi_c = Concatenate()([forward_c, backward_c])
encoder_bi_states = [state_bi_h, state_bi_c]

"""***Decoder***"""

decoder_bi_inputs = Input(shape=(None,))
dec_emb_bi_layer = PositionalEmbedding(sequence_length, vocab, latent_dim)(decoder_bi_inputs)

decoder_bi_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True, recurrent_dropout=0.5)
decoder_bi_outputs, _, _ = decoder_bi_lstm(dec_emb_bi_layer, initial_state=encoder_bi_states)
decoder_bi_dense = Dense(vocab, activation='softmax')
decoder_bi_outputs = decoder_bi_dense(decoder_bi_outputs)

model_bi_lstm = Model([encoder_bi_inputs, decoder_bi_inputs], decoder_bi_outputs)

"""### Compile Model"""

model_bi_lstm.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
plot_model(model_bi_lstm, show_shapes=True, expand_nested=True)
#model.summary()

"""### Train Model"""

model_bi_lstm.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batch_size),
                    steps_per_epoch = train_samples//batch_size,
                    epochs=epochs,
                    validation_data = generate_batch(X_test, y_test, batch_size = batch_size),
                    validation_steps = val_samples//batch_size
                    )

"""### Inference Setup"""

encoder_bi_model = Model(encoder_bi_inputs, encoder_bi_states)

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h_bi = Input(shape=(latent_dim*2,))
decoder_state_input_c_bi = Input(shape=(latent_dim*2,))
decoder_states_inputs_bi = [decoder_state_input_h_bi, decoder_state_input_c_bi]

# Get the embeddings of the decoder sequence
#dec_emb2_bi= dec_emb_bi_layer(decoder_bi_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2_bi, state_h2_bi, state_c2_bi = decoder_bi_lstm(dec_emb_bi_layer, initial_state=decoder_states_inputs_bi)
decoder_states2_bi = [state_h2_bi, state_c2_bi]

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2_bi = decoder_bi_dense(decoder_outputs2_bi)

# Final decoder model
decoder_model_bi = Model(
    [decoder_bi_inputs] + decoder_states_inputs_bi,
    [decoder_outputs2_bi] + decoder_states2_bi)

def decode_sequence_bi(input_seq):
    # Encode the input as state vectors.
    #print(input_seq)
    states_value = encoder_bi_model.predict(input_seq)
    #print(len(states_value[0][0]))
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model_bi.predict([target_seq] + states_value)
        #print("output_tokens: ",output_tokens.shape)
        #print("h: ", h)
        #print("c: ",c)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #print(sampled_token_index)
        sampled_char = reverse_target_char_index[sampled_token_index]
        #print(sampled_char)
        decoded_sentence += ' '+sampled_char
        
        # Exit condition: either hit max length or find stop token.
        if (sampled_char == '_END' or len(decoded_sentence) > 50):
            stop_condition = True
        
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    
    return decoded_sentence

"""## Inference

"""

test_gen = generate_batch(X_test, y_test, batch_size = 1)
k=-1

k+=1
(input_seq, actual_output), target_output = next(test_gen)
decoded_sentence = decode_sequence(input_seq)
decoded_sentence_bi = decode_sequence_bi(input_seq)
print('Input English sentence:', X_test[k:k+1].values[0])
print('Actual Hindi Translation:', y_test[k:k+1].values[0][6:-4])
print('Predicted Hindi Translation by LSTM:', decoded_sentence[:-4])
print('Predicted Hindi Translation by Bi-LSTM:', decoded_sentence_bi[:-4])

input_seq

actual_output

target_token_index['एक']

while True:pass