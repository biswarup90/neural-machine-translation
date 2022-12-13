
# Transformers

### Import



import string
from google.colab import drive
import pandas as pd
import string
from string import digits
import matplotlib.pyplot as plt
# %matplotlib inline
import re
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense, Flatten
from keras.utils.vis_utils import plot_model
from keras.models import Model, Sequential
from tensorflow.keras.layers import TextVectorization, Layer, MultiHeadAttention, Dense, LayerNormalization, Dropout
import tensorflow as tf
import random

"""### Load Data"""

#drive.mount('/content/drive/')
en_hi_data_path = '' ### Provide location of data. The file is uploaded on repo.
lines = pd.read_csv(en_hi_data_path)
lines=lines[lines['source']=='ted']
#lines = lines[:5000]

lines.head(3)

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

lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x : '[start] '+ x + ' [end]')

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

print("num_encoder_tokens: ", num_encoder_tokens)
print("num_decoder_tokens: ", num_decoder_tokens)

print("max_length_src: ", max_length_src)
print("max_length_tar: ", max_length_tar)

sequence_length = max(max_length_src, max_length_tar)
vocab = max(num_encoder_tokens, num_decoder_tokens)

"""### Train-Test Split"""

text_pairs = []

for index, line in lines.iterrows():
    eng = line['english_sentence']
    hin = line['hindi_sentence']
    text_pairs.append((eng, hin))

random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]

"""### Text Vectorization"""

vocab_size = vocab
sequence_length = sequence_length

strip_chars = string.punctuation + "¿"  # strip out stadard punctuations + extra one in spanish
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

# Custom standardization function for spanish
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(    # Replace elements of input matching regex pattern with rewrite.
        lowercase, f"[{re.escape(strip_chars)}]", "")
    
source_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
target_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization
)

train_english_texts = [pair[0] for pair in train_pairs]
train_hindi_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_hindi_texts)

# Preparing datasets for the translation task

batch_size = 64

# returns tuple- ()
def format_dataset(eng, hin):
    # Q: What are eng and spa pre and post re-assignment
    eng = source_vectorization(eng)  
    hin = target_vectorization(hin)
    return ({
        "english": eng,           # encoder nput
        "hindi": hin[:, :-1],    # decoder input Q: what is the first axis?
    }, hin[:, 1:])                  # decoder ouput

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache() #Use in-memory caching to speed up preprocessing.

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    #print(inputs['english'])
    print(f"inputs['hindi'].shape: {inputs['hindi'].shape}")
    print(f"targets.shape: {targets.shape}")

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
        return embedded_tokens + embedded_positions     # ADD the embeddings

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

"""### Transformer Encoder"""

class TransformerEncoder(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim    # Dimension of embedding. 4 in the dummy example
        self.dense_dim = dense_dim    # No. of neurons in dense layer
        self.num_heads = num_heads    # No. of heads for MultiHead Attention layer
        self.attention = MultiHeadAttention(   # MultiHead Attention layer - 
            num_heads=num_heads, key_dim=embed_dim, dropout = 0.1)   # see coloured pic above
        self.dense_proj = Sequential(
            [Dense(dense_dim, activation="relu"),
             Dropout(0.1),
              Dense(embed_dim)]    
        )                               
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()

        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)

    # Call function based on figure above
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]   # Will discuss in next tutorial
            #print(f"**test: mask in not None. mask = {mask}")

        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)  # Query: inputs, Value: inputs, Keys: Same as Values by default
                                                  # Q: Can you see how this is self attention?
        #attention_output = self.dropout1(attention_output)
        proj_input = self.layernorm_1(inputs + attention_output) # LayerNormalization; + Recall cat picture
        proj_output = self.dense_proj(proj_input)
        #proj_output = self.dropout1(proj_output)
        x = self.layernorm_2(proj_input + proj_output)
        #print("Output shape of encoder output: ", x.shape)
        return self.layernorm_2(proj_input + proj_output)  # LayerNormalization + Residual connection

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return

"""### Transformer Decoder"""

class TransformerDecoder(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        # Define the layers. Let's point them out in the diagram
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        # Now we have 2 MultiHead Attention layers - one for ___ attention and one for ____ attention
        self.attention_1 = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout = 0.1)
        self.attention_2 = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout = 0.1)
        self.dense_proj = Sequential(
            [Dense(dense_dim, activation="relu"),
             Dropout(0.1),
             Dense(embed_dim),]
        )
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.layernorm_3 = LayerNormalization()

        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)
        self.dropout3 = Dropout(0.1)

        self.supports_masking = True #ensures that the layer will propagate its input mask to its outputs;

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1])) # sequence_length == input_shape[1]
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
              tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None): # two inputs: decoder i/p and encoder o/p
        causal_mask = self.get_causal_attention_mask(inputs)
        # print(f"*** test: mask = {mask}")
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask) # union of 0s
            # print(f"**** test padding mask: {padding_mask}")
        attention_output_1, weights_1 = self.attention_1(    # Q: What kind of attention? 
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask,
            return_attention_scores=True) # Q: What will the causal_mask do?
        #attention_output_1 = self.dropout1(attention_output_1)    
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        #print("shape of attention_output_1: ", attention_output_1.shape)
        #print("shape of encoder_outputs: ", encoder_outputs.shape)
        attention_output_2, weights_2 = self.attention_2(  # Q: Is this self attention?
            query=attention_output_1,
            value=encoder_outputs,    # Key and Value coming from encoder
            key=encoder_outputs,
            attention_mask=padding_mask,
            return_attention_scores=True
        )
        #attention_output_2 = self.dropout2(attention_output_2)
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)
        
        proj_output = self.dense_proj(attention_output_2)
        #proj_output = self.dropout3(proj_output)
        out = self.layernorm_3(attention_output_2 + proj_output)
        return out#, weights_1, weights_2

"""### The Transformer"""

embed_dim = 256
num_heads = 8
dense_dim = 2048
encoder_inputs = Input(shape=(None,), dtype="int64", name="english")  
x = PositionalEmbedding(max_length_tar, num_encoder_tokens, embed_dim)(encoder_inputs) 
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)


attention_weights = {}
decoder_inputs = Input(shape=(None,), dtype="int64", name="hindi") 
x = PositionalEmbedding(max_length_tar, num_decoder_tokens, embed_dim)(decoder_inputs) 
x = Dropout(0.1)(x)
x= TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs) 
x= TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
x= TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
x= TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
x = Dropout(0.1)(x)
decoder_outputs = Dense(num_decoder_tokens, activation="softmax")(x)
transformer = Model([encoder_inputs, decoder_inputs], decoder_outputs)

"""## Traning and evaluating the model"""

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(embed_dim)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
transformer.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
transformer.summary()
plot_model(transformer, show_shapes=True, expand_nested=True)

transformer.fit(train_ds, epochs=20, validation_data=val_ds)

import numpy as np
spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 32

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization(
            [decoded_sentence])[:, :-1]
        predictions = transformer(
            [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

test_eng_texts = [pair[0] for pair in test_pairs]
test_hin_texts = [pair[1] for pair in test_pairs]
#random.shuffle(test_eng_texts)
for _ in range(4):
    i = random.choice(range(len(test_eng_texts)))
    input_sentence = test_eng_texts[i]
    output_sentence = test_hin_texts[i]
    print("-")
    print(input_sentence)
    print(output_sentence)
    print(decode_sequence(input_sentence))