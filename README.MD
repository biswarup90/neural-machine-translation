# Neural Machine Translation
Demonstrating language translation capability from English to Hindi using various Deep learning mechanisms such as LSTM, transformers.

## LSTM Based Approach

### 1. High Level Approach
1. Two RNNs Encoder and Decoder
2. Input is transformed through positional embedding + embedding
3. Input text is passed through an Encoder and a encoding of the text is produced. This is the state output of the final layer of LSTM
4. This encoded state is fed as the initial state to the decoder. The input text padded with start and end  is fed as input to the decoder.
5. For each word the decoder produces a translation which is passed through a Dense layer to produce the final output
6. We tried various options such as with shorter input sentences, recurrent dropout etc.

### 2. Brief code walkthrough
Refer the file - https://github.com/biswarup90/neural-machine-translation/blob/main/Code/lstm.py
1. Data is in a CSV which has two columns - english_sentence, hindi_sentence
2. **Text preprocessing:** lower case, remove special characters, append tokens START and END at the target column
3. Create vocabulary of source and target words, find length of longest sentence in source and target, vocabulary size(max among source & target vocabulary size)
4. Create a dictionary of word, index for source and target. This is required because models cannot work with actual text, so a numerical representative is needed for the words
5. Similary we need a reverse dictionary of index, word.
6. Split data set into train, test and validation
7. generate_batch function takes a X & Y dataset and a batch size.
	each batch can be viewed as a 2d array where i is batch index and j is each word in the sentence.
	Let's say batch size is 100, max seq length is 25 & vocab size is 1000
	each word's index is found in the input & target dict of word, index. Let they be called input_index, target_index.
	Let's take one instance, in the batch we consider 16th row, in the sentence the 8th word and this corresponds to word #34 in input vocabulary & 37th in target vocab.

	It returns three datasets.
	a. encoder_input_data(batch_size, maximum seq length encountered) -> each element in this array is input_index.
		in our example: encoder_input_data(15, 7) = 33
	b. decoder_input_data(batch_size, maximum seq length encountered) -> each element in this array is target_index
		in our example: decoder_input_data(15, 7) = 36
	c. decoder_target_data(batch_size, maximum seq length encountered, vocab_size)
		in our example: decoder_target_data(15, 7, 36) = 1
#### Model
The model is a Seq2Seq neural machine translation model that consists of an encoder and a decoder. The encoder takes in an input sequence of variable length, represented by the encoder_inputs tensor. The input sequence is passed through a PositionalEmbedding layer, which adds positional embeddings to the input sequence. The resulting tensor enc_emb is then fed into an LSTM layer with latent_dim size and recurrent_dropout of 0.5. The LSTM layer returns the final hidden states state_h and state_c as well as the output tensor encoder_outputs. The final hidden states are collected into a list encoder_states.

The decoder takes in a target sequence of variable length, represented by the decoder_inputs tensor. The input sequence is passed through a PositionalEmbedding layer, similar to the encoder. The resulting tensor dec_emb_layer is then fed into an LSTM layer with latent_dim size, return_sequences=True, and recurrent_dropout of 0.5. The LSTM layer takes in the encoder_states as the initial states. The output tensor decoder_outputs is passed through a Dense layer with vocab size and softmax activation function to produce the final output tensor.

The entire model is defined using the Model class in Keras, taking in both the encoder_inputs and decoder_inputs tensors as input and outputting the decoder_outputs tensor.

## Transformer
Refer the file https://github.com/biswarup90/neural-machine-translation/blob/main/Code/multiple_transformer.py
The code is an implementation of a Transformer-based neural machine translation model with multiple encoder and decoder layers. Here is a breakdown of the code:

1. The first part of the code defines some constants, such as the maximum length of the input and output sequences (MAX_LEN), the number of encoder and decoder layers (NUM_ENCODER_LAYERS and NUM_DECODER_LAYERS), the size of the word embedding and attention hidden layers (EMBEDDING_DIM and ATTENTION_DIM), and the size of the feedforward network in the encoder and decoder layers (FF_DIM).

2. The PositionalEncoding class defines the positional encoding layer that adds positional information to the input sequence.

3. The ScaledDotProductAttention class defines the scaled dot product attention mechanism used in the multi-head attention layer.

4. The MultiHeadAttention class defines the multi-head attention layer that computes multiple attention outputs and concatenates them before feeding them through a linear layer.

5. The TransformerEncoderLayer class defines a single encoder layer that consists of a multi-head attention layer followed by a feedforward network layer.

6. The TransformerEncoder class defines the entire encoder that consists of multiple encoder layers stacked on top of each other.

7. The TransformerDecoderLayer class defines a single decoder layer that consists of three multi-head attention layers and a feedforward network layer.

8. The TransformerDecoder class defines the entire decoder that consists of multiple decoder layers stacked on top of each other.

9. The Transformer class defines the entire transformer model that consists of the encoder and decoder layers.

10. The build_model function initializes and compiles the transformer model with the Adam optimizer and the categorical cross-entropy loss function.

11. The train function trains the transformer model on the input and target data.

12. The evaluate function evaluates the trained model on the validation set.

13. The predict function generates translations for the input data using the trained model.

14. Overall, the code implements a multiple-layer transformer model for neural machine translation and provides functions for training, evaluation, and prediction.
