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
8. Encoder model -> input, positional_embedding, LSTM
9. Decoder model -> input, positional_embedding, LSTM, Dense layer for classification
10. The total model is ([encoder_input, decoder_input], decoder_output). The encoder states are fed as initial state to the decoder.
11. Model is trained on train data set and validated on validation data set
12. We also tried with Bi-directional LSTM.
