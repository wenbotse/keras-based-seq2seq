from __future__ import print_function

import json
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import pickle

np.set_printoptions(threshold=np.inf)

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 80000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'train_data.txt'

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()


with open(data_path, 'r') as f:
    lines = f.read().split('\n')
    #print(lines[1])

#indices = np.arange(len(lines))
#np.random.shuffle(indices)
#lines = lines[indices]

#np.random.shuffle(lines)

print('after shuffle='+lines[1])
for line in lines[: min(num_samples, len(lines) - 1)]:
    if len(line.split('=====')) != 2:
        continue 
    input_text, target_text = line.split('=====')
    #input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

print("step_1")
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])

#with open('input_token_index.json','w') as outfile:
#    json.dump(input_token_index,outfile, ensure_ascii=False)
#    outfile.write('\n')

input_token_index_pkl_file = open('input_token_index.pkl','wb')
pickle.dump(input_token_index,input_token_index_pkl_file)
input_token_index_pkl_file.close()

print("step_2")
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

target_token_index_pkl_file = open('target_token_index.pkl','wb')
pickle.dump(target_token_index,target_token_index_pkl_file)
target_token_index_pkl_file.close()


print("step_3 ")
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
print("step_4")
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
print("step_5")
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
print("step_6")
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    #print(str(i)+" input_text="+input_text)
    try:
    
    	for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
    	for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    except:
        print("exception "+str(i)+" input_text="+input_text)
        
print("step_7")

print('Input token index size=',str(len(input_token_index)))
print('Target token index size=',str(len(target_token_index)))
print('encoder_input_data shape=')
print(encoder_input_data.shape)
print('decoder_input_data shape=')
print(decoder_input_data.shape)
print('decoder_target_data shape=')
print(decoder_target_data.shape)
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
print("step_8")
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
print("prepare model")
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
print("model compile")

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    #print('input_seq')
    #print(input_seq.shape)
    #print('states_value')
    #print(states_value)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

class PredictEpochCallback(Callback):
    def on_train_begin(self, logs={}):
        print("begin train epoch")
    def on_epoch_end(self, epoch, logs={}):
        for seq_index in range(10):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            input_seq = encoder_input_data[seq_index: seq_index + 1]
            print('input_seq.shape')
            print(input_seq.shape)
            decoded_sentence = decode_sequence(input_seq)
            print('epoch i='+str(epoch)+' input='+input_texts[seq_index]+' decode output='+decoded_sentence)
        # save whole model
        print('save s2s h5')
        model.save('s2s.h5.'+str(epoch))
        model_json = model.to_json()
        with open("model.json."+str(epoch), "w") as json_file:
            print('save s2s json')
            json_file.write(model_json)
          

        encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)
'''             
        #save encoder model
        print('save encoder h5')
        encoder_model.save('encoder_model.h5.'+str(epoch))
        encoder_model_json = encoder_model.to_json()
        with open("encoder_model.json."+str(epoch), "w") as json_file:
            print('save encoder json')
            json_file.write(encoder_model_json)

        #save decoder model
        print('save decoder h5')
        decoder_model.save('decoder_model.h5.'+str(epoch))
        decoder_model_json = decoder_model.to_json()
        with open("decoder_model.json."+str(epoch), "w") as json_file:
            print('save decoder json')
            json_file.write(decoder_model_json)
'''

predictEpochCallback = PredictEpochCallback()
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          callbacks=[predictEpochCallback])
print("model fit")
# Save model
model.save('s2s.h5')
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
'''
for seq_index in range(2):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)

'''
