import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from keras.regularizers import l2
import numpy as np


def read_file_in_chunks(file_path, chunk_size=1024):
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            data = file.read(chunk_size)
            if not data:
                break
            yield data


tokenizer = Tokenizer()
for chunk in read_file_in_chunks("hamlet.txt"):
    tokenizer.fit_on_texts([chunk])

word_index = tokenizer.word_index


input_sequences = []
for chunk in read_file_in_chunks("hamlet.txt"):
    token_list = tokenizer.texts_to_sequences([chunk])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


max_length = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_length, padding='pre'))


X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=len(word_index) + 1)


model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=64, input_length=max_length-1))  
model.add(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01))) 
model.add(Dropout(0.2)) 
model.add(LSTM(128, kernel_regularizer=l2(0.01)))  
model.add(Dropout(0.2))  
model.add(Dense(64, activation='relu')) 
model.add(Dense(len(word_index) + 1, activation='softmax')) 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X, y, epochs=50, batch_size=32, verbose=1)


model.summary()


def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)
        output_word = tokenizer.index_word[predicted_word_index[0]]
        seed_text += " " + output_word
    return seed_text


seed_text = "In a faraway land"
generated_text = generate_text(seed_text, 10, max_length)
print(generated_text)
