import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences


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

total_words = len(tokenizer.word_index) + 1


input_sequences = []
for chunk in read_file_in_chunks("hamlet.txt"):
    for line in chunk.split('\n'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)


max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))


X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(SimpleRNN(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X, y, epochs=100, verbose=1)

def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word = tokenizer.index_word[np.argmax(predicted)]
        seed_text += " " + predicted_word
    return seed_text

print(generate_text("In a faraway", 10, max_sequence_len))
