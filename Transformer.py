import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, Dense, Dropout, Add, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
import numpy as np


data = """In a faraway land, there was a village by the sea. People in the village were happy and lived peacefully. They loved to sing songs and tell stories under the moonlight."""


tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
sequences = tokenizer.texts_to_sequences([data])
word_index = tokenizer.word_index


input_sequences = []
for i in range(1, len(sequences[0])):
    n_gram_sequence = sequences[0][:i+1]
    input_sequences.append(n_gram_sequence)


max_length = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_length, padding='pre'))


X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=len(word_index) + 1)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

embed_dim = 128  
num_heads = 4  
ff_dim = 128  

inputs = Input(shape=(max_length-1,))
embedding_layer = TokenAndPositionEmbedding(max_length-1, len(word_index) + 1, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.2)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(len(word_index) + 1, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.fit(X, y, epochs=100, batch_size=32, verbose=1)


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
