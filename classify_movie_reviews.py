import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

"""
https://www.tensorflow.org/tutorials/keras/basic_text_classification

Result:
    Small sample size, no speed gain has been observed from GPU

"""

# ------
# data
# ------
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# convert the integers back to words
word_index = imdb.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START"] = 1
word_index["<UNK"] = 2  # unknown
word_index["<UNUSED"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join(([reverse_word_index.get(i, '?') for i in text]))


review_text = decode_review(train_data[1])
print(review_text)

# ---------------------
# Prepare the data
# ---------------------
num_samples = len(train_data)
max_len = 256

assert train_data.shape == (num_samples,), "err"

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=max_len
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=max_len
)

assert train_data.shape == (num_samples, max_len), "err"
assert test_data.shape == (num_samples, max_len), "err"

# ---------------------
# Build the model
# ---------------------
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

print(model.summary())

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------
# Create a validation set
# -----------------------
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# ---------------
# Train the model
# ---------------
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=2
)

# ------------------
# Evaluate the model
# ------------------
results = model.evaluate(test_data, test_labels)
print(results)
