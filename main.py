import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os

#Code to a void the cuda gpu error
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM


# Storage place
data_path = "data"

# List number of items in that path
data_file_names = os.listdir(path=data_path)
print(data_file_names[:3])
# Number of data files:
print(len(data_file_names))

#############
## code check
full_path = os.path.join(data_path,data_file_names[2])
open(full_path,'r').read()[open(full_path,'r').read().find('GOAL') + 4:open(full_path,'r').read().find('== DATA')].replace('\n','')
# open(full_path,'r').read()
#############


# Data:
corpus = []

for i in range(len(data_file_names)):
    full_path = os.path.join(data_path,data_file_names[i])
    try:
        corpus.append(open(full_path).read()[open(full_path,'r').read().find('GOAL') + 4:open(full_path,'r').read().find('== DATA')].replace('\n',''))
    except:
        pass

# Cropping due to memory issues
corpus = corpus[:800]


# Tokenizer instance:
tokenizer.fit_on_texts(corpus)

# for Using out of vocabulary token we add 1.
total_words = len(tokenizer.word_index) + 1

print(total_words)
# 4314 words

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)
    # We have split the sentence into multiple lists


# Padding the token_list
max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences,
 maxlen=max_sequence_len, padding='pre'))



# Slicing lists
xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]

xs = xs.astype(np.int16)
# Y should be categorical and one-hot encoded
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)



############# MODEL #####################3

# Creating model
model = tf.keras.models.Sequential()
model.add(Embedding(input_dim=total_words, output_dim=250, input_length=max_sequence_len - 1))
model.add(Bidirectional(LSTM(150,return_sequences=True)))
model.add(Bidirectional(LSTM(150,return_sequences=True)))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

tf.keras.utils.plot_model(model, show_shapes=True,to_file="Model_plot.png")


# GPU sync error prevention
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

# Fitting the model
history = model.fit(xs, ys, epochs=500, verbose=1)
'''
659/659 [==============================] - 265s 402ms/step - loss: 0.1859
Epoch 131/500
659/659 [==============================] - 260s 394ms/step - loss: 0.1471
Epoch 132/500
659/659 [==============================] - 260s 395ms/step - loss: 0.1328
Epoch 133/500
659/659 [==============================] - 261s 397ms/step - loss: 0.1295
Epoch 134/500
 40/659 [>.............................] - ETA: 4:21 - loss: 0.1126'''

# Saving the model:
#model.save('unsupervised_model_1')

model_1 = tf.keras.models.load_model('unsupervised_model_1')



####################3 RESULTS #######################
def generate_text(seed_text,next_words,model):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += ' ' + output_word
    return seed_text


generate_text('fruit',200,model_1)
