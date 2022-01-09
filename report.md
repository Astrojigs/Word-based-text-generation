# Report for Hw3 Unsupervised Learning

# Objective:
To generate text (new ideas) based on the files provided.

# Extraction of Data:
I have used only 800 text files for training purposes due to hardware limit on my computer.

From each text file, I've extracted only the 'GOAL'

```
data_path = "data"
data_file_names = os.listdir(path=data_path)

corpus = []

for i in range(len(data_file_names)):
    full_path = os.path.join(data_path,data_file_names[i])
    try:
        corpus.append(open(full_path).read()[open(full_path,'r').read().find('GOAL') + 4:open(full_path,'r').read().find('== DATA')].replace('\n',''))
    except:
        pass

# Minimizing corpus size:
corpus = corpus[:800]
```

# Model:
I have use a Bidirectional LSTM network for the purposes of generating new text.
```
# Creating model
model = tf.keras.models.Sequential()
model.add(Embedding(input_dim=total_words, output_dim=250, input_length=max_sequence_len - 1))
model.add(Bidirectional(LSTM(150,return_sequences=True)))
model.add(Bidirectional(LSTM(150,return_sequences=True)))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
```
