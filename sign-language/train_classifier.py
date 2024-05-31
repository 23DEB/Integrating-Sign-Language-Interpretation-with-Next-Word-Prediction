import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

lengths = [len(seq) for seq in data_dict['data']]
print("Minimum sequence length:", min(lengths))
print("Maximum sequence length:", max(lengths))
def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', truncating='post', value=0.):
    if not maxlen:
        maxlen = max(len(seq) for seq in sequences)

    padded_sequences = np.full((len(sequences), maxlen), value, dtype=dtype)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:  # Truncate if necessary
            if truncating == 'pre':
                trunc = seq[-maxlen:]
            else:
                trunc = seq[:maxlen]
        else:
            trunc = seq

        # Pad if necessary
        if padding == 'post':
            padded_sequences[i, :len(trunc)] = trunc
        else:
            padded_sequences[i, -len(trunc):] = trunc
    return padded_sequences

# Use the padding function
data_padded = pad_sequences(data_dict['data'])
data = np.array(data_padded)  # This should now work without error


#data = np.asarray(data_dict['data'])
data = pad_sequences(data_dict['data'])  # Assuming you've defined pad_sequences as shown above
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
