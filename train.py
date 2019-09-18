import argparse
import model

ap = argparse.ArgumentParser()

ap.add_argument("-i",  help = " Enter your input file [This argument is REQUIRED]", required=True)
ap.add_argument("-s",  help = " Enter your sequence length", default = '50',  type=int)
ap.add_argument("-e",  help = " Enter your number of epochs", default = '100',  type=int)
ap.add_argument("-n",  help = " Save your model every n epochs", default = '10',  type=int)
ap.add_argument("-b",  help = " Enter your batch size", default = '128', type=int)

args = vars(ap.parse_args())

input_file = args['i']
seq_len = args['s']
nb_epochs = args['e']
save_every = args['n']
bs = args['b']


# load doc into memory
def load_doc(filename):
  # open the file as read only
  file = open(filename, 'r')
  # read all text
  text = file.read()
  # close the file
  file.close()
  return text

# load document
print('Loading document ...')
doc = load_doc(input_file)
print('Document is loaded.')

print('Text Cleaning ...')
import string

# turn a doc into clean tokens
def clean_doc(doc):
	# replace '--' with a space ' '
	doc = doc.replace('--', ' ')
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens

# clean document
tokens = clean_doc(doc)

print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))


# organize into sequences of tokens
length = seq_len + 1
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
	sequences.append(line)
print('Total Sequences: %d' % len(sequences))



# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
# save sequences to file
print('Saving sequences')
out_filename = 'sequences.txt'
save_doc(sequences, out_filename)






# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load
in_filename = 'sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

print('Tokenizering ...')
from keras.preprocessing.text import Tokenizer
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
from pickle import dump

# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
from keras.utils import to_categorical
import numpy as np
# separate into input and output
sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
#y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

model, callback = model.create_model(vocab_size ,seq_length, save_every)

# fit model
model.fit(X, y, batch_size=bs, epochs=nb_epochs, callbacks=[callback])

