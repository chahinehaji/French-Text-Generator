import argparse


ap = argparse.ArgumentParser()

ap.add_argument("-m",  help = " Enter your model file [This argument is REQUIRED]", required=True)
ap.add_argument("-t",  help = " Enter your tokenizer pkl file", default = 'tokenizer.pkl')
ap.add_argument("-w",  help = " Enter your number of words", default = '50',  type=int)
ap.add_argument("-s",  help = " Enter your sequences file", default = 'sequences.txt')
ap.add_argument("-i",  help = " Enter your seed text", default = ' ')

args = vars(ap.parse_args())

seed_text = args['i']
model_path = args['m']
tokenizer_path = args['t']
seq_length = args['w']
sequence_file = args['s']



# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)



from random import randint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
from keras.preprocessing.text import Tokenizer




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

doc = load_doc(sequence_file)
lines = doc.split('\n')

model = load_model(model_path)

tokenizer = Tokenizer()
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)



# select a seed text


# generate new text
generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
print(seed_text + '\n')
print(generated)