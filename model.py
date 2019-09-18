from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import keras
from IPython.display import clear_output
import pickle

def create_model(vocab_size, seq_length, save_every):

    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(120, return_sequences=True))
    model.add(LSTM(120))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())




    

    class PlotLosses(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.i = 0
            self.x = []

            
            
            self.logs = []

        def on_epoch_end(self, epoch, logs={}):
            
            self.x.append(self.i)            
            self.i += 1
            
            clear_output(wait=True)
                        
            ## EVERY N EPOCHS THIS WILL SAVE MODEL
            if self.i % save_every == 0:
                name = 'saved_model_at%03d.h5' % self.i
                self.model.save(name)
            ## Save history every 10 epochs
            #if self.i % 10 == 0:
            #    with open('train_History%03d' % self.i, 'wb') as file_pi:
            #        pickle.dump(self.losses, file_pi)
            #
            
    plot_losses = PlotLosses()




    # compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, plot_losses