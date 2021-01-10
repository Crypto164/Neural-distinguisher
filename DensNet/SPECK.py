from pickle import dump
import numpy as np
from os import urandom
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2
import numpy as np
from keras import layers
from keras import models
from keras.layers import Dropout
bs = 5000

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)

cardinality = 32

def WORD_SIZE():
    return(16)

def ALPHA():
    return(7)

def BETA():
    return(2)

MASK_VAL = 2 ** WORD_SIZE() - 1

def shuffle_together(l):
    state = np.random.get_state()
    for x in l:
        np.random.set_state(state)
        np.random.shuffle(x)

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))

def enc_one_round(p, k):
    c0, c1 = p[0], p[1]
    c0 = ror(c0, ALPHA())
    c0 = (c0 + c1) & MASK_VAL
    for i in range(16): 
        c0[i] = c0[i] ^ k
    c1 = rol(c1, BETA())
    c1 = c1 ^ c0
    return(c0,c1)

def dec_one_round(c,k):
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = ror(c1, BETA())
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = rol(c0, ALPHA())
    return(c0, c1)

def expand_key(k, t):
    ks = [0 for i in range(t)]
    ks[0] = k[len(k)-1]
    l = list(reversed(k[:len(k)-1]))
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i)
    return(ks)

def encrypt(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round((x,y), k)
    return(x, y)

def decrypt(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k)
    return(x,y)

def convert_to_binary(arr):
  X = np.zeros((2* WORD_SIZE(),len(arr[0])),dtype=np.uint8)
  for i in range(2* WORD_SIZE()):
    index = i // WORD_SIZE()
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
    X[i] = (arr[index] >> offset) & 1
  return(X)

def make_train_data(n, nr):
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
    ks = expand_key(keys, nr)
    plainl = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plainr = np.frombuffer(urandom(2*n),dtype=np.uint16)
    Xreall = np.zeros((16,n),dtype=np.uint16)
    Xrealr = np.zeros((16,n),dtype=np.uint16)
    Xrandl = np.zeros((16,n),dtype=np.uint16)
    Xrandr = np.zeros((16,n),dtype=np.uint16)
    Yreal = np.ones(n)
    Yrand = np.zeros(n)

    for i in range(16):
        Xreall[i]=plainl+i
        Xrealr[i]=plainr
        Xrandl[i] = np.frombuffer(urandom(2*n),dtype=np.uint16)
        Xrandr[i] = np.frombuffer(urandom(2*n),dtype=np.uint16)    
          
    creall,crealr=encrypt((Xreall,Xrealr),ks)
    crandl,crandr=encrypt((Xrandl,Xrandr),ks)
    Xreal = np.zeros((16,32,n),dtype=np.uint8)
    Xrand = np.zeros((16,32,n),dtype=np.uint8)

    for i in range(16):
        Xreal[i] = convert_to_binary([creall[i], crealr[i]])
        Xrand[i] = convert_to_binary([crandl[i], crandr[i]])

    Xreal=Xreal.reshape(512,n)
    Xrand=Xrand.reshape(512,n)

    X=np.concatenate((Xreal, Xrand), axis=1)
    X=X.transpose()
    Yreal = np.ones(n)
    Yrand = np.zeros(n)
    Y=np.concatenate((Yreal, Yrand))
    return(X,Y)

def grouped_convolution(y, nb_channels,ks):
       
        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        _d = nb_channels // cardinality
        groups = []
        for j in range(cardinality):
            groups.append(layers.Conv1D(_d, kernel_size=ks, padding='same')(y))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)
        

        return y
#make residual tower of convolutional blocks
def make_resnet(num_words=2,multiset=16, num_filters=256, num_outputs=1, d1=512, d2=512, word_size=16, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid'):
  #Input and preprocessing layers
  inp = Input(shape=(num_words * word_size *multiset ,))
  rs = Reshape((num_words*multiset, word_size))(inp)
  perm = Permute((2,1))(rs)
  #add a single residual layer that will expand the data to num_filters channels
  #this is a bit-sliced layer
  input= Conv1D(num_filters , kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
  import tensorflow as tf 
  #add residual blocks
  for i in range(depth):
    
    conv1 = BatchNormalization()(input)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1)
    input = tf.keras.layers.Concatenate()([conv2, input])  
    num_filters += 16
    #add prediction head
  flat1 = Flatten()(input)
  dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1)
  dense1 = BatchNormalization()(dense1)
  dense1 = Activation('relu')(dense1)
  dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
  dense2 = BatchNormalization()(dense2)
  dense2 = Activation('relu')(dense2)
  dense2=Dropout(0.2)(dense2)
  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
  model = Model(inputs=inp, outputs=out)
  return(model)

def train_speck_distinguisher(num_epochs, num_rounds, depth):
    #create the network
    net = make_resnet(depth=depth, reg_param=10**-5)
    net.compile(optimizer='adam',loss='mse',metrics=['acc'])
    #generate training and validation data
    X, Y = make_train_data(2**20,num_rounds)
    X_eval, Y_eval = make_train_data(2**16, num_rounds)
    #set up model checkpoint
    #check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5');
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001))
    #train and evaluate
    #check = make_checkpoint(wdir+'present'+'8round'+'weights.{epoch:02d}-{val_acc:.2f}.hdf5')
    from keras.models import model_from_json
# serialize model to json
    json_model = net.to_json()
    #save the model architecture to JSON file
    #with open(wdir+'speck.json7', 'w') as json_file:
        #json_file.write(json_model)
    #saving the weights of the model
    #net.save_weights(wdir+'speck_weights7.h5')
    #net.load_weights('/content/gdrive/My Drive/b/5roundweights.09-0.50.hdf5')
    h = net.fit(X,Y,epochs=num_epochs,batch_size=bs,validation_data=(X_eval, Y_eval), callbacks=[lr])
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    return(net, h)
    
train_speck_distinguisher(50,num_rounds=3,depth=10)
