
import numpy as np
from os import urandom


def expand_key(k, t,n):
    ks = np.zeros((t,4,16,n),dtype=np.uint8)
    for i in range(0,t):
        for j in range(4):
            ks[i][j]=k[j]
        for j in range(4):
              k[0][j]=k[2][j]^k[0][j]^(k[1][j]&k[3][j])^(k[0][j]&k[1][j]&k[2][j])^(k[1][j]&k[2][j])^(k[0][j]&k[3][j])
              k[1][j]=1^k[1][j]^k[0][j]^(k[2][j]&k[3][j])^(k[1][j]&k[2][j]&k[3][j])^(k[1][j]&k[3][j])^(k[1][j]&k[2][j])^(k[0][j]&k[1][j])
              k[2][j]=1^k[1][j]^k[2][j]^(k[0][j]&k[2][j])^k[3][j]
              k[3][j]=k[0][j]^k[1][j]^(k[2][j]&k[3][j])^k[3][j]
          
        for i in range(16):
              k[0][i]=k[0][(i+8)%16]^k[1][i];k[1][i]=k[2][i];k[2][i]=k[3][i];k[3][i]=k[3][(i+4)%16]^k[4][i];k[4][i]=k[0][i] 
                
          
    return(ks)


def encrypt_one_round(p, k,n):
    Y = np.zeros((4,16,16,n),dtype=np.uint8)
    Z = np.zeros((4,16,16,n),dtype=np.uint8)
    
    for j in range(16):
        for i in range(4):
          for t in range(16):
              p[i][t][j] = p[i][t][j]^k[i][t]

    Z[0]=p[2]^p[0]^(p[1]&p[3])^(p[0]&p[1]&p[2])^(p[1]&p[2])^(p[0]&p[3]);Z[1]=1^p[1]^p[0]^(p[2]&p[3])^(p[1]&p[2]&p[3])^(p[1]&p[3])^(p[1]&p[2])^(p[0]&p[1])
    Z[2]=1^p[1]^p[2]^(p[0]&p[2])^p[3];Z[3]=p[0]^p[1]^(p[2]&p[3])^p[3]
 
    for i in range(16):
          Y[0][i] =Z[0][i]; Y[1][(i+1)%16]=Z[1][i];Y[2][(i+12)%16]=Z[2][i];Y[3][(i+13)%16]=Z[3][i]

    return Y

def encrypt(p, ks,n,nr):
    y=p
    for k in ks[0:nr-1]:
        y = encrypt_one_round(y, k,n)

    for j in range(16):
          for i in range(4):
             for t in range(16):
                      y[i][t][j] = y[i][t][j]^ks[nr][i][t]
    
    return(y)


def make_train_data(n, nr):
    
    K = np.zeros((5,16,n),dtype=np.uint8)
    
    Xreal = np.zeros((4,16,16,n),dtype=np.uint8)
    Xrand = np.zeros((4,16,16,n),dtype=np.uint8)
    
    Yreal = np.ones(n)
    Yrand = np.zeros(n)
    
    
    
    arr=[[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1 ]]


    Active=np.array(arr).transpose()


    for j in range(16):
        for i in range(5):
            K[i][j] = np.frombuffer(urandom(n), dtype=np.uint8); K[i][j] = K[i][j] & 1
    ks=expand_key(K,nr+1,n)
    
    
    for j in range(16):
        for i in range(4):
          Xreal[0][i][j] = Active[i][j]

    for i in range(4,16):
            Xreal[0][i] = 0
  
    for i in range(1,4):
           Xreal[i] = 0
    
    for j in range(16):
          for t in range(0,4):
                for i in range(0,16):
                      rand = np.frombuffer(urandom(n), dtype=np.uint8); rand = rand & 1
                      Xrand[t][i][j] = rand
        
        
    creal=encrypt(Xreal, ks,n,nr)
    crand=encrypt(Xrand, ks,n,nr)
    
  
    creal=creal.reshape(1024,n)
    crand=crand.reshape(1024,n)

    cipher=np.concatenate((creal, crand), axis=1)
    cipher=cipher.transpose()
    Y=np.concatenate((Yreal, Yrand))
    

    
    
    return(cipher,Y)



 import numpy as np
from pickle import dump

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2
from keras.layers import Dropout

import tensorflow as tf 

bs = 5000;

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)

#make residual tower of convolutional blocks
def make_resnet(num_words=16,multiset=16, num_filters=512, num_outputs=1, d1=1024, d2=1024, word_size=4, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid'):
  #Input and preprocessing layers
  inp = Input(shape=(num_words * word_size *multiset ,))
  rs = Reshape((num_words*multiset, word_size))(inp)
  perm = Permute((2,1))(rs)
  #add a single residual layer that will expand the data to num_filters channels
  #this is a bit-sliced layer
  input= Conv1D(num_filters , kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
  #conv0 = BatchNormalization()(conv0)
  #conv0 = Activation('relu')(conv0)
  #add residual blocks
  #shortcut = conv0
  for i in range(depth):
    #conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
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

from os import urandom
from keras.models import model_from_json

import numpy as np

from random import randint


from keras.models import Model
from sklearn.linear_model import Ridge

from random import sample, randint
from collections import defaultdict
from math import log2

linear_model = Ridge(alpha=0.01);

def train_preprocessor(n, nr, epochs):
  net = make_resnet(depth=1)
  net.compile(optimizer='adam',loss='mse',metrics=['acc'])
  #create a random input difference
  X,Y = make_train_data(n, nr)
  net.fit(X,Y,epochs=epochs, batch_size=5000,validation_split=0.1)
  net_pp = Model(inputs=net.layers[0].input, outputs=net.layers[-2].output)
  return(net_pp)

net_pp = train_preprocessor(2**20,6,1)
def evaluate_diff(net_pp, nr=5, n=1000):

  X1,Y1 = make_train_data(n, nr)
  X2,Y2 = make_train_data(n, nr)
  Z1 = net_pp.predict(X1,batch_size=5000)
  Z2 = net_pp.predict(X2,batch_size=5000)
  #perceptron.fit(Z[0:n],Y[0:n]);
  linear_model.fit(Z1[:],Y1[:])
  #val_acc = perceptron.score(Z[n:],Y[n:]);
  Y = linear_model.predict(Z2[:])
  Ybin = (Y > 0.5)
  val_acc = float(np.sum(Ybin == Y2[:])) / (2*n)
  return(val_acc)

val_acc =evaluate_diff(net_pp, nr=7, n=100)
print(str(val_acc))

