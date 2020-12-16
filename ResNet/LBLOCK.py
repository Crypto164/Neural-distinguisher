import numpy as np
from os import urandom
wdir = '/content/drive/My Drive/b/'


def expand_key(k, t,n):
    ks = np.zeros((t,32,n),dtype=np.uint8)
       
    for i in range(0,t):
        ks[i][0:32] = k[48:80]
        for j in range(80):
            k[j]=k[(j+29)%80]
		
		
 
        k[76]=1^k[76]^k[76+1]^k[76+3]^(k[76+1]&k[76+2])^(k[76+1]&k[76+3])
        k[76+1]=k[76]^k[76+1]^k[76+2]^k[76+3]^(k[76]&k[76+1])
        k[76+2]=1^k[76]^k[76+3]^(k[76+1]&k[76+3])^(k[76+1]&k[76+2])^(k[76]&k[76+1]&k[76+2])^(k[76]&k[76+1]&k[76+3])^(k[76]&k[76+3])^(k[76]&k[76+1])
        k[76+3]=1^k[76]^(k[76]&k[76+1]&k[76+3])^(k[76]&k[76+2])^(k[76+1]&k[76+3])^(k[76+2]&k[76+3])
	 
        k[72]=1^k[72+3]^(k[72+1]&k[72+2])^(k[72+1]&k[72+3])^(k[72]&k[72+3])^(k[72]&k[72+1])^(k[72]&k[72+1]&k[72+2])^(k[72]&k[72+1]&k[72+3])
        k[72+1]=k[72+1]^k[72+2]^k[72+3]^(k[72+2]&k[72+3])^(k[72+1]&k[72+3])^(k[72]&k[72+2])^(k[72]&k[72+1])^(k[72]&k[72+1]&k[72+3])
        k[72+2]=k[72]^k[72+1]^k[72+2]^k[72+3]^(k[72]&k[72+1])
        k[72+3]=k[72]^k[72+1]^k[72+3]^(k[72+1]&k[72+2])^(k[72+1]&k[72+3])
        
        if i==1:
           k[46]=k[46]^1
        elif i==2:
           k[47]=k[47]^1
        elif i==3:
            k[47]=k[47]^1
            k[46]=k[46]^1
        elif i==4:
            k[48]=k[48]^1
        elif i==5:
            k[48]=k[48]^1
            k[46]=k[46]^1
        elif i==6:
            k[48]=k[48]^1;k[47]=k[47]^1
        elif i==7:
            k[48]=k[48]^1;k[47]=k[47]^1;k[46]=k[46]^1
        elif i==8:
            k[49]=k[49]^1
        elif i==9:
            k[49]=k[49]^1;k[46]=k[46]^1
	    
		
            
          
    return(ks)


def encrypt_one_round(p, k,n):
      Y = np.zeros((64,16,n),dtype=np.uint8)
      Z = np.zeros((32,16,n),dtype=np.uint8)
      U = np.zeros((32,16,n),dtype=np.uint8)
      W = np.zeros((32,16,n),dtype=np.uint8)
	
	
      for j in range(16):
        for i in range(32):
          p[i][j] = p[i][j]^k[i]
    
      for j in range(16):
          for i in range(32,64):
            W[i-32][j] = p[i][j]
      
      for j in range(16):
          for i in range(32):
            W[i][j] = W[(i+8)% 32][j]

      for i in range(0,1):  
          Z[i+3]=p[i]^p[i+1]^p[i+2]^p[i+3]^(p[i]&p[i+1])
          Z[i+2]=1^p[i]^p[i+1]^p[i+3]^(p[i+1]&p[i+2])^(p[i+1]&p[i+3])
          Z[i+1]=p[i]^p[i+3]^(p[i+1]&p[i+3])^(p[i+1]&p[i+2])^(p[i]&p[i+3])^(p[i]&p[i+1])^(p[i]&p[i+1]&p[i+3])^(p[i]&p[i+1]&p[i+2])^1
          Z[i]=p[i]^(p[i]&p[i+1]&p[i+3])^(p[i]&p[i+2])^(p[i+1]&p[i+3])^(p[i+2]&p[i+3])^1
		
		
      for i in range(4,5):  
            Z[i]=(p[i]&p[i+1]&p[i+3])^(p[i]&p[i+1])^(p[i]&p[i+2])^(p[i+1]&p[i+3])^(p[i+2]&p[i+3])^p[i+1]^p[i+2]^p[i+3]
            Z[i+1]=(p[i]&p[i+1]&p[i+2])^(p[i]&p[i+1]&p[i+3])^(p[i]&p[i+1])^(p[i]&p[i+3])^(p[i+1]&p[i+2])^(p[i+1]&p[i+3])^p[i+3]^1
            Z[i+2]=p[i]^p[i+1]^p[i+2]^p[i+3]^(p[i]&p[i+1])
            Z[i+3]=p[i]^p[i+1]^p[i+3]^(p[i+1]&p[i+2])^(p[i+1]&p[i+3])


		
      for i in range(8,9):  
            Z[i]=(p[i+1]&p[i+3])^(p[i+1]&p[i+2])^p[i+1]^p[i+3]^p[i]
            Z[i+1]=(p[i]&p[i+1]&p[i+3])^(p[i]&p[i+1])^(p[i]&p[i+2])^(p[i+1]&p[i+3])^(p[i+2]&p[i+3])^p[i+1]^p[i+2]^p[i+3]
            Z[i+2]=p[i]^p[i+1]^p[i+2]^p[i+3]^(p[i]&p[i+1])
            Z[i+3]=1^p[i+3]^(p[i+1]&p[i+3])^(p[i+1]&p[i+2])^(p[i]&p[i+3])^(p[i]&p[i+1])^(p[i]&p[i+1]&p[i+3])^(p[i]&p[i+1]&p[i+2])
    

 
      for i in range(12,13):  
            Z[i]=p[i]^p[i+2]^(p[i]&p[i+1]&p[i+2])^(p[i]&p[i+1]&p[i+3])^(p[i]&p[i+1])^(p[i]&p[i+2])^(p[i]&p[i+3])
            Z[i+1]=(p[i]&p[i+1]&p[i+2])^(p[i+1]&p[i+3])^p[i+1]^p[i+2]^1
            Z[i+2]=(p[i]&p[i+1]&p[i+3])^(p[i]&p[i+1])^(p[i]&p[i+2])^(p[i]&p[i+3])^(p[i+1]&p[i+3])^(p[i+2]&p[i+3])^p[i+2]^p[i]^p[i+1]^1
            Z[i+3]=1^p[i+1]^p[i+2]^p[i+3]^(p[i]&p[i+1])
        
      
      for i in range(16,17):  
            Z[i]=p[i]^p[i+1]^p[i+3]^1^(p[i+1]&p[i+2])^(p[i+1]&p[i+3])
            Z[i+1]=(p[i]&p[i+1]&p[i+3])^(p[i]&p[i+2])^p[i]^1^(p[i+1]&p[i+3])^(p[i+2]&p[i+3])
            Z[i+2]=(p[i]&p[i+1]&p[i+2])^(p[i]&p[i+1]&p[i+3])^(p[i]&p[i+1])^(p[i]&p[i+3])^(p[i+1]&p[i+2])^(p[i+1]&p[i+3])^p[i]^p[i+3]^1
            Z[i+3]=p[i]^p[i+1]^p[i+2]^p[i+3]^(p[i]&p[i+1])

    

      for i in range(20,21):  
            Z[i]=p[i+2]^p[i+1]^p[i+3]^(p[i]&p[i+1])^(p[i]&p[i+2])^(p[i+1]&p[i+3])^(p[i+2]&p[i+3])^(p[i]&p[i+1]&p[i+3])
            Z[i+1]=p[i]^p[i+1]^p[i+3]^(p[i+1]&p[i+2])^(p[i+1]&p[i+3])
            Z[i+2]=(p[i]&p[i+1]&p[i+2])^(p[i]&p[i+1]&p[i+3])^(p[i]&p[i+1])^(p[i]&p[i+3])^(p[i+1]&p[i+2])^(p[i+1]&p[i+3])^p[i+3]^1
            Z[i+3]=p[i]^p[i+1]^p[i+2]^p[i+3]^(p[i]&p[i+1])
        


      for i in range(24,25):  
            Z[i]=1^p[i]^p[i+1]^p[i+2]^(p[i]&p[i+1])^(p[i]&p[i+2])^(p[i]&p[i+3])^(p[i+1]&p[i+3])^(p[i]&p[i+1]&p[i+3])^(p[i+2]&p[i+3])
            Z[i+1]=(p[i]&p[i+1]&p[i+2])^(p[i]&p[i+1]&p[i+3])^p[i]^p[i+2]^(p[i]&p[i+2])^(p[i]&p[i+1])^(p[i+1]&p[i+2])^(p[i+1]&p[i+3])
            Z[i+2]=p[i+1]^p[i+2]^p[i+3]^(p[i]&p[i+1])^1
            Z[i+3]=p[i]^p[i+1]^p[i+2]^(p[i+1]&p[i+2])^(p[i+1]&p[i+3])^1
        
	
 
      for i in range(28,29):  
            Z[i]=1^p[i]^(p[i]&p[i+2])^(p[i+1]&p[i+3])^(p[i+2]&p[i+3])^(p[i]&p[i+1]&p[i+3])
            Z[i+1]=1^(p[i]&p[i+1]&p[i+2])^(p[i]&p[i+1]&p[i+3])^p[i]^p[i+3]^(p[i]&p[i+1])^(p[i]&p[i+3])^(p[i+1]&p[i+2])^(p[i+1]&p[i+3])
            Z[i+2]=p[i]^p[i+1]^p[i+2]^p[i+3]^(p[i]&p[i+1])
            Z[i+3]=1^p[i]^p[i+1]^p[i+3]^(p[i+1]&p[i+2])^(p[i+1]&p[i+3])
        
    
    
    
      for i in range(0,4):
            U[i+8] = Z[i]
      for i in range(4,8):
            U[i-4] = Z[i]
      for i in range(8,12):
            U[i+4] = Z[i]
      for i in range(12,16):
          U[i-8] = Z[i]
      for i in range(16,20):
          U[i+8] = Z[i]
      for i in range(20,24):
          U[i-4] = Z[i]
      for i in range(24,28):
          U[i+4] = Z[i]
      for i in range(28,32):
          U[i-8] = Z[i]
			
		
      for j in range(16):
          for i in range(32):
              Y[i][j] = W[i][j]^U[i][j]
          
      for j in range(16):
          for i in range(32):
              Y[i+32][j] = p[i][j]
          

      return(Y) 

def encrypt(p, ks,n,nr):
    y=p
    for k in ks[0:nr-1]:
        y = encrypt_one_round(y, k,n)

    
    return(y)

def make_train_data(n, nr):
    
    K = np.zeros((80,n),dtype=np.uint8)
    
    Xreal = np.zeros((64,16,n),dtype=np.uint8)
    Xrand = np.zeros((64,16,n),dtype=np.uint8)
    
    Yreal = np.ones(n)
    Yrand = np.zeros(n)
    
    
    
    arr=[[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1 ]]

 
    
    

    Active=np.array(arr).transpose()


    for i in range(80):
        K[i] = np.frombuffer(urandom(n), dtype=np.uint8); K[i] = K[i] & 1
    ks=expand_key(K,nr,n)
    
    
    for j in range(16):
        for i in range(4):
          Xreal[i][j] = Active[i][j]

    for i in range(4,64):
        Xreal[i] = 0
    
    for j in range(16):
        for i in range(0,64):
            rand = np.frombuffer(urandom(n), dtype=np.uint8); rand = rand & 1
            Xrand[i][j] = rand
        
        
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



bs = 5000;

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)

def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True)
  return(res)

#make residual tower of convolutional blocks
def make_resnet(num_words=16,multiset=16, num_filters=512, num_outputs=1, d1=1024, d2=1024, word_size=4, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid'):
  #Input and preprocessing layers
  inp = Input(shape=(num_words * word_size *multiset ,))
  rs = Reshape((num_words*multiset, word_size))(inp)
  perm = Permute((2,1))(rs)
  #add a single residual layer that will expand the data to num_filters channels
  #this is a bit-sliced layer
  conv0 = Conv1D(num_filters , kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
  conv0 = BatchNormalization()(conv0)
  conv0 = Activation('relu')(conv0)
  #add residual blocks
  shortcut = conv0
  for i in range(depth):
    conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2=Dropout(0.2)(conv2)
    conv2 = Activation('relu')(conv2)
    shortcut = Add()([shortcut, conv2])
  #add prediction head
  flat1 = Flatten()(shortcut)
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

def train_lblock_distinguisher(num_epochs, num_rounds, depth):
    #create the network
    net = make_resnet(depth=depth, reg_param=10**-5)
    net.compile(optimizer='adam',loss='mse',metrics=['acc'])
    #generate training and validation data
    X, Y = make_train_data(2**20,num_rounds)
    X_eval, Y_eval = make_train_data(2**16, num_rounds)
    
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001))

    from keras.models import model_from_json

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


#print(np.load(wdir+'data_6r_attack7'+'.npy'))
#test(7,8)
train_lblock_distinguisher(500,num_rounds=11,depth=10)

