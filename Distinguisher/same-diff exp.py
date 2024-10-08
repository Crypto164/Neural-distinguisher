import numpy as np
from os import urandom
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras.regularizers import l2
import matplotlib.pyplot as plt
import time

bs = 5000

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)

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
    blindl = np.frombuffer(urandom(2*n),dtype=np.uint16)
    blindr = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plainl = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plainr = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain2l = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain2r = np.frombuffer(urandom(2*n),dtype=np.uint16)
	  
	
    Xreall = np.zeros((16,n),dtype=np.uint16)
    Xrealr = np.zeros((16,n),dtype=np.uint16)
    Xrandl = np.zeros((16,n),dtype=np.uint16)
    Xrandr = np.zeros((16,n),dtype=np.uint16)

    for i in range(16):
        Xreall[i]=plainl+i
        Xrealr[i]=plainr
        Xrandl[i] = plain2l+i
        Xrandr[i] = plain2r   
          
    creall,crealr=encrypt((Xreall,Xrealr),ks)
    crandl,crandr=encrypt((Xrandl,Xrandr),ks)

    for i in range(16):
        crandl[i] = crandl[i]^blindl
        crandr[i] = crandr[i]^blindr

			
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

#make residual tower of convolutional blocks
def make_resnet(num_words=2,multiset=16, num_filters=256, num_outputs=1, d1=512, d2=512, word_size=16, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid'):
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
  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
  model = Model(inputs=inp, outputs=out)
  return(model)

def train_speck_distinguisher(num_epochs, num_rounds, depth):
    # create the network
    net = make_resnet(depth=depth, reg_param=10 ** -5)
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])

    # generate training and validation data and test data
    X, Y = make_train_data(2 ** 15, num_rounds)
    X_eval, Y_eval = make_train_data(2 ** 14, num_rounds)
    X_test, Y_test = make_train_data(2 ** 14, num_rounds)

    # create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))

    # train and evaluate
    h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr])
    loss, accuracy = net.evaluate(X_test, Y_test)

    print("\nWhen training for a", num_rounds, "round SPECK ", num_epochs, "epochs:")
    print("\nBest validation accuracy: ", np.max(h.history['val_acc']))
    print('\nTest loss:', loss)
    print('\nTest accuracy:', accuracy)

    # f = open(save_path + "result_for_lyu_train_SPECK.txt", "a")
    f = open("./result_for_lyu_train_SPECK.txt", "a")
    f.write("\nWhen training for a " + str(num_rounds) + "-round SPECK " + str(num_epochs) + " epochs:")
    f.write("\nBest validation accuracy: " + str(np.max(h.history['val_acc'])))
    f.write('\nTest loss: ' + str(loss))
    f.write('\nTest accuracy: ' + str(accuracy))
    f.close()

    return (net, h)


# test(4,5,6)
# save_path = "./results_for_SPECK/"

time_start = time.time()

_, history = train_speck_distinguisher(10, num_rounds=3, depth=10)

time_end = time.time()
total_time = time_end - time_start
print('\nTotal training time is: %.2f seconds.' % total_time)

# f = open(save_path + "result_for_lyu_train_SPECK.txt","a")
f = open("./result_for_lyu_train_SPECK.txt", "a")
f.write('\nTotally training time is: %.2f seconds.\n' % total_time)
f.close()

acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(6, 4))
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
# plt.savefig(fname = save_path + "Training_10r_SPECK_10_epochs_"+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+".png")
plt.savefig(fname="./Training_5r_SPECK_10_epochs_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".png")
plt.show()
