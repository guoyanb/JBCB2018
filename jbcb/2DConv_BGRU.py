import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Embedding, LSTM, Dense, merge, Convolution2D, GRU, TimeDistributedDense, Reshape,MaxPooling2D,Convolution1D,BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping ,ModelCheckpoint


main_input = Input(shape=(700,), dtype='int32', name='main_input')
#main_input = Masking(mask_value=23)(main_input)
x = Embedding(output_dim=21, input_dim=21, input_length=700)(main_input)
auxiliary_input = Input(shape=(700,21), name='aux_input')  #24
#auxiliary_input = Masking(mask_value=0)(auxiliary_input)
input_feature = merge([x, auxiliary_input], mode='concat', concat_axis=-1)

c_input = Reshape((700, 42, 1))(input_feature)
# print x.get_shape()
# x=BatchNormalization()(x)
c_output = Convolution2D(42,3,3,activation='relu', border_mode='same', W_regularizer=l2(0.001))(c_input)
print 'c_output', c_output.get_shape()

c_output = Reshape((700,42*42))(c_output)

c_output = Dropout(0.5)(c_output)

d_output = Dense(400, activation='relu')(c_output)

####### 2 BGRU Recurrent Neural networks##########
f1 = GRU(output_dim=200,return_sequences=True, activation='tanh', inner_activation='sigmoid',dropout_W=0.5,dropout_U=0.5)(d_output)
f2 = GRU(output_dim=200, return_sequences=True, activation='tanh', inner_activation='sigmoid', go_backwards=True,dropout_W=0.5,dropout_U=0.5)(d_output)
# f = merge([d,e], mode='sum',concat_axis=2)
# f=Dropout(0.5)(f)
f3 = GRU(output_dim=200, return_sequences=True, activation='tanh', inner_activation='sigmoid', dropout_W=0.5,dropout_U=0.5)(f1)
f4 = GRU(output_dim=200, return_sequences=True, activation='tanh', inner_activation='sigmoid', go_backwards=True,dropout_W=0.5,dropout_U=0.5)(f2)

cf_feature = merge([f3, f4 , d_output], mode='concat',concat_axis=2)
cf_feature = Dropout(0.4)(cf_feature)
f_input = Dense(600,activation='relu')(cf_feature)
# f_input = TimeDistributedDense(200,activation='relu', W_regularizer=l2(0.001))(f_input)
main_output = TimeDistributedDense(8,activation='softmax', name='main_output')(f_input)
# auxiliary_output = TimeDistributedDense(4,activation='softmax', name='aux_output')(f_input)
model = Model(input=[main_input, auxiliary_input], output=[main_output])
adam = Adam(lr=0.003)
model.compile(optimizer=adam,
              loss={'main_output': 'categorical_crossentropy'},
              # loss_weights={'main_output': 1},
              metrics=['weighted_accuracy'])
model.summary()

# print "####### look at data's shape#########"
# print train_hot.shape, trainpssm.shape, trainlabel.shape, test_hot.shape, testpssm.shape,testlabel.shape, val_hot.shape,valpssm.shape,vallabel.shape

earlyStopping = EarlyStopping(monitor='val_weighted_accuracy', patience=5, verbose=1, mode='auto')
load_file = "./model/2c(3(21)42-200-0.003-400-600)-de-2LSTM-CB513.h5"
checkpointer = ModelCheckpoint(filepath=load_file,verbose=1,save_best_only=True)
history=model.fit({'main_input': traindatahot, 'aux_input': trainpssm},
          {'main_output': trainlabel},validation_data=({'main_input': valdatahot, 'aux_input': valpssm},{'main_output': vallabel}),
        nb_epoch=200, batch_size=64, callbacks=[checkpointer, earlyStopping], verbose=2, shuffle=True)


model.load_weights(load_file)

print "#########evaluate:##############"
score = model.evaluate({'main_input': testdatahot, 'aux_input': testpssm},{'main_output': testlabel},verbose=2,batch_size=1)
print score 
print 'test loss:', score[0]
print 'test accuracy:', score[1]