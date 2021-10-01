import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets.mnist import load_data
import pickle


(x_train,y_train),(x_test,y_test)=load_data()   #Easy way to devide test train using keras

x_train=x_train/255.0
x_test=x_test/255.0
#print(x_train[0])
#plt.imshow(x_train[0])
#plt.show()

from keras import layers
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras .utils import to_categorical

y_train,y_test=to_categorical(y_train),to_categorical(y_test)    #One hot encoding

model=keras.Sequential()
model.add(layers.Flatten())
#model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer=Adam(),loss=categorical_crossentropy,metrics=["acc"])
model.fit(x_train,y_train,epochs=10,batch_size=200)

#model.save("ann_m")
#a=model.evaluate(x_test,y_test)        #Loss=0.06542830710135167  Accuracy=0.9799000024795532
#print(a)

pickle_out=open("trained_model.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close() 
