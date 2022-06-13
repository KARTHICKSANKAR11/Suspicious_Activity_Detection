#Implementation of CNN Algorithm

#import the libraries
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

import tensorflow
from warnings import filterwarnings

#CNN algorithm for classification
classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2)) 
classifier.add(Conv2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
#adam = , beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)
classifier.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])


from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagenerate_train = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)
datagenerate_test = ImageDataGenerator(rescale=1./255)

print(datagenerate_train)

print(datagenerate_test)

#Training Set
training_set = datagenerate_train.flow_from_directory(r'E:\Study\Project\Suspicious2\Train',
                                             target_size=(64,64),
                                             batch_size=32,
                                             class_mode='binary')
#Validation Set
Validation_set = datagenerate_test.flow_from_directory(r'E:\Study\Project\Suspicious2\Test',
                                           target_size=(64,64),
                                           batch_size = 32,
                                           class_mode='binary',
                                           shuffle=False)

classifier.fit_generator(training_set,
                        steps_per_epoch=10, 
                        epochs = 25,
                        validation_data = Validation_set,
                        validation_steps = 20,
                        )

#Save the weight file 
classifier.save(r'E:\Study\Project\Suspicious2/Trianed_model.h5')

