# Importing keras libraries and packages
# on utilise la librairie keras car est very famous dans le deep learning models, keras contient deux classe la sequentel et la fonctionel
# on utilise sequentiel car adding layer step by step et la sortie du precedente couche et utilisé comme entré de la suivante

from keras.models import Sequential
# mes 4 couches
#convulution
from keras.layers import Convolution2D
#pour applatir l'image
from keras.layers import Flatten
#dense represente la couche des neurones artificiel
from keras.layers import Dense
# le pooling pour encore rendre limage plus petite et les calcul plus rapide
from keras.layers import MaxPooling2D

# step1 Initializing CNN
# j'initialise mon objet ici mon objet est classifier
classifier = Sequential()

# step2 adding 1st Convolution layer and Pooling layer
# jutilise la methode add pour ajouter les couches les unes apres les autres
#dans convolution il ya bcp de parametre et dommage le 32 le son a coupé pour le premier 32 type de filtre pour extraire different feature , le (3,3) est la taille du filtre
#input shape parametre qui decide la taille des image que je vais introduire dans mon reseau le 64 x 64 largeur hauteur, et le 3 color scal rgb
#activation fonction , on va utilisé la non linearité entre les image et relu function est la meilleure des fonction pour la non linearité
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# la taille de polling metrix (2,2)
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step3 adding 2nd convolution layer and polling layer
classifier.add(Convolution2D(32, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step4 Flattening the layers
#je transform mon image en vecteur
classifier.add(Flatten())

# step5 Full_Connection
# je passe les data au reseau de neuronnes
# units represente le nombre de neuronnes dans le resau
#relu toujours pour la non linearité
classifier.add(Dense(units=32, activation='relu'))

classifier.add(Dense(units=64, activation='relu'))

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dense(units=256, activation='relu'))

classifier.add(Dense(units=256, activation='relu'))
# classifier.add(Dense(units=512, activation='relu'))

# le nombre de nouronnes est seulment de 6 car c'est le nombre de mes sorties et il est du au fait que jai que 3 fruit chacun d'entre eux  possede 2 category donc au final j'ai 6 classes
# la fonction activation ici est softmax car softmax est utilise pour categorical classification
classifier.add(Dense(units=6, activation='softmax'))


# step6 Compiling CNN
# optimizer est utilisé pour optimisé notre training effecacité on utilise adam car il est adapté l'apprentisage
# loss similarly to softmax , utilisé calcul les erreur et les actuel result pour entrainer laccuracy en gros cest chaud de le comprendre
#metrics on utilise accuracy as performance metrics
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# je reprends a la minute 31
# step7 Fitting CNN to images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,  # To rescaling the image in range of [0,1]
                                   shear_range=0.2,  # To randomly shear the images
                                   zoom_range=0.2,  # To randomly zoom the images
                                   horizontal_flip=True)  # for randomly flipping half of the images horizontally

test_datagen = ImageDataGenerator(rescale=1. / 255)
print("\nTraining the data...\n")
training_set = train_datagen.flow_from_directory('train',
                                                 target_size=(64, 64),
                                                 batch_size=16,  # Total no. of batches
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('test',
                                            target_size=(64, 64),
                                            batch_size=16,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                         # Total training images
                         # Total no. of epochs
                         validation_data=test_set)  # Total testing images

# step8 saving model

classifier.save("model.h5")
