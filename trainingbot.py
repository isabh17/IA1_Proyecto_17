import random
import json
import pickle
import numpy as np
import tensorflow as tf
import os

import nltk
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios de nltk si es necesario
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Cargar los intents desde el archivo JSON
intents = json.loads(open('intents2.json', encoding='utf-8').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',', '¿', '¡']

# Procesar los datos de entrenamiento
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

# Guardar palabras y clases para su uso posterior
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Crear datos de entrenamiento
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    bag = [1 if word in wordPatterns else 0 for word in words]

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

# Dividir en características (X) y etiquetas (Y)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

# Compilar el modelo
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar el modelo
hist = model.fit(np.array(trainX), np.array(trainY), epochs=300, batch_size=8, verbose=1)
model.save('chatbot_model.h5', hist)
if os.path.exists('chatbot_model.h5'):
    print("Modelo guardado exitosamente.")
else:
    print("Error al guardar el modelo.")