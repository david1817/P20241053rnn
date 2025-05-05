import tensorflow as tf
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Leer datos desde un archivo de texto
def cargar_datos(file_path):
    preguntas = []
    respuestas = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    pregunta, respuesta = line.strip().split('/', 1)
                    preguntas.append(pregunta)
                    respuestas.append(respuesta)
                except ValueError as ve:
                    print(f"Error al procesar la línea: {line.strip()} - {ve}")
    except FileNotFoundError:
        print(f"Error: El archivo {file_path} no se encontró.")
        return [], []
    except Exception as e:
        print(f"Error al leer el archivo {file_path}: {e}")
        return [], []
    return preguntas, respuestas

# Cargar datos
X, y = cargar_datos('datos.txt')

if not X or not y:
    print("No se cargaron datos. Asegúrate de que el archivo de texto esté correctamente formateado.")
else:
    # Crear y ajustar el tokenizador
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X)

    # Guardar el tokenizador
    with open('models/tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Preprocesar las preguntas
    secuencias = tokenizer.texts_to_sequences(X)
    max_length = 20  # Ajusta según la longitud máxima utilizada en el entrenamiento
    X_pad = pad_sequences(secuencias, maxlen=max_length, padding='post')

    # Preprocesar las respuestas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Guardar el codificador de etiquetas
    with open('models/label_encoder.pkl', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Crear un modelo simple
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=max_length),
        tf.keras.layers.SimpleRNN(128),
        tf.keras.layers.Dense(len(set(y_encoded)), activation='softmax')
    ])

    # Compilar y entrenar el modelo
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_pad, y_encoded, epochs=10000)

    # Guardar el modelo
    model.save('models/modelo.keras')
