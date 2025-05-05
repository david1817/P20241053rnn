import tensorflow as tf
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Inicializar el tokenizador y el codificador de etiquetas
tokenizer = None
label_encoder = None

def load_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error al cargar el modelo desde {model_path}: {e}")
        return None

def load_tokenizer(tokenizer_path):
    global tokenizer
    try:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
    except Exception as e:
        print(f"Error al cargar el tokenizador desde {tokenizer_path}: {e}")
        tokenizer = None
    return tokenizer

def load_label_encoder(label_encoder_path):
    global label_encoder
    try:
        with open(label_encoder_path, 'rb') as handle:
            label_encoder = pickle.load(handle)
    except Exception as e:
        print(f"Error al cargar el codificador de etiquetas desde {label_encoder_path}: {e}")
        label_encoder = None
    return label_encoder

def preprocess_input(pregunta, tokenizer, max_length=20):
    pregunta = pregunta.lower()
    secuencia = tokenizer.texts_to_sequences([pregunta])
    return pad_sequences(secuencia, maxlen=max_length, padding='post')

def decode_prediction(prediction, label_encoder, threshold=0.5):
    """
    Decodifica la predicción del modelo en una respuesta legible si supera el umbral.
    
    :param prediction: Predicción del modelo en formato numérico.
    :param label_encoder: Codificador de etiquetas utilizado para transformar las respuestas.
    :param threshold: Umbral de confianza para aceptar la predicción.
    :return: Respuesta legible si la predicción supera el umbral, de lo contrario, un mensaje de error.
    """
    max_prob = np.max(prediction)
    if max_prob >= threshold:
        return label_encoder.inverse_transform([np.argmax(prediction)])[0]
    else:
        return "Lo siento, no dispongo de dicha información."
