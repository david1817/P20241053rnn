import json
import random

def get_data(file_path):
    """
    Carga datos desde un archivo JSON.

    :param file_path: Ruta al archivo JSON.
    :return: Datos cargados desde el archivo.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: El archivo {file_path} no existe.")
        return None
    except json.JSONDecodeError:
        print(f"Error: El archivo {file_path} no es un JSON válido.")
        return None

def split_data(data, train_ratio=0.8, shuffle=True):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    :param data: Lista de datos a dividir.
    :param train_ratio: Proporción de datos para entrenamiento (entre 0 y 1).
    :param shuffle: Si es True, mezcla los datos antes de dividir.
    :return: Una tupla (train_data, test_data).
    """
    if shuffle:
        random.shuffle(data)
    
    num_samples = len(data)
    train_size = int(num_samples * train_ratio)
    
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    return train_data, test_data
