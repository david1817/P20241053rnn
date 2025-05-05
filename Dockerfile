# Utilizar una imagen base de Python
FROM python:3.11-slim

# Crear un directorio de trabajo
WORKDIR /app

# Copiar los archivos de requerimientos al contenedor
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación al contenedor
COPY . .

# Exponer el puerto en el que FastAPI estará escuchando
EXPOSE 8000

# Comando para iniciar el servidor de FastAPI usando Uvicorn con recarga automática
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
