from fastapi import FastAPI, UploadFile
import tensorflow as tf
import os
import cv2
from tensorflow import keras
from keras.models import load_model
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import shutil

app = FastAPI()

# Configurar los orígenes permitidos
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8001",
    
]

# Configurar el middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/diagnostico")
async def diagnostico(imagen: UploadFile):
        # Aquí puedes procesar la imagen y generar la respuesta
        new_model = load_model(os.path.join('models','imageclassifier.h5'))
        new_model2 = load_model(os.path.join('models','imageclassifierojosano.h5'))
        new_model_cata = load_model(os.path.join('models','imageclassifiercatarata.h5'))
        #new_model_chala = load_model(os.path.join('models','imageclassifierchala.h5'))
        content = await imagen.read()  # Lee el contenido de la imagen
        nparr = np.frombuffer(content, np.uint8)  # Convierte los bytes en un array numpy
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decodifica la imagen con OpenCV
        resize = cv2.resize(img, (256, 256))
        yhatnew = new_model.predict(np.expand_dims(resize/255, 0))
        yhatnew2 = new_model2.predict(np.expand_dims(resize/255, 0))
        cata = new_model_cata.predict(np.expand_dims(resize/255, 0))
        #chala = new_model_chala.predict(np.expand_dims(resize/255, 0))
        if cata >= 0.97:
         return "La enfermedad es cataratas"
         
        if yhatnew2 >= 0.99 and yhatnew2 < 1:
            return "OJO SANO"

        if yhatnew > 0.5: 
            return "La prediccion es chalazion"
    
        return "ENFERMEDAD NO ENCONTRADA"
