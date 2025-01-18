import cv2
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.core import Flatten, Dense
from helpers import resize_to_fit

### Código pra corrigir um erro ao salvar o modelo treinado ###
import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__
# https://stackoverflow.com/questions/76650327/how-to-fix-cannot-import-name-version-from-tensorflow-keras
###############################################################

############### Código pra corrigir um erro ao treinar o modelo ###############
from tensorflow.python.keras.engine import data_adapter
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset
# https://stackoverflow.com/questions/77125999/google-colab-tensorflow-distributeddatasetinterface-error
# https://jcbsv.net/2024/04/19/fix-tensorflow-missing-attribute-problem/
###############################################################################

dados = []
rotulos = []
pasta_base_imagens = "base-letras"

imagens = paths.list_images(pasta_base_imagens) # Pega o caminho relativo de todas as imagens na pasta

for arquivo in imagens:
    rotulo = arquivo.split(os.path.sep)[-2] # Pega o nome da pasta em que a imagem está, que no caso indica qual é a letra
    imagem = cv2.imread(arquivo)  # Lê a imagem
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # Converte a imagem para escala de cinza
    imagem = resize_to_fit(imagem, 20, 20) # Redimensiona a imagem para 20x20 para que todas tenham o mesmo tamanho
    imagem = np.expand_dims(imagem, axis=2) # Adiciona uma dimensão para o canal da imagem, pois o Keras espera imagens coloridas com 3 canais
    rotulos.append(rotulo)
    dados.append(imagem)

dados = np.array(dados, dtype="float") / 255 # As imagens podem ter valores entre 0 e 255, então transforma esses valores para entre 0 e 1
rotulos = np.array(rotulos) # Transforma a lista de rótulos, do padrão do python para o numpy

(X_train, X_test, Y_train, Y_test) = train_test_split(dados, rotulos, test_size=0.25, random_state=0) # Divide os dados em treino e teste, 75% para treino e 25% para teste

lb = LabelBinarizer().fit(Y_train) # Entende quais são os rotulos e criar a estrutura de one-hot encoding, para as letras em números
Y_train = lb.transform(Y_train) # Transforma os rótulos de treino para o formato one-hot encoding
Y_test = lb.transform(Y_test) # Transforma os rótulos de teste para o formato one-hot encoding

with open('rotulos_modelo.dat', 'wb') as arquivo_pickle: # Salva o objeto lb no arquivo rotulos_modelo.dat, porque a resposta da IA será em números, então precisa saber qual letra corresponde a cada número pra fazer a tradução
    pickle.dump(lb, arquivo_pickle)

modelo = Sequential() # Cria um modelo sequencial, uma rede neural de várias camadas

# Cria a primeira camada da rede neural
modelo.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Cria a segunda camada da rede neural
modelo.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Cria a terceira camada da rede neural
modelo.add(Flatten())
modelo.add(Dense(500, activation="relu"))

modelo.add(Dense(62, activation="softmax")) # Cria a camada de saída da rede neural

modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) # Compila todas as camadas

modelo.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=62, epochs=10, verbose=1) # Treina o modelo

modelo.save('modelo_treinado.hdf5') # Salva o modelo treinado no modelo_treinado.hdf5