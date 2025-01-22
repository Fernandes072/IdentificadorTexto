from tensorflow.python.keras.models import load_model
from helpers import resize_to_fit
import numpy as np
import cv2
import pickle
from tratar_imagem import tratar_imagem
from separar_palavras import separar_palavras
from separar_letras import separar_letras
import os
import shutil

############### Código pra corrigir um erro ao usar o modelo ###############
from tensorflow.python.keras.engine import data_adapter
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset
# https://stackoverflow.com/questions/77125999/google-colab-tensorflow-distributeddatasetinterface-error
# https://jcbsv.net/2024/04/19/fix-tensorflow-missing-attribute-problem/
###############################################################################

def identificar_texto():
    caminho_imagem = input('Digite o caminho da imagem: ')
    while (caminho_imagem != 'sair'):
        with open('rotulos_modelo.dat', 'rb') as arquivo_tradutor:
            lb = pickle.load(arquivo_tradutor) # Carrega as informações de tradução armazenadas no arquivo
        
        modelo = load_model('modelo_treinado.hdf5') # Carrega o modelo treinado

        if not os.path.exists('identificar'): # Cria a pasta identificar se ela não existir
            os.makedirs('identificar')
        shutil.copy(caminho_imagem, 'identificar/imagem.png') # Copia a imagem para a pasta identificar

        tratar_imagem('identificar', pasta_destino='identificar') # Trata a imagem da pasta identificar
        separar_palavras('identificar') # Separa as palavras da imagem tratada

        texto = ''
        palavras = os.listdir('palavras')
        for palavra in palavras:
            if os.path.exists('palavra'): # Apaga a pasta palavra se ela existir
                shutil.rmtree('palavra')
            os.makedirs('palavra') # Cria a pasta palavra
            shutil.copy(f'palavras/{palavra}', 'palavra/palavra.png') # Copia a imagem da pasta palavras para a pasta palavra
            separar_letras('palavra') # Separa as letras da palavra

            letras = os.listdir('letras')
            for letra in letras:
                letra = cv2.imread(f'letras/{letra}')  # Lê a imagem
                letra = cv2.cvtColor(letra, cv2.COLOR_BGR2GRAY) # Converte a imagem para escala de cinza
                letra = resize_to_fit(letra, 20, 20) # Redimensiona a imagem para 20x20
                letra = np.expand_dims(letra, axis=2) # Adiciona uma dimensão para o canal da imagem
                letra = np.expand_dims(letra, axis=0) # Adiciona outra dimensão para o canal da imagem

                letra_prevista = modelo.predict(letra) # Faz a previsão da letra
                letra_prevista = lb.inverse_transform(letra_prevista)[0] # Transforma a previsão da letra de números para letra
                texto += letra_prevista

            texto += ' '
        print(texto)

        print()
        caminho_imagem = input('Digite o caminho da imagem: ')

if __name__ == '__main__':
    identificar_texto()