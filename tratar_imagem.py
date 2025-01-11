import cv2
import os
from PIL import Image

def tratar_imagem(pasta_origem, pasta_destino='ajeitado'):
    arquivos = os.listdir(pasta_origem) # Pega os arquivos na pasta de origem
    for arquivo in arquivos:
        imagem = cv2.imread(f'{pasta_origem}/{arquivo}') # Lê a imagem
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # Converte a imagem para escala de cinza
        
        _, imagem_tratada1 = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU) # Aplica o método THRESH_BINARY
        _, imagem_tratada2 = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_BINARY_INV or cv2.THRESH_OTSU) # Aplica o método THRESH_BINARY_INV

        contador_imagem1 = 0 # Contador de pixels pretos na imagem 1
        contador_imagem2 = 0 # Contador de pixels pretos na imagem 2

        imagem_tratada1 = Image.fromarray(imagem_tratada1) # Converte a imagem tratada 1 para o formato de imagem do PIL
        imagem_tratada2 = Image.fromarray(imagem_tratada2) # Converte a imagem tratada 2 para o formato de imagem do PIL

        # Conta a quantidade de pixels pretos na imagem 1
        for x in range(imagem_tratada1.size[1]):
            for y in range(imagem_tratada1.size[0]):
                cor_pixel = imagem_tratada1.getpixel((y, x))
                if cor_pixel < 115:
                    contador_imagem1 += 1

        # Conta a quantidade de pixels pretos na imagem 2
        for x in range(imagem_tratada2.size[1]):
                for y in range(imagem_tratada2.size[0]):
                    cor_pixel = imagem_tratada2.getpixel((y, x))
                    if cor_pixel < 115:
                        contador_imagem2 += 1

        nome_arquivo, _ = os.path.splitext(arquivo) # Pega o nome do arquivo sem a extensão
        if contador_imagem1 < contador_imagem2: # Escolhe qual foi o melhor método de tratamento - Verifica qual imagem tem menos pixels pretos, isso indica que o texto ficou preto e o fundo branco
            imagem_tratada1.save(f'{pasta_destino}/{nome_arquivo}.png')
        else:
            imagem_tratada2.save(f'{pasta_destino}/{nome_arquivo}.png')

if __name__ == '__main__':
    tratar_imagem('imagens-teste')