import cv2
import os

metodos = [
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TRUNC,
    cv2.THRESH_TOZERO,
    cv2.THRESH_TOZERO_INV
]

imagens = os.listdir('imagens-teste')

i = 0
for imagem in imagens:
    imagem = cv2.imread(f'imagens-teste/{imagem}')
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    for metodo in metodos:
        i += 1
        _, imagem_tratada = cv2.threshold(imagem_cinza, 127, 255, metodo or cv2.THRESH_OTSU)
        cv2.imwrite(f'testes-metodo/{i}.png', imagem_tratada)