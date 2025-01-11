import cv2
import os
from PIL import Image

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


i = 1
j = 2
for cont in range(1, 11):
    contador_imagem1 = 0
    contador_imagem2 = 0
    imagem1 = Image.open(f'testes-metodo/{i}.png')
    imagem2 = Image.open(f'testes-metodo/{j}.png')

    for x in range(imagem1.size[1]):
            for y in range(imagem1.size[0]):
                cor_pixel = imagem1.getpixel((y, x))
                if cor_pixel < 115:
                    contador_imagem1 += 1

    for x in range(imagem2.size[1]):
            for y in range(imagem2.size[0]):
                cor_pixel = imagem2.getpixel((y, x))
                if cor_pixel < 115:
                    contador_imagem2 += 1

    if contador_imagem1 < contador_imagem2:
        imagem1.save(f'testes-metodo2/{i}.png')
    else:
        imagem2.save(f'testes-metodo2/{j}.png')

    i += 5
    j += 5