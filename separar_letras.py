import cv2
import os

arquivos = os.listdir('palavras') # Pega os arquivos na pasta palavras
for arquivo in arquivos:
    imagem = cv2.imread(f'palavras/{arquivo}') # Lê a imagem
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # Converte a imagem para escala de cinza
    _, nova_imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV) # Aplica o método THRESH_BINARY_INV

    contornos, _ = cv2.findContours(nova_imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encontra os contornos
    regiao_letras = []

    for contorno in contornos:
        (x, y, largura, altura) = cv2.boundingRect(contorno) # Pega as coordenadas do retângulo que envolve o contorno
        area = cv2.contourArea(contorno) # Calcula a área do contorno
        if area > 20 and area < 10000: # Verifica se a área do contorno está entre 20 e 10000
            regiao_letras.append((x, y, largura, altura)) # Adiciona as coordenadas do retângulo na lista regiao_letras

    regiao_letras = sorted(regiao_letras, key=lambda x: x[0]) # Ordena as letras da esquerda para a direita
    imagem_final = cv2.merge([imagem] * 3) # Transforma a imagem em uma imagem RGB, para poder desenhar os retângulos coloridos
    nome_arquivo, _ = os.path.splitext(arquivo) # Pega o nome do arquivo sem a extensão
    i = 1

    for retangulo in regiao_letras: 
        x, y, largura, altura = retangulo # Pega as coordenadas do retângulo
        imagem_letra = imagem[y-1:y+altura+1, x-1:x+largura+1] # Recorta a letra da imagem
        if imagem_letra.shape[0] > 0 and imagem_letra.shape[1] > 0: # Verifica se a imagem da letra é válida, pra não dar erro ao salvar
            cv2.imwrite(f'letras/{nome_arquivo}_{i}.png', imagem_letra) # Salva a letra
            cv2.rectangle(imagem_final, (x-1, y-1), (x+largura+1, y+altura+1), (0, 0, 255), 1) # Desenha o retângulo da letra na imagem final
            i += 1
    #cv2.imwrite(f'identificado/{nome_arquivo}.png', imagem_final) # Salva a imagem final com os retângulos das letras