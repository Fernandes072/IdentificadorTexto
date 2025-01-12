import cv2
import os

arquivos = os.listdir('ajeitado') # Pega os arquivos na pasta ajeitado
for arquivo in arquivos:
    imagem = cv2.imread(f'ajeitado/{arquivo}') # Lê a imagem
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # Converte a imagem para escala de cinza
    _, nova_imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV) # Aplica o método THRESH_BINARY_INV

    contornos, _ = cv2.findContours(nova_imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encontra os contornos
    regiao_letras = []

    for contorno in contornos:
        (x, y, largura, altura) = cv2.boundingRect(contorno) # Pega as coordenadas do retângulo que envolve o contorno
        area = cv2.contourArea(contorno) # Calcula a área do contorno
        if area > 20 and area < 10000: # Verifica se a área do contorno está entre 20 e 10000
            regiao_letras.append((x, y, largura, altura)) # Adiciona as coordenadas do retângulo na lista regiao_letras

    linhas = {} # Dicionário para armazenar as linhas
    for (x, y, largura, altura) in regiao_letras:
        linha = y // 20 # Agrupa as letras verticalmente para formar as linhas
        if linha not in linhas: # Se a linha não estiver no dicionário, cria uma lista vazia
            linhas[linha] = []
        linhas[linha].append((x, y, largura, altura)) # Adiciona as coordenadas do retângulo na lista da linha

    imagem_final = cv2.merge([imagem] * 3) # Transforma a imagem em uma imagem RGB, para poder desenhar os retângulos coloridos
    nome_arquivo, _ = os.path.splitext(arquivo) # Pega o nome do arquivo sem a extensão
    espaco_maximo = 6 # Distância máxima entre as letras para considerar que elas fazem parte da mesma palavra
    i = 1

    for linha in sorted(linhas.keys()): # Itera sobre cada linha de cima para baixo
        letras = sorted(linhas[linha], key=lambda x: x[0]) # Ordena as letras da esquerda para a direita
        palavra_atual = []

        for j in range(len(letras)): # Itera sobre cada letra da linha
            x, y, largura, altura = letras[j] # Pega as coordenadas da letra
            if j == 0: # Se for a primeira letra da linha, adiciona na palavra atual
                palavra_atual.append((x, y, largura, altura))
            else:
                x_anterior, y_anterior, largura_anterior, altura_anterior = letras[j - 1] # Pega as coordenadas da letra anterior
                if x - (x_anterior + largura_anterior) <= espaco_maximo: # Calcula a distância entre a letra atual e a letra anterior, verifica se é menor ou igual ao espaço máximo e adiciona na palavra atual
                    palavra_atual.append((x, y, largura, altura))
                else: # Se a distância for maior que o espaço máximo, salva a palavra atual e começa uma nova palavra
                    x_min = min([p[0] for p in palavra_atual]) # Pega o menor x da palavra atual
                    y_min = min([p[1] for p in palavra_atual]) # Pega o menor y da palavra atual
                    x_max = max([p[0] + p[2] for p in palavra_atual]) # Pega o maior x da palavra atual
                    y_max = max([p[1] + p[3] for p in palavra_atual]) # Pega o maior y da palavra atual
                    imagem_palavra = imagem[y_min-3:y_max+3, x_min-3:x_max+3] # Recorta a palavra da imagem
                    if imagem_palavra.shape[0] > 0 and imagem_palavra.shape[1] > 0: # Verifica se a imagem da palavra é válida, pra não dar erro ao salvar
                        cv2.imwrite(f'palavras/{nome_arquivo}_{i}.png', imagem_palavra) # Salva a palavra
                        cv2.rectangle(imagem_final, (x_min-3, y_min-3), (x_max+3, y_max+3), (0, 0, 255), 1) # Desenha o retângulo da palavra na imagem final
                        i += 1
                    palavra_atual = [(x, y, largura, altura)] # Começa uma nova palavra
        
        #Repete o processo para a última palavra da linha
        if palavra_atual:
            x_min = min([p[0] for p in palavra_atual])
            y_min = min([p[1] for p in palavra_atual])
            x_max = max([p[0] + p[2] for p in palavra_atual])
            y_max = max([p[1] + p[3] for p in palavra_atual])
            imagem_palavra = imagem[y_min-3:y_max+3, x_min-3:x_max+3]
            if imagem_palavra.shape[0] > 0 and imagem_palavra.shape[1] > 0:
                cv2.imwrite(f'palavras/{nome_arquivo}_{i}.png', imagem_palavra)
                cv2.rectangle(imagem_final, (x_min-3, y_min-3), (x_max+3, y_max+3), (0, 0, 255), 1)
                i += 1

    cv2.imwrite(f'identificado/{nome_arquivo}.png', imagem_final) # Salva a imagem final com os retângulos das palavras