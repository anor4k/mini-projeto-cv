# %% [markdown]
# # Mini-Projeto de Visão Computacional
# ## Grupo Turing
# ### Noel Viscome Eliezer
# %% [markdown]
# # Manipulação de Imagens
# %% Módulos
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# %% Pillow
# funciona bem em notebooks mas não retorna como arrays
def readImagePIL(filename, show=False):
    try:
        image = Image.open(filename)
    except FileNotFoundError as error:
        print("Erro: arquivo não encontrado.")
        print(error)
        return
    if show:
        image.show()
    return image


# %% OpenCV
# Lê uma imagem. Por padrão assume RGB 8 bits. Filename não tratado.
# bugado pra caralho meu deus do céu
# retorna imagem em RGB pra manter coerente com os outros métodos
def readImageCV(filename, flags=cv2.IMREAD_COLOR, show=False):
    try:
        image = cv2.imread(filename, flags=flags)
    except FileNotFoundError as error:
        print("Erro: arquivo não encontrado.")
        print(error)
        return
    if show:
        cv2.imshow('img', image)
        cv2.waitKey(0)                                  # pra não travar aaaa
        cv2.destroyAllWindows()                         # ainda trava
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)       # retorna imagem convertida


# %% matplotlib
# funciona melhor em notebooks e retorna os valores como matrizes pro numpy
def readImagePlot(filename, show=False):
    try:
        image = plt.imread(filename)
    except FileNotFoundError as error:
        print("Erro: arquivo não encontrado.")
        print(error)
        return
    if show:
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')
        plt.show()
    return image


readImage = readImageCV             # alias para a função


def viewImage(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    plt.show()


# %% [markdown]
# ## Visualizando a imagem
# Usamos algumas imagens para testar
# %%
jpg = readImage(r'Darth Vader/darth_vader_fandom1.JPEG')
png = readImage(r'Yoda/yoda_fandom10.PNG')

viewImage(jpg);
viewImage(png);
# %% [markdown]
# ## RGB vs YUV:
# A função `toYUV()` converte RGB para YUV usando a matriz de cores especificada (BT.709 por padrão).
# Para obter uma imagem preto e branco a partir dela, podemos apenas usar o plano Y.
#
# A função `toBnW()` converte RGB em preto e branco sem mudar o espaço de cores,
# usando como cor a média entre R G e B de cada pixel.
#
# Todas as funções implementadas agora assumiram que o argumento `image` será um array numpy RGB
# Sinto muito openCV mas nesta casa não trabalhamos com BGR


# %%
# isso teoricamente deve funcionar mas não sei qual o melhor jeito de exibir a imagem correta;
# matplotlib só funciona com RGB
def toYUV(image, matrix='709', bnw=False):
    if matrix == '709':
        # R G B
        Y = np.array([0.2126, 0.7152, 0.0722])
        U = np.array([-0.09991, -0.33609, 0.436])
        V = np.array([[0.615, -0.55861, - 0.05639]])
    elif matrix == '601':
        Y = np.array([0.299, 0.587, 0.114])
        U = np.array([-0.14713, -0.28886, 0.436])
        V = np.array([0.615, -0.51499, -0.10001])
    else:
        print('Matriz de cores inválida! Use \'709\' ou \'601\'.')
    # conversão para YUV
    image = image.copy()                               # source é read-only
    for (i, row) in enumerate(image):
        for (j, pixel) in enumerate(row):
            image[i][j] = pixel@np.array([Y, U, V])
    if bnw:
        image = image[:, :, 0]
    return image


def toBnW(image):
    image = image.copy()                               # source é read-only
    for (i, row) in enumerate(image):
        for (j, pixel) in enumerate(row):
            color = np.round(np.mean(pixel))
            image[i][j] = [color, color, color]
    return image

# %% [markdown]
# # Data Augmentation
# Vamos escrever duas funções para manipular na mão as imagens:
# `flip()`: inverte horizontalmente ou verticalmente a imagem
# `brightness()`: ainda em RGB, aumenta o briho das imagens


# %%
def flip(image, axis='x'):
    image = image.copy()
    if axis == 'x':
        image = np.flip(image, 1)
    elif axis == 'y':
        image = np.flip(image, 0)
    elif axis == 'xy':
        image = np.flip(image, (0, 1))
    return image


# amount em decimal a multiplicar o brilho atual
def brightness(image, amount):
    image = image.copy()
    image = image/255             # normaliza a imagem para prevenir overflow
    image = image*amount          # garante valores entre 0 e 255
    image = np.clip(image*255, 0, 255)
    image = image.astype(np.uint8)
    return image


# %%
img = jpg
viewImage(flip(img, 'xy'));
viewImage(brightness(img, 0.7));
viewImage(brightness(img, 1.3));
# %% [markdown]
# ## Filtros
# Agora que manipulamos algumas imagens, vamos experimentar alguns filtros.
# Vamos implementar um box blur bem simples (referência: https://en.wikipedia.org/wiki/Kernel_(image_processing))


def boxBlur(image, amount=1):
    kernel = (1/9) * np.ones((3, 3))
    blurred = cv2.filter2D(image, -1, kernel=kernel)
    # itera sobre a imagem já borrada para aumentar o blur
    for i in range(amount - 1):
        blurred = cv2.filter2D(blurred, -1, kernel=kernel)
    return blurred


def sharpen(image, amount=1):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(image, -1, kernel=kernel)
    for i in range(amount - 1):
        sharp = cv2.filter2D(image, -1, kernel=sharp)
    return sharp


# %%
viewImage(boxBlur(jpg));
viewImage(sharpen(jpg));
# usando filtros já embutidos no OpenCV
viewImage(cv2.GaussianBlur(jpg, (0, 0), 1));
viewImage(cv2.fastNlMeansDenoisingColored(jpg));
