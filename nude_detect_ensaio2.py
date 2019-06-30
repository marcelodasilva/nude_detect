
from __future__ import print_function

import pprint
import random
import sys
from collections import Counter

import cv2
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


#################################################
#  FUNÇÕES UTILITÁRIAS
#################################################
def util_find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


#################################################
#  FUNÇÕES DO TRABALHO
#################################################
def carregar_imagem(caminho_imagem):
    imagem = cv2.imread(caminho_imagem)
    imagem = imutils.resize(imagem, width = 400)
    return imagem
def detectar_pessoas(imagem):
    imagem_silhueta = imutils.resize(imagem, width=min(400, imagem.shape[1]))
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


    (rects, weights) = hog.detectMultiScale(imagem_silhueta, winStride=(4, 4),
            padding=(8, 8), scale=1.05)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.5)


    if type(pick) is list:
        if (pick.any()):
            status = "CORPO_DETECTADO"
            xA, yA, xB, yB = pick[0]
            corpoDetectado = imagem_silhueta[yA:yB, xA:xB].copy()
            return corpoDetectado,status
    if type(pick) is tuple:
        if (any(pick)):
            status = "CORPO_NAO_DETECTADO"
        return imagem_silhueta,status
    if pick is None:
        status = "CORPO_NAO_DETECTADO"
        return imagem_silhueta,status
        
    # cv2.rectangle(imagem_silhueta, (xA, yA), (xB, yB), cor, 2)
def detectar_faces(imagem):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #convertendo imagem para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # aplicando a detecção da face na imagem em escala de cinza
                                        #imagem, escala, número minimo de vizinhos
    faces = face_cascade.detectMultiScale(imagem_cinza, 1.2, 5)

    if type(faces) is list:
        if not (faces.any()):
            status = "FACE_NAO_DETECTADA"
            return imagem, status
    if type(faces) is tuple:
        if not (any(faces)):
            status = "FACE_NAO_DETECTADA"
            return imagem, status
        

    x = faces[0][0] #ponto x
    y = faces[0][1] #ponto y
    w = faces[0][2] #largura usando como base o ponto x
    h = faces[0][3] #altura usando como base o ponto y


    face_image = imagem[y:y+h, x:x+w].copy()
    cv2.imshow('face detecada',face_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    status = "FACE_DETECTADA"
    return face_image, status
def detectar_pele(imagem,minimo,maximo):
    #redimensionando imagem, convertendo imagem de rgb para hsv e
    # especificando a mascara de detecção de pele usando os limiares maximos e minimos
    
    imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(imagem_hsv, minimo, maximo)


    #aplicando melhorias a mascara de pele, usando transformadores de erosão e dilatação
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)


    #aplicando um leve desfoque para remover os ruídos
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)


    skin = cv2.bitwise_and(imagem, imagem, mask = skinMask)

    return skin
def extrair_area_candidata(imagem_face_pele,margem):
    h, w = imagem_face_pele.shape[:2]
    max_x = imagem_face_pele.shape[1] - margem - 1
    max_y = imagem_face_pele.shape[0] - margem - 1

    min_x = margem + 1
    min_y = margem + 1


    px = random.randint(min_x,max_x)
    py = random.randint(min_y,max_y)


    yA = py - margem
    yB = py + margem + margem + 1
    xA = px - margem
    xB = px + margem + margem + 1


    area_selecionada = imagem_face_pele[yA:yB,xA:xB].copy()
    imagem_area = cv2.rectangle(imagem_face_pele.copy(),(xA,yA),(xB,yB),(255,0,0),0)
    cv2.imshow('area em destaque',imagem_area)

    return area_selecionada
def identificar_cores_dominantes(area_candidata,numero_cores):
    area_candidata = cv2.cvtColor(area_candidata, cv2.COLOR_BGR2RGB)
    area_candidata_reshaped = area_candidata.reshape((area_candidata.shape[0] * area_candidata.shape[1],3))

    clt = KMeans(n_clusters=numero_cores)
    clt.fit(area_candidata_reshaped)
    hist = util_find_histogram(clt)
    cores = clt.cluster_centers_

    return hist,cores
def avaliar_area_candidata(histograma,cores):
    for c,h in zip(cores,histograma):
        b,g,r = c
        if all(x <= 50 for x in (b, g, r)) and (h >= 0.3):
            return False
        if all(x >= 200 for x in (b, g, r)) and (h >= 0.3):
            return False
    return True
def selecionar_cor_mais_dominante(histograma,cores):
    cores_ordenadas = [x for _,x in sorted(zip(histograma,cores),reverse=True)]
    return cores_ordenadas[0]
def marcar_pele_corpo(imagem,cor_dominante,percentual_canais,cor_marcacao):
    
    r_dominante,g_dominante,b_dominante = cor_dominante
    
    r_dominante_max = r_dominante + (r_dominante * percentual_canais)
    r_dominante_min = r_dominante - (r_dominante * percentual_canais)

    g_dominante_max = g_dominante + (g_dominante * percentual_canais)
    g_dominante_min = g_dominante - (g_dominante * percentual_canais)

    b_dominante_max = b_dominante + (b_dominante * percentual_canais)
    b_dominante_min = b_dominante - (b_dominante * percentual_canais)
    
    
    for x in range(imagem.shape[0]):
        for y in range(imagem.shape[1]):
             b,g,r = imagem[x,y]

             if (b_dominante_min <= b <= b_dominante_max) and (g_dominante_min <= g <= g_dominante_max) and (r_dominante_min<= r <=r_dominante_max):
                 imagem[x,y] = cor_marcacao
    return imagem





###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
################                  EXECUÇÃO                     ############################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

def main():
    #################################################
    # CONSTANTES
    #################################################
    margem = 5
    numero_cores = 5
    percentual_canais = 0.5
    cor_marcacao = (0,255,0)
    minimo = np.array([0, 48, 80], dtype=np.uint8)
    maximo = np.array([20, 255, 255], dtype=np.uint8)
    
    #################################################
    # CARREGANDO IMAGEM
    #################################################
    
    caminho_imagem = 'dbNudeDetection/nonnude65.jpg'
    # caminho_imagem = 'alta/mb04.jpg'
    # caminho_imagem = str(sys.argv[1])
    original = carregar_imagem(caminho_imagem)
    cv2.imshow('original', original)
    #################################################
    # DETECÇÕES (FACE E CORPO)
    #################################################
    imagem = original.copy()
    imagem_face,status_face = detectar_faces(original)
    cv2.imshow('face',imagem_face)
    cor_dominante = None
    if status_face == "FACE_DETECTADA":
        
        imagem_face_pele = detectar_pele(imagem_face,minimo,maximo)
        backup_imagem_face_pele = imagem_face_pele.copy()
        cv2.imshow('pele face',imagem_face_pele)
        area_candidata = None
        histograma = None
        cores = None
        while True:
            area_candidata = extrair_area_candidata(imagem_face_pele,margem)
            histograma, cores = identificar_cores_dominantes(area_candidata,numero_cores)
            if avaliar_area_candidata(histograma,cores):
                cv2.imshow('area escolhida',area_candidata)
                # cv2.imshow('area em destaque',imagem_area_destaque)
                break
            
            # break # temporario (REMOVER QUANDO FOR TESTAR NA PRÁTICA)
        
        cor_dominante = selecionar_cor_mais_dominante(histograma,cores)

    else:
        imagem_pele = detectar_pele(imagem,minimo,maximo)
        area_candidata = None
        histograma = None
        cores = None
        while True:
            area_candidata = extrair_area_candidata(imagem_pele,margem)
            histograma, cores = identificar_cores_dominantes(area_candidata,numero_cores)
            if avaliar_area_candidata(histograma,cores):
                cv2.imshow('area escolhida',area_candidata)
                break
        cor_dominante = selecionar_cor_mais_dominante(histograma,cores)

    imagem_marcada = marcar_pele_corpo(original.copy(),cor_dominante,percentual_canais,cor_marcacao)

    cv2.imshow("imagem_marcada",imagem_marcada)
    savepath = caminho_imagem.split('.')
    savepath = '{}-skin.{}'.format(savepath[0],savepath[1])
    print(f"armazenado em {savepath}")
    cv2.imwrite(savepath,imagem_marcada)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


 

if __name__ == '__main__':
    main()
