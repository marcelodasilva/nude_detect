from __future__ import print_function
import cv2
import numpy as np
import imutils
import random
from imutils.object_detection import non_max_suppression


#========================================================================
# CARREGANDO IMAGEM
#========================================================================

imagem = cv2.imread("alta\m02.jpg") #alta exposição corporal
backup = imagem.copy()


#========================================================================
# DETECTANDO FACES NA IMAGEM
#========================================================================

# especificando classificador de detecção de face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#convertendo imagem para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# aplicando a detecção da face na imagem em escala de cinza
                                    #imagem, escala, número minimo de vizinhos
faces = face_cascade.detectMultiScale(imagem_cinza, 1.1, 5)

x = faces[0][0] #ponto x
y = faces[0][1] #ponto y
w = faces[0][2] #largura usando como base o ponto x
h = faces[0][3] #altura usando como base o ponto y


face_image = imagem[y:y+h, x:x+w].copy()
cv2.imshow("imagem original",imagem)
#cv2.imwrite("resultados\original.bmp",imagem)
cv2.imshow("imagem cortada",face_image)
#cv2.imwrite("resultados\face.bmp",face_image)

cv2.waitKey(0)
cv2.destroyAllWindows()




#========================================================================
# DETECTANDO PIGMENTAÇÃO DA PELE NA FACE
#========================================================================

# atribuindo valores limiares de minimo e maximo para tonalidades de pele
minimo = np.array([0, 48, 80], dtype = "uint8")
maximo = np.array([20, 255, 255], dtype = "uint8")


#redimensionando imagem, convertendo imagem de rgb para hsv e
# especificando a mascara de detecção de pele usando os limiares maximos e minimos
imagem_redimensionada = imutils.resize(face_image, width = 400)
imagem_hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
skinMask = cv2.inRange(imagem_hsv, minimo, maximo)

#aplicando melhorias a mascara de pele, usando transformadores de erosão e dilatação
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
skinMask = cv2.erode(skinMask, kernel, iterations = 2)
skinMask = cv2.dilate(skinMask, kernel, iterations = 2)


#aplicando um leve desfoque para remover os ruídos
skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)


skin = cv2.bitwise_and(face_image, face_image, mask = skinMask)


cv2.imshow("face detectada e mascara de pele", np.hstack([face_image, skin]))

#cv2.imwrite("resultados\faceSkin.bmp",skin)
#cv2.imshow("histograma", cv2.calcHist([face_image],[1],None,[256],[0,256]))   



cv2.waitKey(0)
cv2.destroyAllWindows()




#========================================================================
# SELECIONANDO PIXEL ALEATORIAMENTE DENTRO DA IMAGEM COM APENAS PELE
#========================================================================
numTry = 0
while True and numTry != 50:
    numTry+=1
    #print(numTry)
    # selecionando coordenadas x y aleatórias dentro da imagem
    px = random.randint(0,skin.shape[0]-1)
    py = random.randint(0,skin.shape[1]-1)

    # capturando pixel resultante das coordenadas 
    pixel = skin[px,py]

    # extraindo canais de cores do pixel
    b,g,r = pixel

    #avaliando se o pixel corresponde a alguma cor diferente de preto
    if  ((b != 0 and g != 0 and r !=0)and (b>=50 and g >=100 and r >= 150)):
            print("pixel colorido")
            break
    else:
            if numTry==50:
                print("pixel preto")
                print("não foi possivel identificar cor de pele")
            else:
                print("pixel preto")
        #print(px, py)
    
if numTry != 50:
# coordenadas do pixel valido
    print(px, py)
# valores dos canais rgb do pixel valido
    print(pixel)
    img = cv2.rectangle(skin,(px-1,py-1),(px+1,py+1),(255,0,0),0)
    cv2.imshow("pixel selecionado",img)
#    cv2.imwrite("resultados\faceSkinPixel.bmp",img)



cv2.waitKey(0)
cv2.destroyAllWindows()


#========================================================================
# DETECTANDO PESSOAS
#========================================================================
imagem_silhueta = imutils.resize(imagem, width=min(400, imagem.shape[1]))
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


(rects, weights) = hog.detectMultiScale(imagem_silhueta, winStride=(4, 4),
		padding=(8, 8), scale=1.05)

rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.5)


#========================================================================
# DETECTANDO CORPO DA PESSOA E APLICANDO A ANÁLISE DOS PIXELS DA IMAGEM
#========================================================================


xA, yA, xB, yB = pick[0]

corpoDetectado = imagem_silhueta[yA:yB, xA:xB].copy()

cv2.rectangle(imagem_silhueta, (xA, yA), (xB, yB), (0, 255, 0), 2)

cv2.imshow("pessoa detectada",imagem_silhueta)
#cv2.imwrite("resultados\pessoaDetectada.bmp",imagem_silhueta)
cv2.imshow("corpo detectado",corpoDetectado)
#cv2.imwrite("resultados\pessoaDetectadaCorpo.bmp",corpoDetectado)

cv2.waitKey(0)
cv2.destroyAllWindows()

pixelMarcado = corpoDetectado.copy()

linhas, colunas, canais = corpoDetectado.shape
totalPixels = linhas * colunas
percentual = 0.30
b, g, r = pixel

factorB = b * percentual
factorG = g * percentual
factorR = r * percentual



minTreshB = b - factorB
maxTreshB = b + factorB

minTreshG = g - factorG
maxTreshG = g + factorG

minTreshR = r - factorR
maxTreshR = r + factorR

while(maxTreshB > 255 or maxTreshG > 255 or maxTreshR > 255):
    if maxTreshB > 255:
        print("treeshold de B ajustado")
        maxTreshB = 255

    if maxTreshG > 255:
        print("treeshold de G ajustado")
        maxTreshG = 255

    if maxTreshR > 255:
        print("treeshold de R ajustado")
        maxTreshR = 255

print("max and min channels color")
print("B min: {0} max: {1}".format(minTreshB,maxTreshB))
print("G min: {0} max: {1}".format(minTreshG,maxTreshG))
print("R min: {0} max: {1}".format(minTreshR,maxTreshR))

print("percorrendo a imagem")
pixSkin = 0
for pix in range(corpoDetectado.shape[0]):
    for piy in range(corpoDetectado.shape[1]):
        #print("pix {0} {1} - {2}".format(pix,piy,corpoDetectado[pix,piy]))

        pb, pg, pr = corpoDetectado[pix,piy]

        if minTreshB <= pb <= maxTreshB:
            if minTreshG <= pg <= maxTreshG:
                if minTreshR <= pr <= maxTreshR:
                    pixSkin +=1
                    pixelMarcado[pix,piy] = (255,0,0)


print("analise completa")
expos = pixSkin / totalPixels

expos = expos * 100

print("{0} pixels skin | {1} pixels total".format(pixSkin,totalPixels))
print("exposição corporal de {0}%".format(expos))

if expos < 10:
    print("baixa exposição")
elif expos < 20:
    print("média exposição")
else:
    print ("alta exposição")
            
cv2.imshow("pixels reconhecidos",pixelMarcado)
cv2.imwrite("resultados\pessoaDetectadaCorpoMarcado.bmp",pixelMarcado)
        

        
        















   





#========================================================================
# FIM DO ALGORITMO
#========================================================================

cv2.waitKey(0)
cv2.destroyAllWindows()
