import cv2
import numpy as np

def nada(x):
    # Função vazia necessária para a criação das trackbars
    pass

# --- CONFIGURAÇÃO ---
# Coloque aqui o nome da imagem que você quer usar para calibrar
# Use a imagem que deu o resultado 0.00%
NOME_IMAGEM = './imagens_originais/LANA  1E G3 LP 14 S (7) 40X H3.tiff'
# --------------------

# Carrega a imagem
imagem = cv2.imread(NOME_IMAGEM)
if imagem is None:
    print(f"Erro: não foi possível encontrar a imagem em '{NOME_IMAGEM}'")
    exit()

# Redimensiona a imagem para caber na tela, se for muito grande
altura, largura = imagem.shape[:2]
largura_max = 1000
if largura > largura_max:
    fator = largura_max / largura
    imagem = cv2.resize(imagem, None, fx=fator, fy=fator)

# Converte a imagem para o espaço de cor HSV
imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

# Cria uma janela para as trackbars
cv2.namedWindow('Trackbars')
cv2.createTrackbar('H Min', 'Trackbars', 0, 179, nada)
cv2.createTrackbar('H Max', 'Trackbars', 179, 179, nada)
cv2.createTrackbar('S Min', 'Trackbars', 0, 255, nada)
cv2.createTrackbar('S Max', 'Trackbars', 255, 255, nada)
cv2.createTrackbar('V Min', 'Trackbars', 0, 255, nada)
cv2.createTrackbar('V Max', 'Trackbars', 255, 255, nada)

print("\n--- Instruções ---")
print("1. Ajuste os sliders para isolar a cor desejada (amarelo/castanho ou azul).")
print("2. A janela 'Mascara' mostrará em branco os pixels selecionados.")
print("3. Quando estiver satisfeito, anote os valores de H, S e V (Min e Max).")
print("4. Pressione a tecla 'ESC' para fechar as janelas.")
print("--------------------\n")


while True:
    # Pega os valores atuais das trackbars
    h_min = cv2.getTrackbarPos('H Min', 'Trackbars')
    h_max = cv2.getTrackbarPos('H Max', 'Trackbars')
    s_min = cv2.getTrackbarPos('S Min', 'Trackbars')
    s_max = cv2.getTrackbarPos('S Max', 'Trackbars')
    v_min = cv2.getTrackbarPos('V Min', 'Trackbars')
    v_max = cv2.getTrackbarPos('V Max', 'Trackbars')

    # Cria os arrays numpy com os valores dos sliders
    lower_range = np.array([h_min, s_min, v_min])
    upper_range = np.array([h_max, s_max, v_max])

    # Cria a máscara usando os valores atuais
    mascara = cv2.inRange(imagem_hsv, lower_range, upper_range)

    # Mostra a imagem original e a máscara
    cv2.imshow('Imagem Original', imagem)
    cv2.imshow('Mascara', mascara)
    
    # Sai do loop se a tecla ESC for pressionada
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Imprime os valores finais para facilitar o copia e cola
print("\nValores finais para copiar:")
print(f"lower_range = np.array([{h_min}, {s_min}, {v_min}])")
print(f"upper_range = np.array([{h_max}, {s_max}, {v_max}])")


cv2.destroyAllWindows()