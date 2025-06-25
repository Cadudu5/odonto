import cv2
import numpy as np
import os
import csv
from skimage.segmentation import slic
from skimage.util import img_as_float

# --- CONFIGURAÇÃO ---
# Mude para True para salvar as imagens de verificação, ou False para apenas calcular
GERAR_IMAGEM_DE_VERIFICACAO = True
PASTA_VERIFICACAO = "imagens_verificacao"
# --------------------


def calcular_porcentagem_com_superpixels(caminho_imagem, num_segmentos=5000):
    """
    Calcula a porcentagem da área marcada com DAB e opcionalmente retorna
    uma máscara visual da área positiva.
    """
    imagem_bgr = cv2.imread(caminho_imagem)
    if imagem_bgr is None:
        print(f"Não foi possível ler a imagem: {caminho_imagem}")
        return 0.0, None

    imagem_float = img_as_float(imagem_bgr)
    segmentos = slic(imagem_float, n_segments=num_segmentos, sigma=3, start_label=1)

    # NOVO: Cria uma máscara visual vazia que será preenchida
    mascara_positiva_visual = np.zeros(imagem_bgr.shape[:2], dtype="uint8")

    area_positiva = 0
    area_total_tecido = 0

    # --- COLE AQUI OS SEUS VALORES DE COR CALIBRADOS ---

    # cores amarelas/castanhas (DAB)
    lower_dab = np.array([0, 50, 50]) # 
    upper_dab = np.array([179, 255, 219]) # 

    # cores azuladas (hematoxilina)
    lower_hematoxilina = np.array([96, 59, 50]) # 
    upper_hematoxilina = np.array([140, 255, 255]) # 
    # ----------------------------------------------------

    for label in np.unique(segmentos):
        mascara_superpixel = np.zeros(imagem_bgr.shape[:2], dtype="uint8")
        mascara_superpixel[segmentos == label] = 255
        
        cor_media_bgr = cv2.mean(imagem_bgr, mask=mascara_superpixel)
        cor_media_hsv = cv2.cvtColor(np.uint8([[cor_media_bgr[0:3]]]), cv2.COLOR_BGR2HSV)[0][0]

        # Classifica como Positivo (DAB)
        if (cor_media_hsv[0] >= lower_dab[0] and cor_media_hsv[0] <= upper_dab[0] and
            cor_media_hsv[1] >= lower_dab[1] and cor_media_hsv[1] <= upper_dab[1] and
            cor_media_hsv[2] >= lower_dab[2] and cor_media_hsv[2] <= upper_dab[2]):
            
            num_pixels_no_superpixel = cv2.countNonZero(mascara_superpixel)
            area_positiva += num_pixels_no_superpixel
            area_total_tecido += num_pixels_no_superpixel
            
            # NOVO: Preenche a máscara visual com os superpixels positivos
            mascara_positiva_visual[segmentos == label] = 255

        # Classifica como Negativo (Hematoxilina)
        elif (cor_media_hsv[0] >= lower_hematoxilina[0] and cor_media_hsv[0] <= upper_hematoxilina[0] and
              cor_media_hsv[1] >= lower_hematoxilina[1] and cor_media_hsv[1] <= upper_hematoxilina[1] and
              cor_media_hsv[2] >= lower_hematoxilina[2] and cor_media_hsv[2] <= upper_hematoxilina[2]):
            
            area_total_tecido += cv2.countNonZero(mascara_superpixel)

    if area_total_tecido > 0:
        porcentagem_positiva = (area_positiva / area_total_tecido) * 100
    else:
        porcentagem_positiva = 0.0

    return porcentagem_positiva, mascara_positiva_visual

# --- Início da Execução do Script ---
if __name__ == "__main__":
    pasta_entrada = "imagens_originais"
    arquivo_saida_csv = "resultados_superpixels.csv"

    if not os.path.isdir(pasta_entrada):
        print(f"Erro: A pasta '{pasta_entrada}' não foi encontrada.")
        exit()

    # NOVO: Cria a pasta de verificação se ela não existir
    if GERAR_IMAGEM_DE_VERIFICACAO and not os.path.isdir(PASTA_VERIFICACAO):
        os.makedirs(PASTA_VERIFICACAO)

    cabecalho_csv = ["Nome do Arquivo", "Porcentagem Area Positiva com Superpixels (%)"]
    dados_csv = []
    
    arquivos_imagem = [f for f in os.listdir(pasta_entrada) if f.lower().endswith(('.tiff', '.tif', '.jpg', '.png'))]

    if not arquivos_imagem:
        print(f"Nenhuma imagem encontrada na pasta '{pasta_entrada}'.")
    else:
        print("Iniciando análise com superpixels...")
        for nome_arquivo in arquivos_imagem:
            caminho_completo_entrada = os.path.join(pasta_entrada, nome_arquivo)
            
            # A função agora retorna a porcentagem E a máscara
            porcentagem, mascara_positiva = calcular_porcentagem_com_superpixels(caminho_completo_entrada)
            
            dados_csv.append([nome_arquivo, f"{porcentagem:.2f}"])
            print(f"- {nome_arquivo}: {porcentagem:.2f}% de área positiva")
            
            # NOVO: Bloco para criar e salvar a imagem de verificação
            if GERAR_IMAGEM_DE_VERIFICACAO and mascara_positiva is not None:
                # Carrega a imagem original novamente para trabalhar com ela
                imagem_original = cv2.imread(caminho_completo_entrada)
                
                # Cria uma camada de cor verde sólida
                overlay_verde = np.zeros_like(imagem_original, dtype=np.uint8)
                overlay_verde[:] = (0, 255, 0)  # Cor Verde Brilhante em BGR
                
                # Aplica a cor verde apenas nas áreas da máscara positiva
                mascara_verde = cv2.bitwise_and(overlay_verde, overlay_verde, mask=mascara_positiva)
                
                # Mistura a imagem original com a camada verde semi-transparente
                # O peso 0.4 define a transparência da camada verde
                imagem_verificacao = cv2.addWeighted(mascara_verde, 0.4, imagem_original, 0.6, 0)

                # Define o nome e salva a imagem de saída
                nome_base, extensao = os.path.splitext(nome_arquivo)
                nome_saida = f"{nome_base}_verificacao.jpg"
                caminho_saida = os.path.join(PASTA_VERIFICACAO, nome_saida)
                cv2.imwrite(caminho_saida, imagem_verificacao)

        # Salva os resultados no arquivo CSV
        try:
            with open(arquivo_saida_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(cabecalho_csv)
                writer.writerows(dados_csv)
            print(f"\nResultados da análise com superpixels salvos em '{arquivo_saida_csv}'")
            if GERAR_IMAGEM_DE_VERIFICACAO:
                print(f"Imagens de verificação salvas na pasta '{PASTA_VERIFICACAO}'")
        except IOError:
            print(f"\nErro ao salvar o arquivo CSV.")