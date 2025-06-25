import cv2
import numpy as np
import os
import csv
from skimage.segmentation import slic
from skimage.util import img_as_float

def calcular_porcentagem_com_superpixels(caminho_imagem, num_segmentos=400):
    """
    Calcula a porcentagem da área marcada com DAB (amarelo/castanho) em uma
    imagem de imunohistoquímica usando uma abordagem baseada em superpixels.

    Args:
        caminho_imagem (str): O caminho para a imagem a ser analisada.
        num_segmentos (int): O número aproximado de superpixels a gerar.

    Returns:
        float: A porcentagem da área positiva (DAB), ou 0.0 se não houver tecido.
    """
    # Carrega a imagem com OpenCV (formato BGR)
    imagem_bgr = cv2.imread(caminho_imagem)
    if imagem_bgr is None:
        print(f"Não foi possível ler a imagem: {caminho_imagem}")
        return 0.0

    # Converte a imagem para um formato de ponto flutuante para o SLIC
    imagem_float = img_as_float(imagem_bgr)

    # 1. GERAR SUPERPIXELS
    # A função SLIC retorna um mapa de rótulos inteiros
    segmentos = slic(imagem_float, n_segments=num_segmentos, sigma=3, start_label=1)

    # Inicializa as áreas
    area_positiva = 0
    area_total_tecido = 0

    # Intervalos de cor em HSV (ajuste conforme necessário)
    lower_dab = np.array([10, 40, 40])
    upper_dab = np.array([30, 255, 200])
    lower_hematoxilina = np.array([100, 40, 40])
    upper_hematoxilina = np.array([140, 255, 220])
    
    # 2. ANALISAR CADA SUPERPIXEL
    for label in np.unique(segmentos):
        # Cria uma máscara para o superpixel atual
        mascara_superpixel = np.zeros(imagem_bgr.shape[:2], dtype="uint8")
        mascara_superpixel[segmentos == label] = 255
        
        # 3. CALCULAR A COR MÉDIA do superpixel usando a máscara
        cor_media_bgr = cv2.mean(imagem_bgr, mask=mascara_superpixel)
        
        # Converte a cor média BGR para HSV para facilitar a classificação
        # É preciso converter para um array numpy e depois para HSV
        cor_media_hsv = cv2.cvtColor(np.uint8([[cor_media_bgr[0:3]]]), cv2.COLOR_BGR2HSV)[0][0]

        # 4. CLASSIFICAR O SUPERPIXEL
        # Verifica se a cor média está na faixa do DAB (positivo)
        if (cor_media_hsv[0] >= lower_dab[0] and cor_media_hsv[0] <= upper_dab[0] and
            cor_media_hsv[1] >= lower_dab[1] and cor_media_hsv[1] <= upper_dab[1] and
            cor_media_hsv[2] >= lower_dab[2] and cor_media_hsv[2] <= upper_dab[2]):
            
            # 5. ADICIONAR ÁREA
            area_positiva += cv2.countNonZero(mascara_superpixel)
            area_total_tecido += cv2.countNonZero(mascara_superpixel)

        # Verifica se a cor média está na faixa da Hematoxilina (negativo)
        elif (cor_media_hsv[0] >= lower_hematoxilina[0] and cor_media_hsv[0] <= upper_hematoxilina[0] and
              cor_media_hsv[1] >= lower_hematoxilina[1] and cor_media_hsv[1] <= upper_hematoxilina[1] and
              cor_media_hsv[2] >= lower_hematoxilina[2] and cor_media_hsv[2] <= upper_hematoxilina[2]):
            
            # Adiciona apenas à área total do tecido
            area_total_tecido += cv2.countNonZero(mascara_superpixel)

    # 6. CALCULAR A PORCENTAGEM FINAL
    if area_total_tecido > 0:
        porcentagem_positiva = (area_positiva / area_total_tecido) * 100
    else:
        porcentagem_positiva = 0.0

    return porcentagem_positiva

# --- Início da Execução do Script ---
if __name__ == "__main__":
    # Pré-requisitos: pip install opencv-python numpy scikit-image
    pasta_entrada = "imagens_originais"
    arquivo_saida_csv = "resultados_superpixels.csv"

    if not os.path.isdir(pasta_entrada):
        print(f"Erro: A pasta '{pasta_entrada}' não foi encontrada.")
        exit()

    cabecalho_csv = ["Nome do Arquivo", "Porcentagem Area Positiva com Superpixels (%)"]
    dados_csv = []
    
    arquivos_imagem = [f for f in os.listdir(pasta_entrada) if f.lower().endswith(('.tiff', '.tif', '.jpg', '.png'))]

    if not arquivos_imagem:
        print(f"Nenhuma imagem encontrada na pasta '{pasta_entrada}'.")
    else:
        print("Iniciando análise com superpixels...")
        for nome_arquivo in arquivos_imagem:
            caminho_completo_entrada = os.path.join(pasta_entrada, nome_arquivo)
            
            # Chama a função. Você pode ajustar o número de segmentos aqui.
            # Menos segmentos = superpixels maiores; Mais segmentos = superpixels menores.
            porcentagem = calcular_porcentagem_com_superpixels(caminho_completo_entrada, num_segmentos=500)
            
            dados_csv.append([nome_arquivo, f"{porcentagem:.2f}"])
            print(f"- {nome_arquivo}: {porcentagem:.2f}% de área positiva")
            
        try:
            with open(arquivo_saida_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(cabecalho_csv)
                writer.writerows(dados_csv)
            print(f"\nResultados da análise com superpixels salvos em '{arquivo_saida_csv}'")
        except IOError:
            print(f"Erro ao salvar o arquivo CSV.")