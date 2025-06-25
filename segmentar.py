import os
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt

def segmentar_imagem_por_superpixels(caminho_imagem_entrada, caminho_imagem_saida, num_segmentos=300, sigma=5):
    """
    Realiza a segmentação de uma imagem usando o algoritmo SLIC para gerar superpixels.

    Args:
        caminho_imagem_entrada (str): O caminho para a imagem a ser segmentada.
        caminho_imagem_saida (str): O caminho para salvar a imagem segmentada.
        num_segmentos (int): O número aproximado de superpixels a serem gerados.
        sigma (int): O desvio padrão do filtro Gaussiano aplicado antes da segmentação.
    """
    # Carrega a imagem do disco
    # O OpenCV carrega imagens em formato BGR, então convertemos para RGB
    imagem_bgr = cv2.imread(caminho_imagem_entrada)
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

    # Converte a imagem para o formato de ponto flutuante, que é o esperado pela função slic
    imagem_float = img_as_float(imagem_rgb)

    # Aplica o algoritmo SLIC para gerar os superpixels
    # A função retorna uma máscara onde cada superpixel tem um rótulo inteiro único
    segmentos = slic(imagem_float, n_segments=num_segmentos, sigma=sigma, start_label=1)

    # Cria a imagem de saída desenhando as fronteiras dos superpixels na imagem original
    # A função mark_boundaries retorna a imagem com as bordas em destaque
    imagem_segmentada = mark_boundaries(imagem_float, segmentos)

    # Para salvar a imagem, precisamos convertê-la de volta para o formato de 8 bits (0-255)
    imagem_segmentada_8bit = (imagem_segmentada * 255).astype("uint8")

    # Converte a imagem de volta para BGR para que o OpenCV possa salvá-la corretamente
    imagem_segmentada_bgr = cv2.cvtColor(imagem_segmentada_8bit, cv2.COLOR_RGB2BGR)

    # Salva a imagem resultante no caminho de saída
    cv2.imwrite(caminho_imagem_saida, imagem_segmentada_bgr)

    print(f"Imagem segmentada salva em: {caminho_imagem_saida}")


# --- Início da Execução do Script ---
if __name__ == "__main__":
    # Define os nomes das pastas
    pasta_entrada = "imagens_originais"
    #pasta_entrada = "./Histona, MPO e NE"
    pasta_saida = "./imagens_segmentadas"

    # Cria a pasta de saída se ela não existir
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    # Lista todas as imagens na pasta de entrada
    try:
        lista_de_imagens = [f for f in os.listdir(pasta_entrada) if os.path.isfile(os.path.join(pasta_entrada, f))]
    except FileNotFoundError:
        print(f"Erro: A pasta '{pasta_entrada}' não foi encontrada. Crie-a e adicione suas imagens.")
        exit()


    if not lista_de_imagens:
        print(f"Nenhuma imagem encontrada na pasta '{pasta_entrada}'.")
    else:
        # Itera sobre cada imagem e aplica a segmentação
        for nome_arquivo in lista_de_imagens:
            caminho_completo_entrada = os.path.join(pasta_entrada, nome_arquivo)

            # Define o nome do arquivo de saída
            nome_base, extensao = os.path.splitext(nome_arquivo)
            nome_arquivo_saida = f"{nome_base}_segmentada{extensao}"
            caminho_completo_saida = os.path.join(pasta_saida, nome_arquivo_saida)

            # Chama a função para segmentar a imagem
            # Você pode ajustar os parâmetros 'num_segmentos' e 'sigma' conforme sua necessidade
            segmentar_imagem_por_superpixels(caminho_completo_entrada, caminho_completo_saida, num_segmentos=5000, sigma=4)

        print("\nProcesso de segmentação concluído para todas as imagens.")