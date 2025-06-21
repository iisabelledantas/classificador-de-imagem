# classificador de imagem

Este projeto implementa um classificador de imagens utilizando uma Rede Neural Convolucional (CNN) treinada no conjunto de dados CIFAR-10. O classificador é capaz de identificar 10 categorias diferentes de imagens.

## Algoritmo e Treinamento

O classificador é construído com uma Rede Neural Convolucional (CNN) utilizando a biblioteca TensorFlow/Keras. A arquitetura do modelo inclui camadas convolucionais (Conv2D), de pooling (MaxPooling2D), de achatamento (Flatten) e densas (Dense), com uma camada de Dropout para regularização. O modelo é compilado com o otimizador Adam e a função de perda `categorical_crossentropy`.

O treinamento foi realizado no conjunto de dados CIFAR-10, que consiste em 60.000 imagens coloridas de 32x32 pixels em 10 classes, com 6.000 imagens por classe. As imagens foram normalizadas para o intervalo [0, 1] e as etiquetas foram convertidas para o formato one-hot encoding. O modelo foi treinado por 10 épocas.

### Categorias de Imagens

O classificador é capaz de identificar imagens nas seguintes 10 categorias:

*   Avião
*   Automóvel
*   Pássaro
*   Gato
*   Veado
*   Cachorro
*   Sapo
*   Cavalo
*   Navio
*   Caminhão

## Como Executar o Projeto

Para executar este projeto, siga os passos abaixo:

### 1. Criação do ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 2. Instalação das Dependências

```bash
pip install tensorflow streamlit opencv-python Pillow
```

### 3. Treinamento do Modelo

```bash
python model.py
```

### Utilização do Visualizador

O visualizador é uma aplicação web interativa construída com Streamlit. Para iniciá-lo, execute:

```bash
streamlit run app.py
```

Após executar o comando, uma nova aba será aberta em seu navegador com a interface do classificador. Você terá duas opções para classificar imagens:

1.  **Usar Webcam**: Permite capturar uma imagem diretamente da sua webcam para classificação.
2.  **Upload de Imagem**: Permite fazer o upload de um arquivo de imagem do seu computador para classificação.

O visualizador exibirá a classe prevista e a confiança da previsão para a imagem fornecida.

## Métricas de Desempenho

O script de treinamento (`train_cifar10.py`) fornece a acurácia do modelo durante o processo de treinamento. Para uma avaliação mais completa do desempenho do classificador, incluindo métricas como precisão e recall, seria necessário executar um script de avaliação separado após o treinamento, utilizando o conjunto de dados de teste. Abaixo, um exemplo de como essas métricas poderiam ser apresentadas:

| Métrica   | Valor (Exemplo) |
|-----------|-----------------|
| Acurácia  | 0.85            |
| Precisão  | 0.82            |
| Recall    | 0.80            |

# Classificador de Imagens CIFAR-10

Este projeto implementa um classificador de imagens utilizando uma Rede Neural Convolucional (CNN) treinada no conjunto de dados CIFAR-10. O classificador é capaz de identificar 10 categorias diferentes de imagens.

## Algoritmo e Treinamento

O classificador é construído com uma Rede Neural Convolucional (CNN) utilizando a biblioteca TensorFlow/Keras. A arquitetura do modelo inclui camadas convolucionais (Conv2D), de pooling (MaxPooling2D), de achatamento (Flatten) e densas (Dense), com uma camada de Dropout para regularização. O modelo é compilado com o otimizador Adam e a função de perda `categorical_crossentropy`.

O treinamento foi realizado no conjunto de dados CIFAR-10, que consiste em 60.000 imagens coloridas de 32x32 pixels em 10 classes, com 6.000 imagens por classe. As imagens foram normalizadas para o intervalo [0, 1] e as etiquetas foram convertidas para o formato one-hot encoding. O modelo foi treinado por 10 épocas.

### Categorias de Imagens

O classificador é capaz de identificar imagens nas seguintes 10 categorias:

*   Avião
*   Automóvel
*   Pássaro
*   Gato
*   Veado
*   Cachorro
*   Sapo
*   Cavalo
*   Navio
*   Caminhão

## Desempenho do Classificador

Durante o treinamento, a acurácia do modelo foi monitorada. Para obter métricas de desempenho mais detalhadas, como precisão e recall, seria necessário executar uma avaliação pós-treinamento utilizando um conjunto de dados de teste. O script de treinamento (`train_cifar10.py`) salva o modelo treinado como `cifar10_model.h5`.

## Como Executar o Projeto

Para executar este projeto, siga os passos abaixo:

### Pré-requisitos

Certifique-se de ter o Python 3 e o pip instalados. As dependências do projeto podem ser instaladas via `pip`.

### Instalação das Dependências

Crie um ambiente virtual (opcional, mas recomendado) e instale as bibliotecas necessárias:

```bash
pip install tensorflow streamlit opencv-python Pillow
```

### Treinamento do Modelo (Opcional)

Se você deseja treinar o modelo do zero, execute o script de treinamento. Isso irá baixar o conjunto de dados CIFAR-10, treinar o modelo e salvá-lo como `cifar10_model.h5` e os nomes das classes como `class_names.json`.

```bash
python train_cifar10.py
```

### Utilização do Visualizador

O visualizador é uma aplicação web interativa construída com Streamlit. Para iniciá-lo, execute:

```bash
streamlit run app.py
```

Após executar o comando, uma nova aba será aberta em seu navegador com a interface do classificador. Você terá duas opções para classificar imagens:

1.  **Usar Webcam**: Permite capturar uma imagem diretamente da sua webcam para classificação.
2.  **Upload de Imagem**: Permite fazer o upload de um arquivo de imagem (JPG, JPEG, PNG) do seu computador para classificação.

O visualizador exibirá a classe prevista e a confiança da previsão para a imagem fornecida.

## Métricas de Desempenho

O treinamento do modelo foi realizado por 10 épocas, e o desempenho foi monitorado pela acurácia e perda (loss) tanto no conjunto de treinamento quanto no de validação. Abaixo estão os logs de treinamento:

```
Epoch 1/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 34s 21ms/step - accuracy: 0.3627 - loss: 1.7338 - val_accuracy: 0.5772 - val_loss: 1.2005
Epoch 2/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 31s 20ms/step - accuracy: 0.5673 - loss: 1.2252 - val_accuracy: 0.6331 - val_loss: 1.0455
Epoch 3/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 31s 20ms/step - accuracy: 0.6202 - loss: 1.0734 - val_accuracy: 0.6406 - val_loss: 1.0189
Epoch 4/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 31s 20ms/step - accuracy: 0.6641 - loss: 0.9628 - val_accuracy: 0.6858 - val_loss: 0.9125
Epoch 5/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 35s 22ms/step - accuracy: 0.6871 - loss: 0.8933 - val_accuracy: 0.6894 - val_loss: 0.9005
Epoch 6/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 38s 24ms/step - accuracy: 0.7052 - loss: 0.8329 - val_accuracy: 0.6818 - val_loss: 0.9243
Epoch 7/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 35s 20ms/step - accuracy: 0.7176 - loss: 0.7992 - val_accuracy: 0.6748 - val_loss: 0.9853
Epoch 8/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 37s 24ms/step - accuracy: 0.7390 - loss: 0.7433 - val_accuracy: 0.7123 - val_loss: 0.8536
Epoch 9/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 35s 22ms/step - accuracy: 0.7509 - loss: 0.7002 - val_accuracy: 0.7049 - val_loss: 0.8753
Epoch 10/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 34s 21ms/step - accuracy: 0.7678 - loss: 0.6486 - val_accuracy: 0.7128 - val_loss: 0.8690
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step
```

Além disso, um relatório de classificação foi gerado, apresentando a precisão, recall e f1-score para cada classe:

```
              precision    recall  f1-score   support

       aviao       0.73      0.78      0.75      1000
   automovel       0.82      0.84      0.83      1000
     passaro       0.70      0.50      0.58      1000
        gato       0.56      0.50      0.53      1000
       veado       0.61      0.71      0.66      1000
    cachorro       0.57      0.69      0.63      1000
        sapo       0.79      0.76      0.77      1000
      cavalo       0.74      0.77      0.76      1000
       navio       0.83      0.81      0.82      1000
    caminhao       0.81      0.78      0.79      1000
```


