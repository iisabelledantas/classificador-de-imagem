import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import json
from sklearn.metrics import classification_report

# Carregar dados
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizar
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Nomes das classes
class_names = ['aviao', 'automovel', 'passaro', 'gato', 'veado',
               'cachorro', 'sapo', 'cavalo', 'navio', 'caminhao']

# Salvar os nomes
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# Modelo CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Treinamento
model.fit(x_train, y_train_cat, epochs=10, validation_data=(x_test, y_test_cat))

# Obter previsões
y_pred_probs = model.predict(x_test)
y_pred_classes = y_pred_probs.argmax(axis=1)
y_true = y_test.flatten()

# Relatório com precisão, recall, f1-score e acurácia por classe
report = classification_report(y_true, y_pred_classes, target_names=class_names)
print("\nRelatório de classificação:\n")
print(report)


# Salvar modelo
model.save("cifar10_model.h5")
