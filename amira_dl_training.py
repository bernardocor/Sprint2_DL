import json
import pickle
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers


def main():
    # Cargar datos de conversaciones
    with open('conversations.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Preparar datos: patrones y etiquetas
    patterns = []
    tags = []
    tag_response_dict = {}

    for item in data:
        tag = item['tag']
        tag_response_dict[tag] = item['responses']
        for pattern in item['patterns']:
            patterns.append(pattern)
            tags.append(tag)

    # Guardar diccionario de respuestas por categoría
    with open('conversations_category_answers.json', 'w', encoding='utf-8') as f:
        json.dump(tag_response_dict, f, ensure_ascii=False, indent=2)

    # Vectorización de patrones (Bag of Words)
    vectorizer = CountVectorizer(lowercase=True)
    x_matrix = vectorizer.fit_transform(patterns).toarray()

    # Guardar el vectorizador
    with open('conversations_vectorizer_bow.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Codificación de etiquetas (One Hot)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(tags)
    num_classes = len(label_encoder.classes_)
    y_cat = keras.utils.to_categorical(y, num_classes=num_classes)

    # Guardar el orden de las categorías en Pickle
    with open('conversations_categories.pkl', 'wb') as f:
        pickle.dump(list(label_encoder.classes_), f)

    # Crear y entrenar el modelo de Deep Learning (red simple)
    model = keras.Sequential()
    model.add(layers.Input(shape=(x_matrix.shape[1],)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(x_matrix, y_cat, epochs=200, batch_size=4, verbose=0)

    # Guardar el modelo entrenado
    model.save('conversations_dl_model.h5')

    # Predicción de ejemplo: "Mi signo es Tauro"
    def predict_tag(text):
        x_input = vectorizer.transform([text]).toarray()
        pred = model.predict(x_input, verbose=0)
        tag_pred = label_encoder.inverse_transform([np.argmax(pred)])
        return tag_pred[0]

    input_text = "Mi signo es Tauro"
    predicted_tag = predict_tag(input_text)
    possible_answers = tag_response_dict[predicted_tag]
    selected_answer = random.choice(possible_answers)

    print(f'Usuario: {input_text}')
    print(f'ChatBot: {selected_answer}')


if __name__ == "__main__":
    main()
