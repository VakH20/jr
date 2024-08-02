import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model

# Загрузка модели ELMo
elmo = hub.load("https://tfhub.dev/google/elmo/3")

def elmo_embedding(x):
    return elmo.signatures['default'](tf.constant(x))['elmo']

# Входной слой
input_text = Input(shape=(1,), dtype=tf.string)

# ELMo эмбеддинги
embedding = Lambda(lambda x: elmo.signatures['default'](x)['elmo'])(input_text)
embedding = Lambda(lambda x: tf.reduce_mean(x, axis=1))(embedding)

# Полносвязные слои для каждой оценки
plot_output = Dense(1, activation='linear', name='plot_output')(embedding)
emotion_output = Dense(1, activation='linear', name='emotion_output')(embedding)
lore_output = Dense(1, activation='linear', name='lore_output')(embedding)
dialogue_output = Dense(1, activation='linear', name='dialogue_output')(embedding)
review_output = Dense(1, activation='linear', name='review_output')(embedding)

# Создание модели
model = Model(inputs=input_text, outputs=[plot_output, emotion_output, lore_output, dialogue_output, review_output])

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Подготовка данных для обучения (ELMo требует входной формат в виде списка строк)
X_train = X_train.tolist()
X_test = X_test.tolist()

# Обучение модели
model.fit(X_train, [y_train_plot, y_train_emotion, y_train_lore, y_train_dialogue, y_train_review], epochs=10, batch_size=2)


# Оценка модели
loss, plot_loss, emotion_loss, lore_loss, dialogue_loss, review_loss, plot_acc, emotion_acc, lore_acc, dialogue_acc, review_acc = model.evaluate(X_test, [y_test_plot, y_test_emotion, y_test_lore, y_test_dialogue, y_test_review])
print(f"Plot Accuracy: {plot_acc*100}%")
print(f"Emotion Accuracy: {emotion_acc*100}%")
print(f"Lore Accuracy: {lore_acc*100}%")
print(f"Dialogue Accuracy: {dialogue_acc*100}%")
print(f"Review Accuracy: {review_acc*100}%")

# Пример генерации рецензии
def generate_review(script):
    script = [script]
    embeddings = elmo_embedding(script)
    embedding = tf.reduce_mean(embeddings, axis=1)
    plot_score = model.get_layer('plot_output')(embedding)
    emotion_score = model.get_layer('emotion_output')(embedding)
    lore_score = model.get_layer('lore_output')(embedding)
    dialogue_score = model.get_layer('dialogue_output')(embedding)
    review_score = model.get_layer('review_output')(embedding)
    
    # Сформировать рецензию на основе оценок
    review = f"Plot Score: {plot_score.numpy()[0]:.2f}, Emotion Score: {emotion_score.numpy()[0]:.2f}, Lore Score: {lore_score.numpy()[0]:.2f}, Dialogue Score: {dialogue_score.numpy()[0]:.2f}.\nOverall review: {review_score.numpy()[0]}"
    return review

# Пример использования
script = "A brave knight sets out on a quest. He encounters various challenges. A hidden truth is unveiled. A dramatic showdown ensues."
review = generate_review(script)
print(review)

model.save('script_analysis_model.h5')
