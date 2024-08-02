from sklearn.model_selection import train_test_split
import pandas as pd


data = {
    'script': [
        "The hero enters the dark cave. A sudden noise startles everyone. The villain reveals his plan. A touching reunion scene.",
        "A brave knight sets out on a quest. He encounters various challenges. A hidden truth is unveiled. A dramatic showdown ensues."
    ],
    'plot': [4, 5],
    'emotion': [5, 4],
    'lore': [3, 4],
    'dialogue': [4, 5],
    'review': [
        "The plot is engaging with a few twists. The emotional depth is well-captured. It fits well within the franchise lore. Dialogues are well-written.",
        "An epic quest with a strong plot. Emotions are high and well-portrayed. It enhances the lore of the series. Dialogues are impressive."
    ]
}

df = pd.DataFrame(data)

X = df['script'].values
y_plot = df['plot'].values
y_emotion = df['emotion'].values
y_lore = df['lore'].values
y_dialogue = df['dialogue'].values
y_review = df['review'].values

X_train, X_test, y_train_plot, y_test_plot = train_test_split(X, y_plot, test_size=0.2, random_state=42)
_, _, y_train_emotion, y_test_emotion = train_test_split(X, y_emotion, test_size=0.2, random_state=42)
_, _, y_train_lore, y_test_lore = train_test_split(X, y_lore, test_size=0.2, random_state=42)
_, _, y_train_dialogue, y_test_dialogue = train_test_split(X, y_dialogue, test_size=0.2, random_state=42)
_, _, y_train_review, y_test_review = train_test_split(X, y_review, test_size=0.2, random_state=42)
