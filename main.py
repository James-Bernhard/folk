# Preprocessing and analysis of Irish folk music data
# by James Bernhard
# 2023-11-17

import keras
import pandas as pd
from pathlib import Path
import music21
import numpy as np
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import matplotlib.pyplot as plt

# variables to control how the computations are done, saved, and loaded
raw_data_file = Path("tunes.csv")
load_data = False
save_data = True
processed_data_file = Path("df.pkl")
load_fitted_model = False
save_fitted_model = True
fitted_model_file = Path("fitted_model.keras")
save_confusion_matrix = False
confusion_matrix_file = Path("confusion_matrix.pdf")
random_seed = 100


def compile_data(file: Path, n_tunes: int = 0) -> pd.DataFrame:
    df = pd.read_csv(file)
    if n_tunes > 0:
        df = df.iloc[0:n_tunes, ]

    df = df.assign(note_length=df["meter"].map(lambda x: "1/" + x[-1]))
    df = df.assign(pre="M: " + df["meter"] + "\n" + "L: " + df["note_length"] + "\n" + "K: " + df["mode"] + "\n")
    df = df.assign(full_abc=df["pre"] + df["abc"])
    df = df.sort_values(by="setting_id", axis=0)

    def get_durations(abc: str, n_notes: int = None):
        def get_duration(note):
            return note.duration.quarterLength

        notes = music21.converter.parse(abc, format="abc").recurse().notesAndRests
        if n_notes is not None:
            notes = notes[:n_notes]
        return list(map(get_duration, notes))

    def get_all_durations(column: pd.Series) -> pd.Series:
        durations = len(column) * [[]]
        for i in range(len(column)):
            if i % 1000 == 0:
                print(f"Processing row {i}")
            try:
                durations[i] = get_durations(column.iloc[i])
            except:
                durations[i] = [0]
        return pd.Series(durations, index=df.index)

    df = df.assign(durations=get_all_durations(df["full_abc"]))
    return df


# compile the data
if load_data:
    df: pd.DataFrame = pd.read_pickle(Path(processed_data_file))
else:
    df: pd.DataFrame = compile_data(raw_data_file)

if save_data:
    df.to_pickle(processed_data_file)
    print(f"Processed data saved as {processed_data_file.name}.")

# omit the rows where the abc notation couldn't be processed
df = df.loc[df["durations"].map(len) > 1]

# make sure the tune type is categorical
df["type"] = df["type"].astype("category")

# assemble the data
df = df.assign(meter_split=list(map(lambda x: [x.split("/")[0], x.split("/")[1]], df["meter"])))
df = df.assign(meter_and_durations=df["meter_split"] + df["durations"])

X = tf.keras.utils.pad_sequences(df["meter_and_durations"], padding="post", truncating="post", maxlen=128)
y_categorical = df["type"].values
enc = sklearn.preprocessing.OneHotEncoder(sparse_output=False)
y = enc.fit_transform(y_categorical.reshape(-1, 1))

# separate into training, validation, and test data
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=100)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=100)


def reshape_X(input):
    """
    Reshape X for model fitting and prediction.
    """
    return input.reshape(input.shape[0], input.shape[1], 1)


def compile_model(input_shape):
    """
    Compile the model, which will then be fitted.
    """
    inputs = keras.Input(shape=input_shape)
    x = tf.keras.layers.Masking()(inputs)
    x = tf.keras.layers.GRU(units=64, return_sequences=True)(x)
    x = tf.keras.layers.GRU(units=32, return_sequences=True)(x)
    x = tf.keras.layers.GRU(units=16)(x)
    x = tf.keras.layers.Dense(units=16)(x)
    x = tf.keras.layers.Dense(units=8)(x)
    outputs = keras.layers.Dense(len(df["type"].cat.categories), activation="softmax")(x)
    output_model = tf.keras.Model(inputs, outputs)

    output_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                         loss="categorical_crossentropy",
                         metrics=[tf.keras.metrics.F1Score(average="weighted"),
                                  tf.keras.metrics.CategoricalAccuracy()]
                         )

    return output_model


def fit_model(model, X_train, y_train, X_val, y_val, epochs=1):
    """
    Fit the model that has been compiled by compile_model.
    """
    model.fit(reshape_X(X_train), y_train,
              epochs=epochs,
              validation_data=(X_val, y_val))
    return model


def assess_model(model, X_val, y_val):
    """
    Assess the fitted model with a confusion matrix (that can be displayed or saved).
    """
    pred = np.argmax(model.predict(reshape_X(X_val)), axis=1)
    y_val_num = np.argmax(y_val, axis=1)
    cm = confusion_matrix(y_val_num, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df["type"].cat.categories.values)
    plt.rcParams["figure.figsize"] = [16, 16]
    disp.plot()


# Do the computations for the analysis

if load_fitted_model:
    model = tf.keras.models.load_model(fitted_model_file)
else:
    tf.random.set_seed(random_seed)
    model = compile_model(input_shape=(X_train.shape[1], 1))
    history = fit_model(model, X_train, y_train, X_val, y_val, epochs=5)

if save_fitted_model:
    model.save(fitted_model_file)
    print(f"Fitted model saved as {fitted_model_file.name}.")

assess_model(model, X_val, y_val)

if save_confusion_matrix:
    plt.savefig(confusion_matrix_file)
    print(f"Model's confusion matrix saved as {confusion_matrix_file.name}.")
else:
    plt.show()
