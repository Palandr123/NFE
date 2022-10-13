from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


def get_boundary(vectors, attributes, feature, freq_bound=None):
    X = vectors[~attributes[feature].isna()]
    y = attributes[~attributes[feature].isna()][feature]
    if freq_bound is not None:
        freq = y.value_counts(normalize=True)
        less_freq = freq[freq <= freq_bound]
        y.loc[y.isin(less_freq.index.tolist())] = "other"
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    svm = SVC(kernel="linear")
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_train)
    print(f"Train accuracy: {accuracy_score(y_train, y_pred)}")
    y_pred = svm.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred)}")
    filename = f'svm_{feature}.sav'
    pickle.dump(svm, open(filename, 'wb'))
    return svm
