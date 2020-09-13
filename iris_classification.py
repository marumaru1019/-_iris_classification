import tensorflow as tf
import numpy as np#数値計算のライブラリ
import pandas as pd#データ分析のライブラリ
import matplotlib.pyplot as plt#グラフ描画のライブラリ
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json


def collect_iris():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["label"] = iris.target

    data_X = preprocessing.scale(df.iloc[:,0:-1])#大文字を使っているのは複数のx1~x3で成り立つ行列だから
    data_y = to_categorical(df.iloc[:,-1])#one-hotベクトル化
    #トレーニングデータとテストデータに分類
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=1)
    return X_train,X_test,y_train,y_test


def iris_learning(X_train,y_train):
    model= Sequential()#モデルの宣言
    model.add(Dense(100,activation ="relu",input_dim=4))#1層目
    model.add(Dense(100,activation="relu"))#2層目
    model.add(Dense(3,activation="softmax"))#3層目

    model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics = ["accuracy"],
                )
    
    history = model.fit(X_train,y_train,
                epochs=20,
                verbose=2#出力形式→0:表示なし,1;プログラスバー,2:1行の簡潔なログ
                )
    
    return model,history


def iris_predict(X_test,y_test,history,model):
    accuracy = history.history["accuracy"]
    loss = history.history["loss"]
    epochs = range(1, len(loss) + 1)
    #accuracyの可視化
    plt.plot(epochs,accuracy,color="red",label="Training accuracy")
    plt.title("Training accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Training accuracy")
    plt.legend()
    plt.show()
    #lossの可視化
    plt.plot(epochs, loss, label="Training loss")
    plt.title("Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    #テストデータで検証
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("loss:",loss)
    print("accuracy:",accuracy)

def model_save(model):
    model_json = model.to_json()
    with open('model_iris100.json', 'w') as file:
        file.write(model_json)
        model.save_weights('weights.hdf5')

def model_call():
    with open('model_iris100.json', 'r') as file:
        model_json = file.read()
        model = model_from_json(model_json)

    model.load_weights('weights.hdf5')
    return model

def main():
    X_train,X_test,y_train,y_test=collect_iris()
    iris_model,iris_history = iris_learning(X_train,y_train)
    model_save(iris_model)
    model=model_call()
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    iris_pre = iris_predict(X_test,y_test,iris_history,model)


if __name__ == "__main__":
    main()




