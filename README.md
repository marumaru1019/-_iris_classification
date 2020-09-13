# iris_classification.pyの使い方
iris_classification.pyには授業で紹介するirisデータのmodel2を実行するためのコードが入っています。  
もしgoogle colab上で実行できない場合はこちらを実行して結果を確かめてください。  
ターミナル上での実行方法を以下に解説します。  

```  
user$ cd ./ファイルが入ったpassを指定
user$ python iris_classification.py  
```
実行すると 
```  
Epoch 1/20
105/105 - 1s - loss: 1.1133 - accuracy: 0.3524
Epoch 2/20
105/105 - 0s - loss: 0.9087 - accuracy: 0.8381
Epoch 3/20
105/105 - 0s - loss: 0.7533 - accuracy: 0.8571
・
・
・
```  

と学習が進み、  

<img src="https://user-images.githubusercontent.com/70362624/93015308-6b0b9280-f5f3-11ea-836f-384eb3bc9d1b.png" width="80%">  
<img src="https://user-images.githubusercontent.com/70362624/93015310-6d6dec80-f5f3-11ea-9b4a-bbf8c0002f50.png" width="80%">  
と表示されるはずです。
