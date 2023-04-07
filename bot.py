# Importing all the required libraties

import pickle
import tensorflow as tf
import numpy as np
import json

# Importing the trained AI model

def get_model(path, value):
    
    words = pickle.load(open(path+"\\" + "words.p", "rb"))
    bag = [0] * len(words)
    for i ,word in enumerate(words):
        if  word in value:
            bag[i] = 1

    model = tf.keras.models.load_model(path +"\\"+ "model.h5")

    pred = model.predict(np.array([bag]));
    pred = tf.argmax(pred,1)

    pred = np.array(pred)
    return pred


# Predicting the value using the AI model

def predict(path, value):
    
    pred = get_model(path, value)
    d = open(path + "\\" +"data.json")
    json_data = json.load(d)

    answer = []
    for data in json_data['intends']:
        for ans in data["answer"]:
            answer.append(np.array(ans))
            
    print("\n")
    print(answer[pred[0]])
    
    return pred[0]
    


# Taking input from the user

def chat(path):
    a = True
    while(a == True):
        value = input("\n")
        pred = predict(path,value.lower())
        classes = pickle.load(open(path+"\\" + "classes.p", "rb"))
        
        if classes[pred] == classes[-1]:
            a = False



path = "C:\\Users\\USER\Documents\ML\\NLP\\bot\src"
chat(path)
