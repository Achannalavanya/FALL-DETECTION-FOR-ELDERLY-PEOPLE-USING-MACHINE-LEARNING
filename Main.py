from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import time
import webbrowser
from sklearn.ensemble import RandomForestClassifier

main = tkinter.Tk()
main.title("FALL DETECTION FOR ELDERLY PEOPLE USING MACHINE LEARNING") 
main.geometry("1300x1200")

global filename
global X, Y
global svm_accuracy, dt_accuracy, svm_train_time, dt_train_time,svm_test_time, dt_test_time, rf_accuracy, rf_train_time, rf_test_time
global X_train, X_test, y_train, y_test
global dataset
global scaler
global classifier

label = ['Standing','Walking','Sitting','Falling','Cramps','Running']

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")#uploading dataset
    dataset = pd.read_csv(filename) #reading dataset from loaded file
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    text.insert(END,str(dataset.head()))
    label = dataset.groupby('ACTIVITY').size()#ploting graph with number of on time and default payment with class label as 0 and 1
    label.plot(kind="bar")
    plt.show()

def preprocessDataset():
    global dataset
    global X, Y
    global X_train, X_test, y_train, y_test
    global scaler
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True) #replacing missing or NA values with 0 in dataset
    dataset = dataset.values #converting entire dataset into values and assign to X
    Y = dataset[:,0]
    X = dataset[:,1:dataset.shape[1]]
    scaler = MinMaxScaler() 
    scaler.fit(X) #applying MIX-MAX function on dataset to preprocess dataset
    X = scaler.transform(X)
    print(X.shape)
    text.insert(END,str(X)+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset Train & Test Data Split\n\n")
    text.insert(END,"80% dataset records used for ML training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used for ML training : "+str(X_test.shape[0])+"\n")    

def SVMAlgorithm():
    global X, Y
    global svm_accuracy, svm_train_time, svm_test_time
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    start = time.time()
    cls = svm.SVC()
    cls.fit(X,Y)
    end = time.time()
    svm_train_time = end - start
    start = time.time()
    svm_predict = cls.predict(X_test)
    svm_accuracy = accuracy_score(svm_predict, y_test)
    end = time.time()
    svm_test_time = end - start
    text.insert(END,"\nSVM Accuracy   : "+str(svm_accuracy)+"\n")
    text.insert(END,"SVM Train Time : "+str(svm_train_time)+"\n")
    text.insert(END,"SVM Test Time  : "+str(svm_test_time)+"\n")


def DTAlgorithm():
    global classifier
    global X, Y
    global dt_accuracy, dt_train_time, dt_test_time
    global X_train, X_test, y_train, y_test
    start = time.time()
    classifier = DecisionTreeClassifier()
    classifier.fit(X, Y)
    end = time.time()
    dt_train_time = end - start
    start = time.time()
    svm_predict = classifier.predict(X_test)
    dt_accuracy = accuracy_score(svm_predict, y_test)
    end = time.time()
    dt_test_time = end - start
    text.insert(END,"\nDecision Tree Accuracy : "+str(dt_accuracy)+"\n")
    text.insert(END,"Decision Tree Train Time : "+str(dt_train_time)+"\n")
    text.insert(END,"Decision Tree Test Time  : "+str(dt_test_time)+"\n\n")

def RFAlgorithm():
    global X, Y
    global rf_accuracy, rf_train_time, rf_test_time
    global X_train, X_test, y_train, y_test
    start = time.time()
    classifier = RandomForestClassifier()
    classifier.fit(X, Y)
    end = time.time()
    rf_train_time = end - start
    start = time.time()
    svm_predict = classifier.predict(X_test)
    rf_accuracy = accuracy_score(svm_predict, y_test)
    end = time.time()
    rf_test_time = end - start
    text.insert(END,"\nRandom Forest Tree Accuracy : "+str(rf_accuracy)+"\n")
    text.insert(END,"Random Forest Train Time : "+str(rf_train_time)+"\n")
    text.insert(END,"Random Forest Test Time  : "+str(rf_test_time)+"\n\n")
    


def graph():
    output = "<html><body><table align=center border=1><tr><th>Algorithm Name</th><th>Accuracy</th><th>Training Time</th><th>Prediction Time</th></tr>"
    output+="<tr><td>SVM Algorithm</td><td>"+str(svm_accuracy)+"</td><td>"+str(svm_train_time)+"</td><td>"+str(svm_test_time)+"</td></tr>"
    output+="<tr><td>Random Forest Algorithm</td><td>"+str(rf_accuracy)+"</td><td>"+str(rf_train_time)+"</td><td>"+str(rf_test_time)+"</td></tr>"
    output+="<tr><td>Decision Tree Algorithm</td><td>"+str(dt_accuracy)+"</td><td>"+str(dt_train_time)+"</td><td>"+str(dt_test_time)+"</td></tr></table></body></html>"
    f = open("table.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("table.html",new=2)
    
    height = [svm_accuracy,dt_accuracy, rf_accuracy]
    bars = ('SVM Accuracy','Decision Tree Accuracy','Random Forest Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()


def predict():
    text.delete('1.0', END)
    global scaler
    global classifier
    testfile = filedialog.askopenfilename(initialdir = "Dataset")
    dataset = pd.read_csv(testfile)
    dataset = dataset.values 
    temp = dataset
    test = dataset[:,0:dataset.shape[1]]
    test = scaler.transform(test)
    print(test.shape)
    predict = classifier.predict(test)
    print(predict)
    for i in range(len(temp)):
        text.insert(END,"Test Record = "+str(temp[i])+" PREDICTION = "+label[int(predict[i])]+"\n\n")
    

font = ('times', 16, 'bold')
title = Label(main, text="FALL DETECTION FOR ELDERLY PEOPLE USING MACHINE LEARNING")
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Fall Detection Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Features Calculation & Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=360,y=550)
preprocessButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=SVMAlgorithm)
svmButton.place(x=720,y=550)
svmButton.config(font=font1) 

dtButton = Button(main, text="Run Decision Tree Algorithm", command=DTAlgorithm)
dtButton.place(x=50,y=600)
dtButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest Algorithm", command=RFAlgorithm)
rfButton.place(x=360,y=600)
rfButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=720,y=600)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Fall from Test Data", command=predict)
predictButton.place(x=50,y=650)
predictButton.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
