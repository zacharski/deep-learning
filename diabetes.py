import pandas as pd
import numpy as np

class Diabetes:
    currentBucket = 0
    
    def __init__(self):
        data = pd.read_csv('https://raw.githubusercontent.com/zacharski/machine-learning/master/data/diabetes.csv')
        X = np.array(data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']])
        Y = np.array(data[['Outcome']])
        self.trainX = X[:500]
        self.testX = X[500:]
        self.trainY = Y[:500]
        self.testY = Y[500:]

    def next_batch(self):
        print(self.currentBucket)
        x = self.trainX[self.currentBucket:self.currentBucket+50]
        y = self.trainY[self.currentBucket:self.currentBucket+50]
        self.currentBucket = (self.currentBucket + 50) % 500
        
        return (x, y)


if __name__ == "__main__":
	diabetes = Diabetes()
	batch_X, batch_Y = diabetes.next_batch()
	print(batch_X[:3])
	batch_X, batch_Y = diabetes.next_batch()
	print(batch_X[:3])
	
