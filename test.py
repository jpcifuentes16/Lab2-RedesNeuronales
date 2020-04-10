import numpy as np
import mnist_reader
from fns_redes_neuronales import *
import pickle
import pandas as pd

def ConfusionMatrix(real,prediccion):
	actual=[
		int(i)
		for i in real
		]

	prediccionVect=[
			int(i)
			for i in prediccion
			]



	y_actu = pd.Series(actual, name='Actual')
	y_pred = pd.Series(prediccionVect, name='Predicted')
	df_confusion = pd.crosstab(y_actu, y_pred)
	print(df_confusion)

def Test(factNormalizar,neuronasOcultas,nombreModelo):
	#Constantes
	'''
	Factor por el que se divide para
	que no baje tan rapido la sigmoide
	'''
	NORMALIZAR=factNormalizar

	#Cantidad de neuronas en la
	#Capa oculta
	NEURONAS_OCULTAS=neuronasOcultas

	#Neuronas en la capa salida
	NEURONAS_SALIDA=10


	#Lectura de datos
	print("Lectura de datos...")
	X_test, y_test = mnist_reader.load_mnist ( './data/fashion' , kind = 't10k' )


	#TEST

	modeloCarga=open(nombreModelo,'rb')
	flat_thetas=pickle.load(modeloCarga)
	modeloCarga.close()


	X=X_test/NORMALIZAR

	m,n=X.shape

	#Y son las labels de cada imagen
	y=y_test.reshape(m,1)

	Y=(y==np.array(range(10))).astype(int)


	#Se construye el modelo
	theta_shapes=np.array([
		[NEURONAS_OCULTAS,n+1],
		[NEURONAS_SALIDA,NEURONAS_OCULTAS+1]
		])



	resultado=feed_forward(
	        inflate_matrixes(flat_thetas, theta_shapes),
	        X
	    )

	prediccion=np.argmax(resultado[-1], axis=1).reshape(m,1)


	correctos=((prediccion==y)*1).sum()
	incorrectos=len(y)-correctos

	print("Correctas "+str(correctos))
	print("Incorrectos "+str(incorrectos))
	print("Exactitud "+str(correctos*100/float(len(y))))

	return y,prediccion

y,prediccion=Test(1000,533,'modeloFinal2')
ConfusionMatrix(y,prediccion)
