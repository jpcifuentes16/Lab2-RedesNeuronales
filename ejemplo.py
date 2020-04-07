import mnist_reader
#from PIL import Image

X_train, y_train = mnist_reader.load_mnist ( './data/fashion' , kind = 'train' )
X_test, y_test = mnist_reader.load_mnist ( './data/fashion' , kind = 't10k' )
'''
imagen=X_train[1]


im= Image.new("RGB", (28, 28), "#FF0000")
pixels = im.load()
contador=0
for i in range(0,28):
	for j in range(0,28):
		pixels[j,i] = (imagen[contador],imagen[contador],imagen[contador])
		contador+=1
im.show()
'''
for i in y_train:
	print(i)