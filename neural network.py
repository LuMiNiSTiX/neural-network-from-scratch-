import cupy as np
from sklearn.datasets import fetch_openml


class neuronlayer:
    def __init__(self, input_dim, output_dim, weights=None, bias=None, use_softmax=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_softmax = use_softmax
        


        if weights is None:
            self.weight = np.random.randn(self.input_dim, self.output_dim) * np.sqrt(2/ self.input_dim).astype(np.float32)
        
        else:
            self.weight = weights

        if bias is None:

            self.bias = np.zeros((1,output_dim), dtype=np.float32)
        else:
            self.bias = bias

    def forward(self, inputs):
        self.X = inputs
        self.T= np.transpose(self.X)

        self.Z = np.dot(inputs, self.weight) + self.bias


        if self.use_softmax:
            self.amount= self.softy(self.Z)
        else:
            self.amount=self.activation(self.Z)
        return self.amount

    def activation(self, inp):
        return np.maximum(0.001 * inp, inp)

    def acti_deri(self, inp):
        return np.where(inp > 0, 1, 0.001)

    def softy(self, inp):
        e = np.exp(inp - np.max(inp, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
    def backout(self,Y):
        batch=self.X.shape[0]
        self.dZ= self.amount-Y
        self.dW = np.dot(self.T, self.dZ) / batch
        self.dB = np.sum(self.dZ, axis=0, keepdims=True) /batch

    def backhand(self, learning_rate):
        self.weight -= learning_rate * self.dW
        self.bias -= learning_rate * self.dB
    def backlayer(self,dZnx,Wnx):
        batch=self.X.shape[0]
        err=np.dot(dZnx,Wnx.T)
        self.dZ=err*self.acti_deri(self.Z)
        self.dW=np.dot(self.T,self.dZ)/batch
        self.dB=np.sum(self.dZ,axis=0,keepdims=True)/batch
        return self.dZ 

         
        
class Network:
    def __init__(self,lay):
        self.lay=lay

    def forward(self,X):
        for layer in self.lay:
            X=layer.forward(X)
        return X
    def backward(self,Y):
        for i in range(len(self.lay)-1,-1,-1):
            if i==len(self.lay)-1:
                self.lay[i].backout(Y)
            else:
                self.lay[i].backlayer(self.lay[i+1].dZ,self.lay[i+1].weight)

        
    def traincar(self, X,Y,learningrate=0.01,epochs=10,batch_size=64):
        sample=X.shape[0]

        for u in range(epochs):
            ind=np.arange(int(sample))
            np.random.shuffle(ind)
            X_=X[ind]
            Y_=Y[ind]
            for i in range(0,sample,batch_size):
                X_batch=X_[i:i+batch_size]
                Y_batch=Y_[i:i+batch_size]
                out = self.forward(X_batch)
                self.backward(Y_batch)
                for layer in self.lay:
                    layer.backhand(learningrate)
                # 3. Calculate epoch loss
            epoch_out = self.forward(X)
            epoch_loss = -np.mean(np.sum(Y * np.log(epoch_out + 1e-8), axis=1))
            print(f"Epoch {u + 1}/{epochs} - Loss: {float(epoch_loss):.4f}")


if __name__ == "__main__":
    print("Fetch MNIST ")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    X_raw = mnist.data
    y_raw = mnist.target.astype(int)

    X_gpu = np.asarray(X_raw / 255.0)
    y_gpu = np.asarray(y_raw)

    Y_onehot = np.zeros((y_gpu.shape[0], 10), dtype=np.float32)
    Y_onehot[np.arange(y_gpu.shape[0]), y_gpu] = 1.0


    X_train, X_test = X_gpu[:60000], X_gpu[60000:]
    Y_train, Y_test = Y_onehot[:60000], Y_onehot[60000:]


    layers = [
        neuronlayer(784, 128 , use_softmax=False),
        neuronlayer(128, 10, use_softmax=True)
    ]

    model = Network(layers)

    print("Trainin MNIST pics...")
    model.traincar(X_train, Y_train, learningrate=0.1, epochs=15, batch_size=64)


    test_preds = model.forward(X_test)
    predicted_labels = np.argmax(test_preds, axis=1)
    true_labels = np.argmax(Y_test, axis=1)

    accuracy = np.mean(predicted_labels == true_labels) * 100
    print(f"\n Accuracy: {float(accuracy):.2f}%")
    


    
    




































































































        
         
