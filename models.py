import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset


class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        product = nn.DotProduct(x, self.w)
        return product

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        product = self.run(x)
        if nn.as_scalar(product) >= 0 :
            pred = 1
        else :
            pred = -1
        return pred

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        convergence = 0      
        while convergence != 1 :    # si un point est mal classé, on continue l'apprentissage
            convergence = 1
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y) : # on a trouvé un point mal classé  
                    # mise à jour des poids
                    self.w.update(x,nn.as_scalar(y))
                    convergence = 0                           # on remet la variable de convergence à 0


        


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialisation des paramètres du modèle avec des dimensions appropriées
        self.W1 = nn.Parameter(1,50)
        self.b1 = nn.Parameter(1,50)

        self.W2 = nn.Parameter(50,25)
        self.b2 = nn.Parameter(1,25)

        self.W3 = nn.Parameter(25,1)
        self.b3 = nn.Parameter(1,1)

        

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # première couche
        pred = nn.Linear(x,self.W1)
        pred = nn.AddBias(pred, self.b1)
        pred = nn.ReLU(pred)

        # deuxième couche
        pred = nn.Linear(pred, self.W2)
        pred = nn.AddBias(pred, self.b2)
        pred = nn.ReLU(pred)

        pred = nn.Linear(pred, self.W3)
        pred = nn.AddBias(pred, self.b3)

        return pred

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        # hyperparamètres
        batch_size = 100
        alpha = 0.01      #learning rate
        
        for x,y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(x,y)
            
            # calcul des gradients
            grad = nn.gradients(loss,[self.W1,self.b1,self.W2,self.b2,self.W3,self.b3])
            
            # mise à jour des paramètres
            self.W1.update(grad[0],-alpha)
            self.b1.update(grad[1],-alpha)
            self.W2.update(grad[2],-alpha)
            self.b2.update(grad[3],-alpha)
            self.W3.update(grad[4],-alpha)
            self.b3.update(grad[5],-alpha)
            #print("loss: ",nn.as_scalar(loss))
            if nn.as_scalar(loss) < 0.01 : # on prend une borne légèrement inférieure à 0.02 pour avoir une meilleure garantie d'obtenir une perte moyenne de moins de 0.02
                break
        
        


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        self.W1 = nn.Parameter(784, 100)
        self.b1 = nn.Parameter(1, 100)

        self.W2 = nn.Parameter(100, 50)
        self.b2 = nn.Parameter(1, 50)

        self.W3 = nn.Parameter(50, 10)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # première couche
        pred = nn.Linear(x,self.W1)
        pred = nn.AddBias(pred, self.b1)
        pred = nn.ReLU(pred)

        # deuxième couche
        pred = nn.Linear(pred, self.W2)
        pred = nn.AddBias(pred, self.b2)
        pred = nn.ReLU(pred)

        pred = nn.Linear(pred, self.W3)
        pred = nn.AddBias(pred, self.b3)
        
        return pred

        
        

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        # hyperparamètres
        batch_size = 250
        alpha = 0.4      #learning rate
        
        for x,y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(x,y)
            
            # calcul des gradients
            grad = nn.gradients(loss,[self.W1,self.b1,self.W2,self.b2,self.W3,self.b3])
            
            # mise à jour des paramètres
            self.W1.update(grad[0],-alpha)
            self.b1.update(grad[1],-alpha)
            self.W2.update(grad[2],-alpha)
            self.b2.update(grad[3],-alpha)
            self.W3.update(grad[4],-alpha)
            self.b3.update(grad[5],-alpha)
            #print("accuracy: ",dataset.get_validation_accuracy())
            
            #condition d'arrêt
            if dataset.get_validation_accuracy() > 0.9745 :   # on prend une borne légèrement supérieur à 97% pour avoir une meilleure garantie de dépasser 97% sur l'ensemble de test
                break                                         # mais pas trop haute pour ne pas avoir un temps de calcul trop élevé
