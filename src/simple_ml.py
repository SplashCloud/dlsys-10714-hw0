import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # 1. ungzip the file
    with gzip.open(image_filename) as f:
      magic_number = f.read(4)
      N_images = struct.unpack(">i", f.read(4))[0]
      row = struct.unpack(">i", f.read(4))[0]
      col = struct.unpack(">i", f.read(4))[0]
      print(f'N_images={N_images}, row={row}, col={col}')
      X = []
      for _ in range(N_images):
        item = []
        for _ in range(row):
          for _ in range(col):
            pixel = struct.unpack(">B", f.read(1))[0]
            item.append(pixel)
        X.append(item)
    
    with gzip.open(label_filename) as f:
      magic_number = f.read(4)
      N_labels = struct.unpack(">i", f.read(4))[0]
      print(f'N_labels={N_labels}')
      y = []
      for _ in range(N_labels):
        label = struct.unpack(">B", f.read(1))[0]
        y.append(label)
    
    def norm(arr):
      min_v = np.min(arr)
      max_v = np.max(arr)
      return (arr - min_v) / (max_v - min_v)
    
    X = np.array(X, dtype=np.float32) 
    return norm(X), np.array(y, dtype=np.uint8)
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    return np.mean(
          np.log(
            np.sum(
              np.exp(
                Z - Z[np.arange(Z.shape[0]), y].reshape(-1, 1) # z_i - z_y
              ),
              axis=1
            )
          )
        )
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    # 1. partition all examples into batches
    n = X.shape[0]
    total_batch = []
    batch_num = n // batch
    i = 1
    while i <= batch_num:
      total_batch.append(
        (
          X[(i-1)*batch : i*batch],
          y[(i-1)*batch : i*batch]
        )
      )
      i += 1
    if n % batch != 0:
      total_batch.append(
        (
          X[(i-1)*batch:],
          y[(i-1)*batch:]
        )
      )
    # 2.begin to iterate the batches and update the theta
    
    def norm(arr):
      arr = np.exp(arr)
      return arr / np.sum(arr)

    def calc_gradient(X, theta, y):
      Z = np.apply_along_axis(norm, axis=1, arr=np.matmul(X, theta))
      I = np.zeros(shape=Z.shape)
      I[np.arange(Z.shape[0]),y] = 1
      return np.matmul(np.transpose(X), Z-I) / Z.shape[0]

    loss = 0.0
    for one_batch in total_batch:
      theta -= lr * calc_gradient(one_batch[0], theta, one_batch[1])
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    n = X.shape[0]
    batch_num = n // batch
    total_data = []
    i = 1
    while i <= batch_num:
      total_data.append(
        (
          X[(i-1)*batch : i*batch],
          y[(i-1)*batch : i*batch]
        )
      )
      i += 1
    if n % batch != 0:
      total_data.append(
        (
          X[(i-1)*batch : ],
          y[(i-1)*batch : ]
        )
      )

    def relu(arr):
      return np.maximum(0, arr)

    def norm(arr):
      arr = np.exp(arr)
      return arr / np.sum(arr)

    for one_batch in total_data:
      X0, y0 = one_batch
      Z1 = relu(np.matmul(X0, W1))
      Z2 = np.matmul(Z1, W2)
      Iy = np.zeros(shape=Z2.shape)
      Iy[np.arange(Iy.shape[0]),y0] = 1
      G2 = np.apply_along_axis(norm, axis=1, arr=Z2) - Iy
      G1 = np.where(Z1 > 0, 1, 0) * np.matmul(G2, np.transpose(W2))

      W1 -= lr * np.matmul(np.transpose(X0), G1) / X0.shape[0]
      W2 -= lr * np.matmul(np.transpose(Z1), G2) / X0.shape[0]
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
