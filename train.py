import numpy as np
from math import ceil
import pickle

# **********************optimizers******************************

def sgd(w, dw, config=None):
    """
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config

def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    beta1 = config['beta1']
    beta2 = config['beta2']
    config['t'] += 1
    config['m'] = beta1 * config['m'] + (1 - beta1) * dw
    config['v'] = beta2 * config['v'] + (1 - beta2) * (dw * dw)
    m_hat = config['m'] / (1 - beta1**config['t'])
    v_hat = config['v'] / (1 - beta2**config['t'])
    next_w = w - config['learning_rate'] * m_hat / (np.sqrt(v_hat) + config['epsilon'])

    return next_w, config

class Solver:
    def __init__(self, 
                model, 
                data, 
                num_epochs = 10, 
                optimizer = "sgd", 
                optim_config = {}, 
                lr_decay = 0.6, 
                batch_size = 100, 
                print_every = 50, 
                exp_num = 0
                ):
        
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        self.optimizer = globals()[optimizer]# convert str to function
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.print_every = print_every
        self.exp_num = exp_num
        
        # for checkpoint
        self.best_val_acc = 0
        self.best_params = {}

        #for visualization
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # in Adam, optim_configs of each parameter should be updated individually
        # so deep copy is needed
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in optim_config.items()}
            self.optim_configs[p] = d

    def check_accuracy(self, X, y, batch_size=100, get_label = False):
        """
        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.
        - get_label: If true, returns the predict labels.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        - y_pred: predict labels of X.
        """
        N = X.shape[0]

        # Compute predictions in batches
        iterations = ceil(N / batch_size)
        y_pred = []
        for i in range(iterations):
            scores = self.model.loss(X[i*batch_size:(i+1)*batch_size])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)

        if get_label:
            return y_pred
        else:
            acc = np.mean(y_pred == y)
            return acc

    def train(self):
        num_train = self.X_train.shape[0]
        iterations = ceil(num_train / self.batch_size)
        # flush historical data
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        for epoch in range(self.num_epochs):
            for iter in range(iterations):
                # get data from dataloader
                # X_batch = self.X_train[(iter*self.batch_size):((iter+1)*self.batch_size)]
                # y_batch = self.y_train[(iter*self.batch_size):((iter+1)*self.batch_size)]
                # batch_mask = np.random.choice(num_train, self.batch_size)
                # batch_mask = np.arange(start=iter, stop=num_train, step=iterations)
                batch_mask = np.arange(start=iter * self.batch_size, stop=(iter+1)*self.batch_size)
                X_batch = self.X_train[batch_mask]
                y_batch = self.y_train[batch_mask]
                # forward & backward
                loss, grads = self.model.loss(X_batch, y_batch)
                # optimizer.step()
                for p, w in self.model.params.items():
                    self.model.params[p], self.optim_configs[p] \
                    = self.optimizer(w, grads[p], self.optim_configs[p])
                # keep track of losses
                self.loss_history.append(loss)
                # output loss per 100 iterations
                if (iter + 1) % self.print_every == 0:
                    print("(Iteration %d / %d) loss: %f"% (iter + 1, iterations, loss))
            # scheduler.step()
            if epoch > 10:
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay
            # valid after every epoch
            train_acc = self.check_accuracy(self.X_train, self.y_train)
            val_acc = self.check_accuracy(self.X_val, self.y_val)
            print("(Epoch %d / %d) train acc: %f; val_acc: %f" % (epoch + 1, self.num_epochs, train_acc, val_acc))
            # keep track of the accuracies
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            # Keep track of the best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params = {}
                for k, v in self.model.params.items():
                    self.best_params[k] = v.copy()
            # save params in checkpoint
            if (epoch + 1) % 10 == 0 or epoch + 1 == self.num_epochs:
                filename = "%d_epoch_%d.pkl" % (self.exp_num, epoch + 1)
                print('Saving checkpoint to "%s"' % filename)
                with open(filename, "wb") as f:
                    pickle.dump(self.model, f)
        # At the end of training swap the best params into the model
        self.model.params = self.best_params