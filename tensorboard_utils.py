## Source: https://github.com/chasingbob/keras-visuals/blob/master/visual_callbacks.py

import tensorflow.keras as keras
import tensorflow as tf
from keras.callbacks import Callback

import matplotlib.pyplot as plt
import numpy as np

class ConfusionMatrixPlotter(Callback):
    """Plot the confusion matrix on a graph and update after each epoch
    # Arguments
        X_val: The input values 
        Y_val: The expected output values
        classes: The categories as a list of string names
        normalize: True - normalize to [0,1], False - keep as is
        cmap: Specify matplotlib colour map
        title: Graph Title
        
    """
    def __init__(self, X_val, Y_val, classes, normalize=False, cmap=plt.cm.Blues, title='Confusion Matrix'):
        self.X_val = X_val
        self.Y_val = Y_val
        self.title = title
        self.classes = classes
        self.normalize = normalize
        self.cmap = cmap
        plt.ion()
        #plt.show()
        plt.figure()

        plt.title(self.title)
        
        

    def on_train_begin(self, logs={}):
        pass

    
    def on_epoch_end(self, epoch, logs={}):    
        plt.clf()
        pred = self.model.predict(self.X_val)
        max_pred = np.argmax(pred, axis=1)
        max_y = np.argmax(self.Y_val, axis=1)
        cnf_mat = confusion_matrix(max_y, max_pred)
   
        if self.normalize:
            cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]

        thresh = cnf_mat.max() / 2.
        for i, j in itertools.product(range(cnf_mat.shape[0]), range(cnf_mat.shape[1])):
            plt.text(j, i, cnf_mat[i, j],                                          
                         horizontalalignment="center",
                         color="white" if cnf_mat[i, j] > thresh else "black")

        plt.imshow(cnf_mat, interpolation='nearest', cmap=self.cmap)

        # Labels
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        plt.colorbar()
                                                                                                         
        plt.tight_layout()                                                    
        plt.ylabel('True label')                                              
        plt.xlabel('Predicted label')                                         
        #plt.draw()
        plt.show()
        plt.pause(0.001)
        
        
# image grid callback
from keras.utils import OrderedEnqueuer, GeneratorEnqueuer
import random
class TensorBoardImage(keras.callbacks.Callback):
    def __init__(
        self,
        data_generator: keras.utils.Sequence,
        batch_size: int,
        num_samples: int,
        classes: list,
        output_dir: str,
        ):
        super().__init__() 
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.classes = classes
        self.enqueuer = GeneratorEnqueuer(
            data_generator, use_multiprocessing=True
        )
        self.enqueuer.start(workers=1, max_queue_size=1)
        self.writer = tf.summary.create_file_writer(output_dir)
        

    def on_epoch_end(self, epoch, logs=None):
        """ After the end of each epoch, a sample of num_samples predicted images
        is displayed. The samples are selected by picking the first one from
        each batch until the total number of needed samples is reached.
        
        """

        output_generator = self.enqueuer.get()

        with self.writer.as_default():

            steps_done = 0
            while steps_done < self.num_samples:
                generator_output = next(output_generator)
                x_batch, y_batch = generator_output
                y_pred = self.model.predict(x_batch)
                true_label = self.classes[np.argmax(y_batch[0])]
                predicted_label = self.classes[np.argmax(y_pred[0])]
                
                tf.summary.image(
                    "Epoch-{}/{}/True:{} Predicted:{}".format(epoch, 
                                                              steps_done,
                                                              true_label,
                                                              predicted_label
                                                             ),
                    x_batch[[0], :, :, :],
                    step=epoch,
                    
                )
                steps_done += 1  
    
    def on_train_end(self, logs=None):
        """Defines what is run at the end of training.
        - Stops the data_generator
        - Closes TensorBoard writer
        """
        self.enqueuer.stop()
        self.writer.close()