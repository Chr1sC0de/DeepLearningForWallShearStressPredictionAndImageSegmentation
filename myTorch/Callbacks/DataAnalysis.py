from . import Callback
import matplotlib.pyplot as plt
import numpy as np


class DataAnalysis(Callback):

    def __init__(
            self, in_memory=False, save_folder = r'./Analysis', n_step=1, n_average = 5,
            **metrics):

            self.n_step = n_step
            self.n_average = n_average
            self.global_step = 0
            self.loss_log = []
            self.mean_loss_accuracy = []

    def post_batch(self,env):
    
        y_pred = env.y_pred.to('cpu').detach().numpy()
        y_true = env.y_true.to('cpu').detach().numpy()
        accuracy = 100 - np.abs(((y_true - y_pred)/y_true * 100).mean())
        
        self.loss_log.append(accuracy)

        if self.global_step % self.n_step == 0:
            plt.figure()
            plt.xlabel('Epochs')
            plt.ylabel('Training Loss %')
            if len(self.loss_log) > self.n_average:
                self.mean_loss_accuracy.append(
                    sum(self.loss_log[-self.n_average:])/self.n_average
                )
            self.global_step += 1
                
        





            
        
        

            


        # initialize when to save state
