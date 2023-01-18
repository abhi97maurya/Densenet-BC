### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, time
import numpy as np
from Network import DenseNet
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# from NetWork import ResNet
from ImageUtils import parse_record
from Network import DenseNet
from ImageUtils import parse_record

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs, training_configs):
        super(MyModel, self).__init__()

        self.config = configs
        self.training_configs = training_configs
        # self.network = MyNetwork(configs)
        self.network =  DenseNet(growthRate=self.config["growthrate"], depth=self.config["layers"], compression_factor=self.config["compression_factor"],
                            bottleneck=self.config["bottleneck"], nClasses=self.config["nclasses"])
        # convert network to cuda
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network = self.network.to(device).cuda()       
        # device = torch.device("cpu")
        # self.network = self.network.to(device)                 
        # YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.learning_rate = self.config["learning_rate"]
        self.criteria = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001, weight_decay = self.config["weight_decay)
        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay = self.config["weight_decay"])
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate, weight_decay = self.config["weight_decay"], momentum=self.config["momentum"])
        ### YOUR CODE HERE

    def train(self, x_train, y_train, training_configs, x_valid=None, y_valid=None):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config["batch_size"]

        print('### Training... ###')
        print('### Model Config... ###', self.config)
        print('### Training Configs... ###', self.training_configs)
        for epoch in range(1, training_configs["max_epoch"]+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            # YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            # self.learning_rate = 0.1 if epoch > 2 else 0.01
            # if epoch * num_batches > 48000:
            #     self.learning_rate *= 0.01
            # elif epoch * num_batches > 32000:
            #     self.learning_rate *= 0.1
            # if (epoch % 50) == 0:
            #     self.learning_rate *= 0.1
            if (epoch == 0.5 * training_configs["max_epoch"] or epoch == 0.75 * training_configs["max_epoch"]):
                self.learning_rate *= 0.1
                
            batchsize = self.config["batch_size"]
            num_batches = int(num_samples/batchsize)
            accuracy = 0
            # YOUR CODE HERE
            
            for i in range(num_batches):
                # YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                
                # YOUR CODE HERE
                start_ix = i * batchsize
                end_ix = start_ix + batchsize
                if end_ix > curr_x_train.shape[0]:
                    continue
                x_batch = [parse_record(curr_x_train[ix],True) for ix in range(start_ix,end_ix)]
                y_batch = curr_y_train[start_ix:end_ix]

                # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                x_batch = torch.from_numpy(np.array(x_batch)).float().to(device).cuda()
                # device = torch.device("cpu")
                # x_batch = torch.from_numpy(np.array(x_batch)).float().to(device)
                y_batch = curr_y_train[start_ix:end_ix]
                #print("Load batch for training: ", batch_x.shape, batch_y.shape)

                pred = self.network(x_batch)
                #print("Predicted vs Actual",pred.shape, batch_y.shape, type(pred), type(batch_y))
                #print(pred, batch_y)
                y_batch = torch.from_numpy(y_batch).long().to(device).cuda()
                # y_batch = torch.from_numpy(y_batch).long().to(device)
                loss = self.criteria(pred,y_batch)
                _, pred = torch.max(pred,1)
                #print(pred, batch_y)

                accuracy += torch.sum(pred==y_batch.data)
                # print(accuracy, "ACC", accuracy.float)
                # print("accuracy", accuracy.data.cpu().numpy())
                #print(pred,batch_y.data)
                accuracy_percentage = (accuracy.data.cpu().numpy() / num_samples)
                accuracy_percentage *= 100
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #print("LOSSS:", loss)

                print('Batch {:d}/{:d} Loss {:.6f} accuracy percentage {:.6f}'.format(i, num_batches, loss, accuracy_percentage), end='\r', flush=True)
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} accuracy percentage {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, accuracy_percentage, duration))

            # if epoch % self.config["save_interval"] == 0:
            if (epoch % self.config["save_interval"] == 0) or (epoch in range(138, 162)):
                self.save(epoch)

    def evaluate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config["modeldir"], 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)
            # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # device = torch.device("cpu")
            preds = []
            for i in tqdm(range(x.shape[0])):
                # YOUR CODE HERE
                x1 = parse_record(x[i],training = False)[None, ...]
                x1 = torch.from_numpy(np.array(x1)).float().to(device).cuda()
                # x1 = torch.from_numpy(np.array(x1)).float().to(device)
                pred = self.network(x1) 
                
                #preds.append(pred)
                # _, pred = torch.max(pred,1)
                _, pred = torch.max(pred,0)
                #print(pred.item(),y[i]) 
                pred = pred.item()
                preds.append(pred)
                # END CODE HERE

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print('Test accuracy for checkpoint_num : {:d} : {:.4f}'.format(checkpoint_num, torch.sum(preds==y)/y.shape[0]))
    
    def predict_prob(self, x, checkpoint_num_list):
        self.network.eval()
        print('### Prediction of private dataset ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config["modeldir"], 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)
            # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # device = torch.device("cpu")
            preds = []
            for i in tqdm(range(x.shape[0])):
                # YOUR CODE HERE
                x1 = parse_record(x[i],training = False)[None, ...]
                x1 = torch.from_numpy(np.array(x1)).float().to(device).cuda()
                # x1 = torch.from_numpy(np.array(x1)).float().to(device)
                pred = self.network(x1) 
                
                pred_values = pred.detach().cpu().numpy()
                pred_values = np.exp(pred_values)
                preds.append(pred_values)
                # END CODE HERE
            preds = np.array(preds)
        return preds
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config["modeldir"], 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config["modeldir"], exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

    # def model_setup(self):
    #     pass

    # def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
    #     pass

    # def evaluate(self, x, y):
    #     pass

    # def predict_prob(self, x):
    #     pass

### END CODE HERE