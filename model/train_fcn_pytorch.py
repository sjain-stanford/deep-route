import argparse
import os
import time
import sys

import h5py
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 10.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('../')
from datagen.decoder import decodeData

parser = argparse.ArgumentParser(description='deep-route: FCN Model Training')
parser.add_argument('--data', metavar='PATH', default=os.getcwd()+'/data/', help='path to dataset (default: ./data/)')
parser.add_argument('--batch_size', metavar='N', default=100, type=int, help='mini-batch size (default: 100)')
parser.add_argument('--num_workers', metavar='N', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--num_epochs', metavar='N', default=200, type=int, help='number of total epochs to run (default: 200)')
parser.add_argument('--use_gpu', action='store_true', help='use GPU if available')
parser.add_argument('--pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--lr', metavar='LR', default=5e-4, type=float, help='initial learning rate (default: 5e-4)')
parser.add_argument('--adapt_lr', action='store_true', help='use learning rate schedule')
parser.add_argument('--reg', metavar='REG', default=1e-5, type=float, help='regularization strength (default: 1e-5)')
parser.add_argument('--print-freq', metavar='N', default=10, type=int, help='print frequency (default: 10)')


def main(args):
  # Unutilized GPU notification
  if torch.cuda.is_available() and not args.use_gpu:
    print("GPU is available. Provide command line flag --use_gpu to use it!")
  
  # To run on GPU, specify command-line flag --use_gpu
  if args.use_gpu and torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
  else:
    dtype = torch.FloatTensor
  
  # Dataset filenames
  #train_fname = 'train_4_32pix.hdf5'
  #val_fname = 'val_4_32pix.hdf5'
  #train_fname = 'train_1k_32pix.hdf5'
  #val_fname = 'val_200_32pix.hdf5'
  train_fname = 'train_50k_32pix.hdf5'
  val_fname = 'val_10k_32pix.hdf5'
  #test_fname = 'test_10k_32pix.hdf5'
  
  # Save dir
  train_id = 'train50k_val10k_pix32' + '_lr' + str(args.lr) + '_reg' + str(args.reg) + '_batchsize' + str(args.batch_size) + '_epochs' + str(args.num_epochs) + '_gpu' + str(args.use_gpu)
  #train_id = 'temp'
  save_dir = os.getcwd() + '/training/' + train_id + '/'
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  
  # Weighted loss to overcome unbalanced dataset (>98% pixels are off ('0'))
  weight = torch.Tensor([1, 3]).type(dtype)
  
  # Read dataset at path provided by --data command-line flag
  train_data = h5py.File(args.data + train_fname, 'r')
  X_train = np.asarray(train_data['X'])    # Data: X_train.shape = (N, 1, H, W); X_train.dtype = uint8
  Y_train = np.asarray(train_data['Y'])    # Labels: Y_train.shape = (N, 8, H, W); Y_train.dtype = uint8
  print("X_train: %s \nY_train: %s\n" %(X_train.shape, Y_train.shape))

  val_data = h5py.File(args.data + val_fname, 'r')
  X_val = np.asarray(val_data['X'])
  Y_val = np.asarray(val_data['Y'])
  print("X_val: %s \nY_val: %s\n" %(X_val.shape, Y_val.shape))

  #test_data = h5py.File(args.data + test_fname, 'r')
  #X_test = np.asarray(test_data['X'])
  #Y_test = np.asarray(test_data['Y'])
  #print("X_test: %s \nY_test: %s\n" %(X_test.shape, Y_test.shape))

  # Dimensions
  N_train = X_train.shape[0]
  N_val = X_val.shape[0]
  C = Y_train.shape[1]
  H = X_train.shape[2]
  W = X_train.shape[3]
  dims_X = [-1, 1, H, W]
  dims_Y = [-1, C, H, W]
  
  # Setup DataLoader
  # https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets  
  # PyTorch tensors are of type torch.ByteTensor (8 bit unsigned int)
  # Stored as 2D --> train: (N, 1*H*W), val: (N, 8*H*W)
  train_dset = TensorDataset(torch.from_numpy(X_train).view(N_train, -1),
                             torch.from_numpy(Y_train).view(N_train, -1))
  
  train_loader = DataLoader(train_dset, batch_size=args.batch_size,
                            # Disable shuffling in debug mode
                            #num_workers=args.num_workers, shuffle=False)
                            num_workers=args.num_workers, shuffle=True)
  
  val_dset = TensorDataset(torch.from_numpy(X_val).view(N_val, -1),
                             torch.from_numpy(Y_val).view(N_val, -1))
  
  val_loader = DataLoader(val_dset, batch_size=args.batch_size,
                          num_workers=args.num_workers, shuffle=False)
  
  # Define NN architecture
  model = nn.Sequential(      # Input (N, 1, 32, 32)    
    nn.Conv2d(1, 16, kernel_size=33, stride=1, padding=16, bias=True),    # Output (N, 16, 32, 32)
    nn.BatchNorm2d(16),
    nn.LeakyReLU(inplace=True),
                    
    #nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
    #nn.BatchNorm2d(16),
    #nn.LeakyReLU(inplace=True),

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
    nn.BatchNorm2d(16),
    nn.LeakyReLU(inplace=True),
    
    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
    nn.BatchNorm2d(16),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
    nn.BatchNorm2d(16),
    nn.LeakyReLU(inplace=True),
    
    # Layer 5
    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
    nn.BatchNorm2d(16),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
    nn.BatchNorm2d(16),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
    nn.BatchNorm2d(16),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
    nn.BatchNorm2d(16),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
    nn.BatchNorm2d(16),
    nn.LeakyReLU(inplace=True),

    # Layer 10
    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
    nn.BatchNorm2d(16),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
    nn.BatchNorm2d(16),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
    nn.BatchNorm2d(16),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
    nn.BatchNorm2d(16),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
    nn.BatchNorm2d(16),
    nn.LeakyReLU(inplace=True),

    # Layer 15
    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),    # Output (N, 16, 32, 32)
  )

  # Load pretrained model parameters
  if args.pretrained:
    model.load_state_dict(torch.load(save_dir + '../saved_model_params'))
    
  # Cast model to the correct datatype
  model.type(dtype)  
  
  loss_fn = nn.CrossEntropyLoss(weight=weight).type(dtype)
  
  # Use Adam optimizer with default betas
  optimizer = optim.Adam(model.parameters(), lr=args.lr,
                         betas=(0.9, 0.999), weight_decay=args.reg)
  loss_history = []
  train_precision = [0]
  train_recall = [0]
  train_f1score = [0]
  val_precision = [0]
  val_recall = [0]
  val_f1score = [0]
  
  best_val_f1score = 0
  
  epoch_time = AverageMeter()
  end = time.time()
  
  # Run the model for given epochs
  for epoch in range(args.num_epochs):
    # Adaptive learning rate schedule
    if args.adapt_lr:
      adjust_learning_rate(optimizer, epoch)
    
    # Run an epoch over the training data
    loss = train(model, train_loader, loss_fn, optimizer, dtype, dims_X, dims_Y, epoch)
    loss_history.extend(loss)
    
    # Check precision/recall/accuracy/F1_score on the train and val sets
    prec, rec, f1 = check_accuracy(model, train_loader, dtype, dims_X, dims_Y, epoch, save_dir, 'train')
    train_precision.append(prec)
    train_recall.append(rec)
    train_f1score.append(f1)    
    prec, rec, f1 = check_accuracy(model, val_loader, dtype, dims_X, dims_Y, epoch, save_dir, 'val')    
    val_precision.append(prec)
    val_recall.append(rec)
    val_f1score.append(f1)

    plt.subplot(2, 2, 1)
    plt.title('Training loss')
    plt.plot(loss_history, 'o')
    plt.yscale('log')
    plt.xlabel('Iteration')

    plt.subplot(2, 2, 2)
    plt.title('Accuracy (F1 Score)')
    plt.plot(train_f1score, '-o', label='train')
    plt.plot(val_f1score, '-o', label='val')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
    plt.subplot(2, 2, 3)
    plt.title('Precision')
    plt.plot(train_precision, '-o', label='train')
    plt.plot(val_precision, '-o', label='val')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    plt.subplot(2, 2, 4)
    plt.title('Recall')
    plt.plot(train_recall, '-o', label='train')
    plt.plot(val_recall, '-o', label='val')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)    
    plt.savefig(save_dir + 'training_history.jpg')
    #plt.savefig(save_dir + 'training_history.eps', format='eps')
    plt.close()
    
    # Save best model parameters
    if f1 > best_val_f1score:
      best_val_f1score = f1
      print('Saving best model parameters with Val F1 score = %.4f' %(best_val_f1score))
      torch.save(model.state_dict(), save_dir + 'saved_model_params')
      

    # Measure elapsed time
    epoch_time.update(time.time() - end)
    end = time.time()
    
    print('Timer Epoch [{0}/{1}]\t'
          't_epoch {epoch_time.val:.3f} ({epoch_time.avg:.3f})'.format(
             epoch+1, args.num_epochs, epoch_time=epoch_time))


def train(model, loader, loss_fn, optimizer, dtype, dims_X, dims_Y, epoch):
  """
  Train the model for one epoch
  """
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  
  # Set the model to training mode
  model.train()

  loss_hist = []
  
  end = time.time()
  for i, (x, y) in enumerate(loader):
    # The DataLoader produces 2D Torch Tensors, so we need to reshape them to 4D,
    # cast them to the correct datatype and wrap them in Variables.
    #
    # Note that the labels should be a torch.LongTensor on CPU and a
    # torch.cuda.LongTensor on GPU; to accomplish this we first cast to dtype
    # (either torch.FloatTensor or torch.cuda.FloatTensor) and then cast to
    # long; this ensures that y has the correct type in both cases.
    
    # Measure data loading time
    data_time.update(time.time() - end)
    
    x = x.view(dims_X)    # (N_batch, 1, H, W)
    y = y.view(dims_Y)    # (N_batch, 8, H, W)
    x_var = Variable(x.type(dtype), requires_grad=False)
    y_var = Variable(y.type(dtype).long(), requires_grad=False)

    # Run the model forward to compute scores and loss
    scores = model(x_var)   # (N_batch, 16, H, W)
    
    # To convert scores from (N_batch, 16, H, W) to (N_batch*H*W*8, 2) where 2 = number of classes (on/off),
    # for PyTorch's cross entropy loss format (http://pytorch.org/docs/nn.html#crossentropyloss)
    _, twoC, _, _ = scores.size()
    scores = scores.permute(0, 2, 3, 1).contiguous().view(-1, twoC)    # (N_batch*H*W, twoC)
    scores = torch.cat((scores[:, 0:twoC:2].contiguous().view(-1, 1), 
                        scores[:, 1:twoC:2].contiguous().view(-1, 1)), 1)    # (N_batch*H*W*8, 2)
    
    # To convert y_var from (N_batch, 8, H, W) to (N_batch*H*W*8)
    # for PyTorch's cross entropy loss format (http://pytorch.org/docs/nn.html#crossentropyloss)
    y_var = y_var.permute(0, 2, 3, 1).contiguous().view(-1)    # (N_batch*H*W*8)
    
    # Use cross entropy loss - 16 filter case
    loss = loss_fn(scores, y_var)
    
    losses.update(loss.data[0], y_var.size(0))    
    loss_hist.append(loss.data[0])
    
    # Run the model backward and take a step using the optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if (i % args.print_freq == 0) or (i+1 == len(loader)):
      print('Train Epoch [{0}/{1}]\t'
            'Batch [{2}/{3}]\t'
            't_total {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            't_data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
             epoch+1, args.num_epochs, i+1, len(loader), batch_time=batch_time,
             data_time=data_time, loss=losses))
  
  return loss_hist


def check_accuracy(model, loader, dtype, dims_X, dims_Y, epoch, save_dir, train_val):
  """
  Check the accuracy of the model
  """
  # Filenames
  y_pred_fname = 'Y_' + train_val + '_pred'
  y_act_fname = 'Y_' + train_val + '_act'
    
  # Set the model to eval mode
  model.eval()

  tp, tn, fp, fn = 0, 0, 0, 0
  
  for i, (x, y) in enumerate(loader):
    # Reshape 2D torch tensors from DataLoader to 4D, cast to the 
    # correct type and wrap it in a Variable.
    #
    # At test-time when we do not need to compute gradients, marking 
    # the Variable as volatile can reduce memory usage and slightly improve speed.
    x = x.view(dims_X)    # (N_batch, 1, H, W)
    y = y.view(dims_Y)    # (N_batch, 8, H, W)
    x_var = Variable(x.type(dtype), volatile=True)
    y_var = Variable(y.type(dtype), volatile=True)

    # Run the model forward, and compute the y_pred to compare with the ground-truth
    scores = model(x_var)    # (N, 16, H, W)
    
    _, twoC, _, _ = scores.size()    
    scores_off = scores[:, 0:twoC:2, :, :]
    scores_on = scores[:, 1:twoC:2, :, :]
    
    y_pred = (scores_on > scores_off)    # (N_batch, 8, H, W)
    
    # Precision / Recall / F-1 Score
    #https://en.wikipedia.org/wiki/Precision_and_recall
    # tp = true_pos, tn = true_neg, fp = false_pos, fn = false_neg
    tp += ((y_pred.data == 1) * (y_var.data == 1)).sum()
    tn += ((y_pred.data == 0) * (y_var.data == 0)).sum()
    fp += ((y_pred.data == 1) * (y_var.data == 0)).sum()
    fn += ((y_pred.data == 0) * (y_var.data == 1)).sum()
    
    # Preview images from first mini-batch after every 5% of epochs
    # E.g., if num_epochs = 20, preview every 1 epoch
    # if num_epochs = 200, preview every 10 epochs
    if i == 0 and ((epoch % (args.num_epochs*5//100) == 0) or (epoch+1 == args.num_epochs)):
      Y_act_dec = decodeData(y_var.data.cpu().numpy())  # (N_batch, 3, H, W)
      Y_act_dec = np.swapaxes(Y_act_dec, 1, 2)    # (N_batch, H, 3, W)
      Y_act_dec = np.swapaxes(Y_act_dec, 2, 3)    # (N_batch, H, W, 3)

      Y_pred_dec = decodeData(y_pred.data.cpu().numpy())  # (N_batch, 3, H, W)
      Y_pred_dec = np.swapaxes(Y_pred_dec, 1, 2)    # (N_batch, H, 3, W)
      Y_pred_dec = np.swapaxes(Y_pred_dec, 2, 3)    # (N_batch, H, W, 3)

      num_images = 9
      for n in range(num_images):
        plt.subplot(3, 3, n+1)
        plt.imshow(Y_act_dec[n].astype('uint8'))
        #plt.axis('off')
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        plt.title('Y_%s_actual (epoch %d)' % (train_val, epoch+1))
      plt.savefig(save_dir + 'epoch_' + str(epoch+1) + '_' + y_act_fname + '.jpg')
      #plt.savefig(save_dir + 'epoch_' + str(epoch+1) + '_' + y_act_fname + '.eps', format='eps')
      plt.close()

      for n in range(num_images):
        plt.subplot(3, 3, n+1)
        plt.imshow(Y_pred_dec[n].astype('uint8'))
        #plt.axis('off')
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        plt.title('Y_%s_predicted (epoch %d)' % (train_val, epoch+1))
      plt.savefig(save_dir + 'epoch_' + str(epoch+1) + '_' + y_pred_fname + '.jpg')
      #plt.savefig(save_dir + 'epoch_' + str(epoch+1) + '_' + y_pred_fname + '.eps', format='eps')
      plt.close()

  # 1e-8 to avoid division by zero        
  precision = tp / (tp + fp + 1e-8)
  recall = tp / (tp + fn)
  accuracy = (tp + tn) / (tp + tn + fp + fn)
  f1_score = 2 * (precision*recall) / (precision + recall + 1e-8)

  print('{0}\t'
        'Check Epoch [{1}/{2}]\t'
        'Precision {p:.4f}\t'
        'Recall {r:.4f}\t'
        'Accuracy {a:.4f}\t'
        'F1 score {f1:.4f}'.format(
         train_val, epoch+1, args.num_epochs, p=precision, r=recall, a=accuracy, f1=f1_score))
  
  return precision, recall, f1_score


def bce_loss(input, target):
  """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Variable of shape (N, 8, H, W) giving scores.
    - target: PyTorch Variable of shape (N, 8, H, W) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Variable containing the mean BCE loss over the minibatch of input data.
    """
  # bce_loss(input, target) = target * -log(sigmoid(input)) + (1 - target) * -log(1 - sigmoid(input))
  
  neg_abs = - input.abs()
  bce_loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()    # (N, 8, H, W)
  return bce_loss.mean()


def wt_bce_loss(input, target, weight):
  """
    Numerically stable version of the weighted binary cross-entropy loss function.

    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Variable of shape (N, 8, H, W) giving scores.
    - target: PyTorch Variable of shape (N, 8, H, W) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Variable containing the mean weighted BCE loss over the minibatch of input data.
    """
  # wt_bce_loss(input, target, weight) = weight * target * -log(sigmoid(input)) + (1 - target) * -log(1 - sigmoid(input))
  
  neg_abs = - input.abs()
  wt_bce_loss = (-input).clamp(min=0) + (1 - target) * input + (1 + (weight - 1) * target) * (1 + neg_abs.exp()).log()    # (N, 8, H, W)
  return wt_bce_loss.mean()


def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""  
  lr = args.lr * (0.1 ** (epoch // 30))
  print("Adaptive learning rate: %e" %(lr))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

        
if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
