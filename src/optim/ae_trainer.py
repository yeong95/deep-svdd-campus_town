from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


def check_autoencoder_quality(processed_image, raw_image, output_image):
  min_ = -3.9348
  max_ = 0.7229
  mean_ = np.mean(raw_image)
  tmp_raw_image = raw_image - mean_
  x_scale = np.mean(np.absolute(tmp_raw_image))
  unnormalized_image = processed_image*(max_-min_) + min_
  unnormalized_image = unnormalized_image*x_scale + mean_
  unnormalized_image = unnormalized_image.cpu().numpy().reshape(640,640).astype(np.uint8)
  unnormalized_output = output_image*(max_-min_) + min_
  unnormalized_output = unnormalized_output*x_scale + mean_
  unnormalized_output = unnormalized_output.cpu().numpy().reshape(640,640).astype(np.uint8)
  from IPython.core.debugger import set_trace
  set_trace()
  image = Image.fromarray(raw_image)
  image.save("raw_image.png")
  image2 = Image.fromarray(unnormalized_image, mode='L')
  image2.save("reconstructed_raw_image.png")
  image3 = Image.fromarray(unnormalized_output, mode='L')
  image3.save("reconstructed_output_image.png")

class AETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

    def train(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        pretrain_time = time.time() - start_time
        logger.info('Pretraining time: %.3f' % pretrain_time)
        logger.info('Finished pretraining.')

        return ae_net
    

    def test(self, dataset: BaseADDataset, ae_net: BaseNet, test_image):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = ae_net(inputs)
                # import pdb;pdb.set_trace()
                if labels==0:
                  check_autoencoder_quality(inputs, test_image[i], outputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1

        logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * auc))

        test_time = time.time() - start_time
        logger.info('Autoencoder testing time: %.3f' % test_time)
        logger.info('Finished testing autoencoder.')




