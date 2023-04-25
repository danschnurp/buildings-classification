#  @author Daniel Schnurpfeil
#  @version 1.1
#  @date 25.4. 2023

import copy
import logging
import time
from os import path

from dateutil.utils import today
from torch import nn, optim, load, profiler, cuda, version, device, set_grad_enabled, max, sum, save, backends
from torch.backends.cudnn import deterministic
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision import models


def train_model(settings, model, criterion, optimizer, scheduler, prof, net_type, num_epochs):
    """
    This function trains a machine learning model using the specified settings, criterion, optimizer, scheduler, and number
    of epochs.

    :param prof:
    :param settings: It is a class containing various settings for the model training such as batch size, learning
    rate, etc
    :param model: The neural network model that you want to train. It should be an instance of a PyTorch nn.Module subclass
    :param criterion: The criterion is the loss function used to evaluate the performance of the model during training. It
    measures the difference between the predicted output and the actual output. The goal of the training process is to
    minimize this loss function. Common loss functions include mean squared error, cross-entropy loss, and binary cross-
    :param optimizer: The optimizer is an algorithm used to update the weights and biases of the neural network during
    training. It is responsible for minimizing the loss function by adjusting the parameters of the model. There are various
    types of optimizers available such as Stochastic Gradient Descent (SGD), Adam, Adagrad,
    :param scheduler: The scheduler is an object that adjusts the learning rate during training. It can be used to gradually
    decrease the learning rate as the training progresses, which can help the model converge to a better solution. The
    scheduler can be set up to adjust the learning rate based on various criteria, such as the number of
    :param net_type: The type of neural network architecture being used for the model, such as "CNN" for convolutional
    neural network or "RNN" for recurrent neural network
    :param num_epochs: The number of times the entire dataset will be passed through the model during training. Each pass
    through the dataset is called an epoch
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    logger = logging.getLogger("results")
    logger.setLevel(logging.INFO)

    handler_local = logging.FileHandler("./cnn_models/auto_results.txt")
    # handler_disk = logging.FileHandler("C:\\Users\\dschn\\SynologyDrive\\results\\auto_results"
    #                                    + net_type + str(today().today().date()) + ".txt")

    formatter = logging.Formatter('[%(levelname)s] - %(asctime)s - %(message)s')
    handler_local.setFormatter(formatter)
    # handler_disk.setFormatter(formatter)
    #
    # logger.addHandler(handler_disk)
    logger.addHandler(handler_local)

    logger.info('-' * 10)
    logger.info('Beginning training with: ' + net_type)

    for epoch in range(num_epochs):
        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)
        # prof.start()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in settings.dataloaders[phase]:
                inputs = inputs.to(settings.device)
                labels = labels.to(settings.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # adds to tensorboard
                settings.iteration += 1
                settings.writer.add_scalar("Loss/all", loss, settings.iteration)
                settings.writer.flush()
                # prof.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / settings.dataset_sizes[phase]
            epoch_acc = float(running_corrects) / float(settings.dataset_sizes[phase])
            if phase == 'train':
                settings.writer.add_scalar("Loss/train", epoch_loss, epoch)
                settings.writer.add_scalar("Acc/train", epoch_acc, epoch)
                settings.writer.flush()
            if phase == 'val':
                settings.writer.add_scalar("Loss/val", epoch_loss, epoch)
                settings.writer.add_scalar("Acc/val", epoch_acc, epoch)
                settings.writer.flush()
                # shutil.copytree("runs", "C:\\Users\\dschn\\SynologyDrive\\results\\runs", dirs_exist_ok=True)
            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
                save(model, "./cnn_models/model_backup_" +
                     str(today().today().date()) + ".pt")
        logger.info(" ")
        # prof.stop()

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_acc))
    logger.info(" ")
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


class CnnPict:

    def __init__(self, batchSize, net_type, device_to_set):
        # plt.ion()  # interactive mode

        data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                # transforms.Resize(500),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        data_dir = "../ready_data/"

        image_datasets = {x: datasets.ImageFolder(path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(image_datasets[x],
                                          batch_size=batchSize,  # IterableDataset= class_length / batch_size
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=8)  # (torch 1.10) maximum = (logical cpus / 2)
                            for x in ['train', 'val']}

        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        print("DATA classes: " + str(class_names))

        self.device = device(device_to_set)

        # tensorboard
        self.iteration = 0
        self.writer = SummaryWriter(filename_suffix=net_type)

    ##########################################################################


def main(batchSize=5, epo=25, input_model=None, net_type="mobilenet_v2", device_to_set="cuda"):
    """
    This function takes in several parameters including batch size, number of epochs, input model, and network type and runs
    a machine learning model.

    :param device_to_set:
    :param batchSize: The number of images that will be processed in each iteration of training. A larger batch size can
    lead to faster training times, but may also require more memory and can result in less accurate models, defaults to 5
    (optional)
    :param epo: epo stands for "epochs" and refers to the number of times the entire dataset is passed through the neural
    network during training. Each epoch consists of multiple iterations where the model updates its weights based on the
    error between the predicted output and the actual output. Increasing the number of epochs can improve the accuracy,
    defaults to 25 (optional)
    :param input_model: The input_model parameter is used to specify a pre-trained model that will be used as a starting
    point for training a new model. If this parameter is not specified, a new model will be created from scratch
    :param net_type: The type of neural network architecture to be used for the model. In this case, it is set to
    "mobilenet_v2", defaults to mobilenet_v2 (optional)
    """

    print("running with batch_size: " + str(batchSize))

    ###########
    # training
    ###########
    cnn = CnnPict(batchSize, net_type, device_to_set)
    # model
    if path.isfile(input_model):
        model_ft = load(input_model,  map_location=device(device_to_set))
    else:
        model_ft = models.mobilenet_v2(pretrained=True)

    # Remove last fully connected layer
    model_ft.classifier = nn.Sequential(*list(model_ft.classifier.children())[:-1])

    # Add new fully connected layer with 10 output neurons
    model_ft.classifier.add_module('fc', nn.Linear(1280, 10))

    model_ft = model_ft.to(device_to_set)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(cnn, model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           None, net_type,
                           num_epochs=epo)
    # saving the model
    save(model_ft, "./cnn_models/model_" + net_type + str(today().today().date()) + ".pt")
    save(model_ft, "./cnn_models/model_" + net_type + "_latest.pt")


def control_torch():
    """
    test if cuda is available
    """
    print("cuda availability: " + str(cuda.is_available()))

    import gc
    gc.collect()

    if not cuda.is_available():
        return "cpu"

    print("version: " + version.cuda)
    cuda.empty_cache()
    # Storing ID of current CUDA device
    cuda_id = cuda.current_device()
    print(f"ID of current CUDA device:{cuda.current_device()}")
    print(f"Name of current CUDA device:{cuda.get_device_name(cuda_id)}")

    # VERY IMPORTANT
    ######################################
    backends.cudnn.enabled = True  #
    backends.cudnn.benchmark = True  #
    backends.cudnn.deterministic = True  #
    #######################################
    return "cuda"


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='SImple NEUral trainer using pytorch. :-) ne asi')
    parser.add_argument('-e', '--number_of_epochs', default=3, type=int)
    parser.add_argument('-b', '--batch_size', default=6, type=int)
    parser.add_argument('-i', '--pretrained_input',
                        default="./cnn_models/dinosaur_ntb/mobilenet_v2/model_mobilenet_v22022-03-18.pt")
    parser.add_argument('-t', '--net_type', default="mobilenet_v2",)

    args = parser.parse_args()
    main(
            epo=args.number_of_epochs,
            batchSize=args.batch_size,
            device_to_set=control_torch(),
            input_model=args.pretrained_input,
            net_type=args.net_type
        )
