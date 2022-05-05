from utils import *

use_cuda = torch.cuda.is_available()

class Model():
    '''
    This class initializes a neural network model as a torch object and contains functions to train and test the same. Input parameters for the constructor are defined as follows:
    model_type: model architecture (resnet, vgg16, wide-resnet)
    optimizer_type: type of optimizer to be used (padam, sgd, adam, adamw, nadam and adadelta)
    dataset: use cifar10 or cifar100 dataset
    save_folder: directory to save the models
    train_val_split: % split for train and validation
    batch_size: batch size for training
    epochs: no. of epochs for training
    learning_rate: learning rate for optimizer
    momentum: momentum for optimizer
    p: p hyperparameter for padam
    weight_decay: weight decay scalar
    beta1: beta1 value
    beta2: beta2 value
    lr_scheduler_milestones: at what epochs the lr should be multiplied by lr_scheduler_gamma
    lr_scheduler_gamma: lr multipler
    '''

    def __init__(self, model_type,
                 optimizer_type,
                 dataset,
                 save_folder,
                 train_val_split = 0.9,
                 batch_size = 32,
                 epochs = 20,
                 learning_rate = 0.1,
                 momentum = 0.9,
                 p = 1/8,
                 weight_decay = 5e-4,
                 beta1 = 0.9,
                 beta2 = 0.999,
                 lr_scheduler_milestones = [100, 150],
                 lr_scheduler_gamma = 0.1):
        
        super(Model, self).__init__()
            
        if dataset not in ['cifar10', 'cifar100']:
            raise ValueError("Wrong dataset. Select one from ['cifar10', 'cifar100']")

        if model_type not in ['resnet', 'vgg', 'wideresnet']:
            raise ValueError("Wrong model type. Select one from ['resnet', 'vgg', 'wideresnet']")

        if optimizer_type not in ['padam', 'sgd', 'adam', 'adamw', 'nadam', 'adadelta']:
            raise ValueError("Wrong optimizer type. Select one from ['padam', 'sgd', 'adam', 'adamw', 'nadam', 'adadelta']")
            
        self.model_type = model_type
        self.optimizer_type = optimizer_type
        self.dataset = dataset
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.p = p
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr_scheduler_milestones = lr_scheduler_milestones
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.save_folder = save_folder
        
        if self.dataset == 'cifar10':
            self.num_classes = 10

        if self.dataset == 'cifar100':
            self.num_classes = 100
    
        print("Preparing the dataset.")
        trainloader, valloader, testloader = get_dataloader(dataset = self.dataset, batch_size = self.batch_size, train_val_split = self.train_val_split) #Prepare the dataset generator objects using helper functions 
        self.dataloaders = {'train': trainloader, 'val' : valloader, 'test': testloader} #Define a dictionary containing dataset generator objects for train, val and test sets

        #Changing the output size (100 in the pre-trained model) of the last layer in resnet to the number of classes in the dataset
        if self.model_type == 'resnet':
            self.net = models.resnet18()
            self.net.fc.out_features = self.num_classes
        
        #Changing the output size (100 in the pre-trained model) of the last layer in vgg to the number of classes in the dataset
        if self.model_type == 'vgg':
            self.net = models.vgg16()
            self.net.classifier[6].out_features = self.num_classes

        #Changing the output size (100 in the pre-trained model) of the last layer in wide-resnet to the number of classes in the dataset 
        if self.model_type == 'wideresnet':
            self.net = models.wide_resnet50_2()
            self.net.fc.out_features = self.num_classes
            
        #Defining the loss computation criteria
        self.criterion = nn.CrossEntropyLoss()
        
        #Defining the optimizer object based on the user's choice
        if self.optimizer_type == 'padam':
            self.optimizer = Padam(self.net.parameters(), 
                              lr = self.learning_rate, 
                              partial = self.p, 
                              weight_decay = self.weight_decay, 
                              betas = (self.beta1, self.beta2))
            
        if self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr = self.learning_rate, momentum = self.momentum)
            
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), betas = (self.beta1, self.beta2), lr = self.learning_rate, weight_decay = self.weight_decay)
            
        if self.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(self.net.parameters(), betas = (self.beta1, self.beta2), lr = self.learning_rate, weight_decay = self.weight_decay)
            
        if self.optimizer_type == 'nadam':
            self.optimizer = optim.NAdam(self.net.parameters(), betas = (self.beta1, self.beta2), lr = self.learning_rate, weight_decay = self.weight_decay)
            
        if self.optimizer_type == 'adadelta':
            self.optimizer = optim.Adadelta(self.net.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)

        #Defining learning rate scheduler
        self.scheduler = MultiStepLR(self.optimizer, milestones = self.lr_scheduler_milestones, gamma = self.lr_scheduler_gamma)

        print('')
        print(self.net)
        total_params = sum([p.data.nelement() for p in self.net.parameters()])

        print('')
        print('Total trainable parameters in the network: %.4f' % (total_params / 1e6) + 'M')

    
    def train(self):
        '''
        This function starts the training of torch model based on the model, optimizer, and dataset defined by the user.
        Input: none
        Output: list of per epoch training loss values, list of per epoch validation error values
        '''

        self.net = self.net.to('cuda' if use_cuda else 'cpu')

        best_val_accuracy = -math.inf

        epoch_loss_list = []
        epoch_val_error_list = []

        for epoch in range(self.epochs):

            self.net.train()

            batches_in_pass = len(self.dataloaders['train'])

            #Training starts here ----
            training_epoch_start_time = time.time()

            loss_total = 0.0
            epoch_loss = 0.0

            #Generate batches for training and enumerate across each
            for idx, data in enumerate(self.dataloaders['train']):
                
                #Load the training batch data into cpu/gpu for computation
                inputs, labels = data
                inputs = inputs.to('cuda' if use_cuda else 'cpu')
                labels = labels.to('cuda' if use_cuda else 'cpu')

                self.optimizer.zero_grad() #Initialize the gradients to 0 for back prop

                outputs = self.net(inputs) #Generate the output using the input
                loss = self.criterion(outputs, labels) #Compute the loss based on the output and ground truth for the batch
                loss.backward() #Propogate the loss backwards
                self.optimizer.step() #Move the optimizer to the next step

                epoch_loss += loss.item()
                loss_total += loss.item()

            self.scheduler.step()

            #Validation starts here ----
            epoch_loss /= batches_in_pass

            epoch_loss_list.append(epoch_loss)

            self.net.eval()

            correct = 0.0
            total = 0.0
            for idx, data in enumerate(self.dataloaders['val']):

                #Load the validation batch data into cpu/gpu for computation
                inputs, labels = data
                inputs = inputs.to('cuda' if use_cuda else 'cpu')
                labels = labels.to('cuda' if use_cuda else 'cpu')

                outputs = self.net(inputs) #Generate the output using input
                _, predicted = torch.max(outputs, 1) #Generate predicted class labels

                total += labels.shape[0]
                correct += (predicted == labels).sum().item() #Compute the number of labels predicted correctly

            epoch_accuracy = 100 * correct / total #Compute current epoch's accuracy

            print(f'Epoch {epoch + 1} | Training Loss (Avg): {epoch_loss:.6f} | Validation Accuracy: {epoch_accuracy}% | Time elapsed: {time.time() - training_epoch_start_time:.2f}s')

            epoch_val_error_list.append(100 - epoch_accuracy)

            #Saving the model
            save_path = os.path.join(os.getcwd(), 'models', self.save_folder)
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if epoch_accuracy > best_val_accuracy:

                #Save the model for current epoch if the accuracy is greater than the accuracy of previous epoch
                torch.save(self.net.state_dict(), os.path.join(save_path, 'best'+str(self.epochs)+'.pt'))
                best_val_accuracy = epoch_accuracy

        print('Training Complete.') 
        
        return epoch_loss_list, epoch_val_error_list
    
    
    def test(self):
        '''
        This function tests the accuracy of the trained model on test set of the data.
        Input: none
        Output: none (prints accuracy value)
        '''
        
        self.net.eval()

        correct = 0.0
        total = 0.0

        with torch.no_grad():

            #Generate batches for testing and enumerate across each
            for idx, data in enumerate(self.dataloaders['test']):

                #Load the validation batch data into cpu/gpu for computation
                inputs, labels = data
                inputs = inputs.to('cuda' if use_cuda else 'cpu')
                labels = labels.to('cuda' if use_cuda else 'cpu')

                outputs = self.net(inputs) #Generate output using input
                _, predicted = torch.max(outputs, 1) #Generate predicted class labels

                total += labels.shape[0]
                correct += (predicted == labels).sum().item() #Compute the number of labels predicted correctly

            print(f'Accuracy of the model on test images: {100 * correct // total}%') #Compute and print the overall accuracy of the model on test set

