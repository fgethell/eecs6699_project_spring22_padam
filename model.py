from utils import *

use_cuda = torch.cuda.is_available()

class Model():
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
        trainloader, valloader, testloader = get_dataloader(dataset = self.dataset, batch_size = self.batch_size, train_val_split = self.train_val_split)
        self.dataloaders = {'train': trainloader, 'val' : valloader, 'test': testloader}

        if self.model_type == 'resnet':
            self.net = models.resnet18()
            self.net.fc.out_features = self.num_classes
        
        if self.model_type == 'vgg':
            self.net = models.vgg16()
            self.net.classifier[6].out_features = self.num_classes
            
        if self.model_type == 'wideresnet':
            self.net = models.wide_resnet50_2()
            self.net.fc.out_features = self.num_classes
            
        self.criterion = nn.CrossEntropyLoss()
        
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

        self.scheduler = MultiStepLR(self.optimizer, milestones = self.lr_scheduler_milestones, gamma = self.lr_scheduler_gamma)

        print('')
        print(self.net)
        total_params = sum([p.data.nelement() for p in self.net.parameters()])

        print('')
        print('Total trainable parameters in the network: %.4f' % (total_params / 1e6) + 'M')

    
    def train(self):
        
        self.net = self.net.to('cuda' if use_cuda else 'cpu')

        best_val_accuracy = -math.inf

        epoch_loss_list = []
        epoch_val_error_list = []

        for epoch in range(self.epochs):

            self.net.train()

            batches_in_pass = len(self.dataloaders['train'])

            #Training
            training_epoch_start_time = time.time()

            loss_total = 0.0
            epoch_loss = 0.0

            for idx, data in enumerate(self.dataloaders['train']):

                inputs, labels = data
                inputs = inputs.to('cuda' if use_cuda else 'cpu')
                labels = labels.to('cuda' if use_cuda else 'cpu')

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                loss_total += loss.item()

            self.scheduler.step()

            #Validation   
            epoch_loss /= batches_in_pass

            epoch_loss_list.append(epoch_loss)

            self.net.eval()

            correct = 0.0
            total = 0.0
            for idx, data in enumerate(self.dataloaders['val']):

                inputs, labels = data
                inputs = inputs.to('cuda' if use_cuda else 'cpu')
                labels = labels.to('cuda' if use_cuda else 'cpu')

                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, 1)

                total += labels.shape[0]
                correct += (predicted == labels).sum().item()

            epoch_accuracy = 100 * correct / total

            print(f'Epoch {epoch + 1} | Training Loss (Avg): {epoch_loss:.6f} | Validation Accuracy: {epoch_accuracy}% | Time elapsed: {time.time() - training_epoch_start_time:.2f}s')

            epoch_val_error_list.append(100 - epoch_accuracy)

            #Saving the model
            save_path = os.path.join(os.getcwd(), 'models', self.save_folder)
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if epoch_accuracy > best_val_accuracy:

                torch.save(self.net.state_dict(), os.path.join(save_path, 'best'+str(self.epochs)+'.pt'))
                best_val_accuracy = epoch_accuracy

        print('Training Complete.') 
        
        return epoch_loss_list, epoch_val_error_list
    
    
    def test(self):
        
        self.net.eval()

        correct = 0.0
        total = 0.0

        with torch.no_grad():
            for idx, data in enumerate(self.dataloaders['test']):

                inputs, labels = data
                inputs = inputs.to('cuda' if use_cuda else 'cpu')
                labels = labels.to('cuda' if use_cuda else 'cpu')

                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, 1)

                total += labels.shape[0]
                correct += (predicted == labels).sum().item()

            print(f'Accuracy of the model on test images: {100 * correct // total}%')

