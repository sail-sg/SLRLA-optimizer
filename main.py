# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from _preamble_ import *
from models import get_networks
from models.LookAhead import Lookahead
from dataset import get_dataset
from utils import timer, get_epoch_logger, get_logger

parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--SEED', default=0, type=int)
parser.add_argument('--is_Train', action='store_true')
parser.add_argument('--dataset', default='CIFAR10')
parser.add_argument('--data_path', default='./Data/CIFAR')
parser.add_argument('--network', default='ResNet-18', choices=['ResNet-50', 'ResNet-34', 'ResNet-18', 'ResNet-12', 'ResNet-10', \
                                                                'WRN-28-10', 'WRN-16-10', 'WRN-16-8', 'VGG-11', 'VGG-19'])
parser.add_argument('--root', default='./experiments')
parser.add_argument('--exp_name', default='cifar10_resnet18_sgd_zp')
parser.add_argument('--tr_epochs', default=100, type=int)
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-3, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--milestones', nargs='+', default=[0.3, 0.6, 0.8], type=float)
parser.add_argument('--gamma', type=float, default=0.2)
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--input_norm', action='store_true')
parser.add_argument('--optim', default='SGD', type=str)
parser.add_argument('--lookahead', default=None)
parser.add_argument('--slr', default=None)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint', type=str, default='./experiments/')
parser.add_argument('--net_only', default=True, type=lambda x: bool(int(x)))
args, _ = parser.parse_known_args()

np.random.seed(args.SEED)
torch.manual_seed(args.SEED)

if args.lookahead is None:
    Exp_Path = os.path.join(args.root, '_'+args.dataset+'_'+args.network, args.exp_name + 
                        '_wd' + '{:.1e}'.format(args.weight_decay) + '_lr'+'{:.1e}'.format(args.lr))
elif args.slr is None:
    Exp_Path = os.path.join( args.root, '_'+args.dataset+'_'+args.network, args.exp_name + 
                        '_wd' + '{:.1e}'.format(args.weight_decay) + '_lr'+'{:.1e}'.format(args.lr) + '_LA' + args.lookahead )
else:
    Exp_Path = os.path.join( args.root, '_'+args.dataset+'_'+args.network, args.exp_name + 
                        '_wd' + '{:.1e}'.format(args.weight_decay) + '_lr'+'{:.1e}'.format(args.lr) + '_LA' + args.lookahead +'_SLR' + args.slr )
pathlib.Path(os.path.join(Exp_Path, 'nets')).mkdir(parents=True, exist_ok=True)

def topk_accuracy(predict, device='cuda', test_loader=None, num_batch=None, topk=5, criterion=None, target_transform=None):
    clncorrect = 0
    idx_batch = 0
    lst_label = []
    lst_pred = []
    lst_loss = []
    assert topk>=5
    for clndata, target in tqdm.tqdm(test_loader):
        if target_transform is not None:
            target = target_transform(target)
        clndata, target = clndata.to(device), target.to(device)
        num = target.shape[0]
        with torch.no_grad():
            output = predict(clndata)
            if criterion is not None:
                lst_loss.append(criterion(output, target))
            else:
                lst_loss.append(torch.ones(num,1)*1e6)
            pred = output.topk(topk, dim=1)[1]
            lst_label.append(target)
            lst_pred.append(pred)
            idx_batch += 1
            if idx_batch == num_batch:
                break
    label = torch.cat(lst_label).view(-1, 1).cpu().numpy()
    pred = torch.cat(lst_pred).view(-1, topk).cpu().numpy()
    loss = torch.cat(lst_loss).view(-1,1).cpu().numpy()
    num = label.shape[0]
    top1_acc = (label == pred[:, :1]).sum() / num
    top5_acc = (label == pred[:, :5]).sum() / num
    avg_loss = np.mean(loss)
    message = f'***** Val set Top-1 acc: {top1_acc :.2f}; Top-5 acc: {top5_acc :.2f}'
    return top1_acc, avg_loss, message

class regularized_LA_loss():
    def __init__(self, beta=0.1, step=5) -> None:
        self.beta = beta
        self.local_penalty = nn.MSELoss(size_average=False)
        self._steps = 0
        self.K = step
        self.params_cache = defaultdict(dict)
    def __call__(self, data, labels, net, is_step=False):
        loss = F.cross_entropy(net(data), labels)
        reg_loss = 0
        if self._steps % self.K == 0:
            for name, param in net.named_parameters():
                current_param = param.clone().detach()
                self.params_cache[name] = current_param
                reg_loss += self.local_penalty(param, current_param)
        else:
            for name, param in net.named_parameters():   
                reg_loss += self.local_penalty(param, self.params_cache[name])
        loss += reg_loss * self.beta
        if is_step:
            self.step()
        return loss
    def step(self):
        self._steps += 1


class Classifier():
    def __init__(self, network_type, in_channels, num_classes, is_train, tr_epochs, optimizer, lookahead, slr, 
                 lr, momentum, milestones_ratio, gamma, weight_decay, device='cuda') -> None:
        super().__init__()
        self.net = get_networks(network_type)(in_channels, num_classes).to(device)
        self.device = device
        self.num_GPUs = torch.cuda.device_count()
        self.lookahead = lookahead
        self.slr = slr
        if self.num_GPUs > 1:
            self.net = nn.DataParallel(module=self.net, device_ids=list(range(self.num_GPUs)))
        if is_train:
            if optimizer == 'SGD':
                self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            elif optimizer == 'Adam':
                self.optimizer = optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            elif optimizer == 'AdamW':
                self.optimizer = optim.AdamW(self.net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            else:
                assert False, 'Optimizer not found!'
            if lookahead is not None:
                la_steps, la_alpha = lookahead.split('_')
                self.optimizer = Lookahead(self.optimizer, la_steps=int(la_steps), la_alpha=float(la_alpha))
            if slr is not None:
                slr_steps, slr_beta = args.slr.split('_')
                self.criterion = regularized_LA_loss(float(slr_beta), int(slr_steps))
            else:
                self.criterion = nn.CrossEntropyLoss()
            milestones_epoch = [int(s * tr_epochs) for s in milestones_ratio] 
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones_epoch, gamma=gamma)
            self.tr_epochs = tr_epochs
            self.start_epoch = 0
            self.log_interval = 50
        else:
            self.eval_mode()
            self.set_requires_grad([self.net], False)

    def eval_mode(self):
        self.net.eval()
        
    def train_mode(self):
        self.net.train()
        
    def load_networks(self, path):
        self.checkpoint = torch.load(path)
        if self.num_GPUs == 1:
            self.net.load_state_dict(self.checkpoint['state_dict'])
        else:
            self.net.module.load_state_dict(self.checkpoint['state_dict'])
    
    def resume_training(self, path, net_only=True):
        self.load_networks(path)
        if not net_only:
            self.start_epoch = self.checkpoint['stop_epoch']
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.scheduler.last_epoch = self.start_epoch
            
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
         
    def fit(self, train_loader, test_loader, logger, save_path='.', logging_tr=True, target_transform=None):        
        tic_toc = timer()
        epoch_logger = get_epoch_logger()
        for epoch in range(self.start_epoch, self.tr_epochs):
            logger.info('Training Epoch: {}; Learning rate: {:0.8f}  .....'.format(epoch, self.optimizer.param_groups[0]['lr']))
            self.train_mode()
            self._fit_one_epoch(train_loader, epoch, logger=logger)
            if logging_tr:
                _, _ = self._evaluate(epoch, train_loader, tag='train', logger=logger, target_transform=target_transform)
            result, _ = self._evaluate(epoch, test_loader, logger=logger, target_transform=target_transform)
            epoch_logger.append_results([epoch, result])
            best_epoch, message = epoch_logger.update_best_epoch()
            model_state_dict = self.net.state_dict() if self.num_GPUs<=1 else self.net.module.state_dict()
            checkpoint = {'state_dict':model_state_dict, 'stop_epoch':epoch, 'optimizer': self.optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(save_path, "ckp_latest.pt"))
            if best_epoch == epoch:
                torch.save(checkpoint, os.path.join(save_path, "ckp_best.pt"))
            logger.info(message + '\t' + 'Time for an epoch: {:.2f}s.\n'.format(tic_toc.toc()))
                       
    def _fit_one_epoch(self, train_loader, epoch, logger):
        tic_toc = timer()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)      
            self.optimizer.zero_grad()
            if self.slr is None:
                loss = self.criterion(self.net(data), target)
            else:
                loss = self.criterion(data, target, self.net, is_step=True)
            loss.backward()
            self.optimizer.step()
            if (batch_idx+1) % self.log_interval == 0:
                logger.info('[{}/{} ({:.0f}%)], Loss: {:.6f}, Time for Batches: {:03f}'.format(
                    batch_idx *len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                    loss.item(), 
                    tic_toc.toc()))
        if self.scheduler is not None:
            self.scheduler.step()
            
    def _evaluate(self, epoch, loader, tag='test', logger=None, target_transform=None):
        if self.lookahead is not None:
            self.optimizer._backup_and_load_cache()
        self.eval_mode()
        accuracy, _, message = topk_accuracy(
                            self.net, device=self.device , test_loader=loader, criterion=nn.CrossEntropyLoss(reduction='none'))
        if self.lookahead is not None:
            self.optimizer._clear_and_load_backup()
        if logger is not None:
            logger.info(message)
        return accuracy, message
    
def main():
    logger = get_logger(os.path.join(Exp_Path, 'logging.txt'))
    logger.info(args)
    loader, train_loader, test_loader = get_dataset(name=args.dataset, batch_size=args.train_batch_size, num_workers=6, input_norm=args.input_norm, data_path=args.data_path)
    model = Classifier(network_type=args.network, in_channels=loader.img_channels, num_classes=loader.num_classes, 
                        is_train=True, tr_epochs=args.tr_epochs, 
                        optimizer=args.optim, lookahead=args.lookahead, slr=args.slr, 
                        lr=args.lr, momentum=args.momentum, milestones_ratio=args.milestones, gamma=args.gamma,
                        weight_decay=args.weight_decay)
    logger.info(model.net)
    if args.resume:
        model.resume_training(path=args.checkpoint, net_only=args.net_only)
    model.fit(train_loader=train_loader, test_loader=test_loader, 
                logger=logger, save_path=os.path.join(Exp_Path, 'nets'), logging_tr=False)  

if __name__ == '__main__':
    main()

        

