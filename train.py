import numpy as np
import yaml

from box import Box
from festanet import FESTA_Net
from data_module import DataModule_OpenAcessDatasets
from utils import *
from torchmetrics.classification import BinaryConfusionMatrix, MulticlassF1Score, MulticlassConfusionMatrix


def leave_one_out_cross_validation(config):
    ManualSeed(0)
    learning_rate = config.training.learning_rate
    num_batch = config.training.num_batch
    num_epochs = config.training.num_epochs
    min_epoch = 50
    
    path = 'D:/KMS/data/brain_2025' # your path
    
    dataset = DataModule_OpenAcessDatasets(
        path=path,
        data_mode=config.training.data_mode,
        label_type=config.training.label_type,
        num_val=config.training.num_val,
        batch_size=num_batch,
        transform_eeg=None,
        transform_fnirs=None
        )

    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    test_acc = []
    test_sen = []
    test_spc = []
    
    # confusion matrix
    cf_size = 2 if config.model.num_classes < 3 else 3
    cf_out = np.zeros((cf_size,cf_size),int)

    # leave-one-out cross-validation
    for subj, data_loaders in enumerate(dataset):
        train_loader, val_loader, test_loader = data_loaders

        model = FESTA_Net(
            eeg_shape=dataset.data_shape_eeg, 
            fnirs_shape=dataset.data_shape_fnirs, 
            num_segments=config.model.num_segments,
            embed_dim=config.model.embed_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            num_groups=config.model.num_groups,
            actv_mode=config.model.actv_mode,
            pool_mode=config.model.pool_mode, 
            k_size=config.model.k_size,
            hid_dim=config.model.hid_dim,
            num_classes=config.model.num_classes
            ).to(DEVICE)

        es = EarlyStopping(model, patience=10, mode='min')

        # train model
        train_results = trainer(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader,
            num_epoch=num_epochs, 
            optimizer_name=config.training.optimizer,
            learning_rate=str(learning_rate),
            early_stop=es,
            min_epoch=min_epoch,
            exlr_on=config.training.exlr_on,
            num_classes=config.model.num_classes
            )
        train_acc.append(train_results[0])
        train_loss.append(train_results[1])
        val_acc.append(train_results[2])
        val_loss.append(train_results[3])

        if es:
            model.load_state_dict(torch.load('best_model.pth'))

        # test model 
        test_results = tester(
            model=model, 
            tst_loader=test_loader, 
            num_classes=config.model.num_classes
            )
        
        # compute accuracy sensitivity specificity f1-score
        if config.model.num_classes > 1: # multiclass
            f1_ = MulticlassF1Score(num_classes=3, average='none')
            test_sen.append(list(f1_(torch.from_numpy(test_results[1]), torch.from_numpy(test_results[2]))))
            f1_ = MulticlassF1Score(num_classes=3, average='micro')
            test_acc.append(f1_(torch.from_numpy(test_results[1]), torch.from_numpy(test_results[2])) * 100)
            bcm = MulticlassConfusionMatrix(3)
            cf = bcm(torch.from_numpy(test_results[1]), torch.from_numpy(test_results[2]))
            cf_out += cf.numpy()
        else: # binary
            bcm = BinaryConfusionMatrix()
            cf = bcm(torch.from_numpy(test_results[1]), torch.from_numpy(test_results[2]))
            test_sen.append(cf[1,1]/(cf[1,1]+cf[1,0]))
            test_spc.append(cf[0,0]/(cf[0,0]+cf[0,1]))
            test_acc.append((cf[0,0]+cf[1,1])/(cf.sum()) * 100)
            cf_out += cf.numpy()

    # print cross-validation results
    if config.model.num_classes > 1: # multiclass
        test_sen = np.array(test_sen)
        print(f'[{config.training.data_mode} {config.training.label_type}]  avg Acc: {np.mean(test_acc):.2f} %,  std: {np.std(test_acc):.2f},'
              + f'f1: {np.mean(test_sen[:,0])*100:.2f}  {np.mean(test_sen[:,1])*100:.2f}  {np.mean(test_sen[:,2])*100:.2f}')
    else: # binary
        print(f'[{config.training.data_mode} {config.training.label_type}]  avg Acc: {np.mean(test_acc):.2f} %,  std: {np.std(test_acc):.2f},'
              + f'sen: {np.mean(test_sen)*100:.2f},  spc: {np.mean(test_spc)*100:.2f}')

    # plot confusion matrix
    if config.training.label_type == 2:
        lab = ['WG','Resting']
    elif config.training.label_type == 3:
        lab = ['DSR','Resting']
    elif config.training.label_type == 4:
        lab = ['0-back','2-back','3-back']
    plot_confusion_matrix(cf_out, lab)


if __name__ == "__main__":
    for task in ['wg','dsr','nback']:
        with open(f"yamls/{task}_best.yaml", 'r') as f:
            config_yaml = yaml.load(f, Loader=yaml.FullLoader)
            config = Box(config_yaml)
            leave_one_out_cross_validation(config)