from model.brainnetmlp import BrainNetMLP
from utilis import *
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json
import argparse

class ABIDE(Dataset):
    def __init__(self):
        super().__init__()
        pass

    def __len__(self):
        return train_feature_matrix.shape[0]

    def __getitem__(self, idx):
        return train_feature_matrix[idx], train_label[idx], train_time_series[idx]


class Trainer:
    def __init__(self, report=False):
        self.num_epoch = config['num_epoch']
        self.num_repeat = config['num_repeat']
        self.report = report
        test_accuracy_list, test_auroc_list = [], []
        test_sensitivity_list, test_specificity_list = [], []
        for _ in range(self.num_repeat):
            self.classifier = BrainNetMLP(config["dim"], config["hidden_dim"], config["dropout_rate"], config["norm"], config["k"]).to(device)
            self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config["learning_rate"])
            self.train_data_loader = torch.utils.data.DataLoader(ABIDE(), batch_size=batch_size, shuffle=False)
            self.train()
            test_acc, test_auroc, test_sens, test_spec = self.test()
            test_accuracy_list.append(test_acc)
            test_auroc_list.append(test_auroc)
            test_sensitivity_list.append(test_sens)
            test_specificity_list.append(test_spec)
        print(f"test acc: {test_accuracy_list}, test auroc: {test_auroc_list}, test sens: {test_sensitivity_list}, test spec: {test_specificity_list}")
        # torch.save(self.classifier.state_dict(), "./ckpt/full_mlp.pt")


    def train(self):
        self.classifier.train()
        for _ in tqdm(range(self.num_epoch)):
            for i_iter, data in enumerate(self.train_data_loader):
                feature, _, ts = data
                pred = self.classifier(feature, ts)
                self.optimizer.zero_grad()
                loss = config["loss_ratio"] * loss_func(pred, data[1])
                loss.backward()
                self.optimizer.step()


    def test(self):
        self.classifier.eval()
        with torch.no_grad():
            pred = self.classifier(test_feature_matrix, test_time_series)
            prob = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            test_correct = cal_accuracy(pred, test_label)
            test_au_roc = roc_auc_score(test_label.cpu().numpy(), prob[:, 1].cpu().numpy())
            sensitivity, specificity = cal_specificity_sensitivity(pred.cpu().numpy(), test_label.cpu().numpy())
        return test_correct,  test_au_roc, sensitivity, specificity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config/config.json', type=str)
    parser.add_argument('-d', '--dataset', default='abide', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('-s', '--save_result', action='store_true')
    args = parser.parse_args()
    config = json.load(open(args.config))[args.dataset]
    device = config['device']
    data_root = config['data_root']
    batch_size = config["batch_size"]
    set_seed(config["seed"])
    loss_func = torch.nn.CrossEntropyLoss(reduction=config["reduction"])
    time_series, feature_matrix, A, label = load_data(data_root, device=device)
    train_index, test_index = load_index(config["index_path"])
    train_feature_matrix, train_time_series = feature_matrix[train_index], time_series[train_index]
    test_feature_matrix, test_time_series = feature_matrix[test_index], time_series[test_index]
    train_label, test_label = label[train_index], label[test_index]
    Trainer()