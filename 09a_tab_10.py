import torch, os, sys, copy
import numpy as np
import pandas as pd
import torch.nn as nn
from gnns import *
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, precision_recall_curve, auc

def train(model, epochs = 100, gnn = True, lr = 1e-3, wdecay = 5e-4):
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = wdecay)
   model.train()
   losses = []
   best_state = None
   best_loss = 1e6
   for epoch in range(epochs):
      optimizer.zero_grad()
      if gnn:
         y_hat_log = model(data)
         loss = criterion(y_hat_log[data.train_mask], data.y[data.train_mask])
      else:
         y_hat_log = model(data.x_train)
         loss = criterion(y_hat_log, data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      loss = loss.item()
      if loss < best_loss:
         best_state = copy.deepcopy(model.state_dict())
         best_loss = loss
      losses.append(loss)
      if epoch % 10 == 0:
         print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
   return best_state

def test(model, tag):
   model.eval()
   with torch.no_grad():
      y_test_hat_softmax = model(data)[data.test_mask]
      y_test_hat = torch.max(y_test_hat_softmax.data, 1)
   y_test_np = data.y[data.test_mask].cpu().numpy()
   y_test_hat_np = y_test_hat.indices.cpu().numpy()
   y_test_micro = np.array([(data.y[data.test_mask].cpu().numpy() == i).astype(int) for i in range(len(data.y.unique()))]).ravel()
   precision, recall, thresholds = precision_recall_curve(y_test_micro, y_test_hat_softmax.data.cpu().numpy().ravel(order = "F"))
   with open("fig_9_raw.tsv", 'a') as f:
      f.write(f"{task}\t{bb_method}\t{bb_decorator}\t{auc(recall, precision)}\n")
   print(f"Model {task} {tag} AU PRC: {auc(recall, precision):.4f}")

def run(classifier, lr, params, n_hidden, wdecay, dropout, epochs):
   if classifier == "gcn":
      model = model_params[classifier]["model"](data.x.shape[1], len(data.y.unique()) * params, len(data.y.unique()), n_hidden = n_hidden, dropout = dropout).to(device)
   else:
      model = model_params[classifier]["model"](data.x.shape[1], len(data.y.unique()) * params, len(data.y.unique()), n_hidden = n_hidden).to(device)
   best_state = train(model, gnn = model_params[classifier]["gnn"], lr = lr, wdecay = wdecay, epochs = epochs)
   model.load_state_dict(best_state)
   test(model, classifier)
   del model
   torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"

lr = 0.001
params = 1
n_hidden = 0
wdecay = 5e-3
dropout = 0.5
epochs = 500

model_params = {
   "gcn": {"model": GCN_classifier, "gnn": True}
}

for task in ("country", "genre", "label", "year", "success"):
   for bb_method in ("df", "nc", "naive"):
      for bb_decorator in ("top6", "classic"):
         sys.stderr.write(f"{task}\t{bb_method}\t{bb_decorator}\n")
         data = torch.load(f"tensor_{bb_method}_{bb_decorator}_{task}.pt", weights_only = False).to(device)
         data.test_mask = ~data.train_mask
         if task in ("year", "success"):
            data.y = (data.y > data.y.median()).long()
         elif task == "genre":
            data.edge_index = add_self_loops(data.edge_index)[0]
            data.edge_attr = torch.cat((data.edge_attr, torch.ones(data.edge_index.shape[1] - data.edge_attr.shape[0], 1).to(device)))
         run("gcn", lr, params, n_hidden, wdecay, dropout, epochs)


