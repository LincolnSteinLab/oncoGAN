#!/usr/local/bin/python3

import os
import sys
import click
import json
import numpy as np
import pandas as pd
from utils import vcf2input
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class MLP(torch.nn.Module):
    def __init__(self, num_fc_layers, num_fc_units, dropout_rate):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(3047, num_fc_units))
        self.layers.append(nn.ReLU(True))
        self.layers.append(nn.Dropout(p=dropout_rate))
        for i in range(num_fc_layers):
            self.layers.append(nn.Linear(num_fc_units, num_fc_units))
            self.layers.append(nn.ReLU(True))
            self.layers.append(nn.Dropout(p=dropout_rate))

        self.layers.append(nn.Linear(num_fc_units, 29))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        return x

    def feature_list(self, x):
        out_list = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            out_list.append(x)

        return out_list

    def intermediate_forward(self, x, layer_index):
        for i in range(layer_index):
            x = self.layers[i](x)

        return x

class EnsembleClassifier(nn.Module):
    def __init__(self, model_list):
        super(EnsembleClassifier, self).__init__()
        self.model_list = model_list

    def forward(self, x):
        logit_list = []
        for model in self.model_list:
            model.eval()
            logits = model(x)
            logit_list.append(logits)

        return logit_list

class EnsemblePredictor(nn.Module):
    # This is the ensemble to construct when making predictions on PCAWG data.
    # Exmnple of how to construct it and use it is available in the main() function
    def __init__(self, model):
        super(EnsemblePredictor, self).__init__()
        self.model = model

    def predict_proba(self, x):
        logits_list = self.model.forward(x)
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
        probs_tensor = torch.stack(probs_list, dim=2)
        probability = probs_tensor.detach().cpu().numpy()
        probability = np.asarray([p.mean(1) for p in probability])

        return probability

    def predict(self, x):
        probs = self.predict_proba(x)
        predictions = np.argmax(probs, axis=1)

        return predictions

    def per_set_entropy(self, x):
        logits_list = self.model.forward(x)
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
        probs_tensor = torch.stack(probs_list, dim=2)
        probs = torch.mean(probs_tensor, dim=2)

        def entropy(prob_array, eps=1e-8):
            return -np.sum(np.log(prob_array + eps) * prob_array, axis=0)

        probs = [x.detach().numpy() for x in probs]
        entropy_list = [entropy(x) for x in probs]

        return entropy_list

class CompleteEnsemble(nn.Module):
    # Models in this case are of EnsemblePredictors
    # This is the ensemble to use when testing on non-PCAWG data
    # An example of how to construct it is in the main() function
    def __init__(self, model_list):
        super(CompleteEnsemble, self).__init__()
        self.model_list = model_list

    def forward(self, x):
        list_of_lists = []
        for model in self.model_list:
            logits = model.forward(x)
            list_of_lists.append(logits)

        return list_of_lists

    def get_entropy(self, x):
        entropy = np.mean([model.per_set_entropy(x) for model in self.model_list], 0)

        return entropy

    def predict_proba(self, x):
        probs_list = [model.predict_proba(x) for model in self.model_list]
        probs = np.mean(probs_list, 0)

        return probs

    def predict(self, x):
        probs = self.predict_proba(x)
        predictions = np.argmax(probs, axis=1)

        return predictions

class EnsembleClassifierAvg(nn.Module):
    def __init__(self, model_list):
        super(EnsembleClassifierAvg, self).__init__()
        self.model_list = model_list

    def forward(self, x):
        logit_list = []
        for model in self.model_list:
            model.eval()
            logits = model(x)
            logit_list.append(logits)
        output = torch.mean(torch.stack(logit_list, 0), dim=0)

        return output

class MyDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        return x, y

    def __len__(self):
        return len(self.data)

@click.command(name="DeepTumour")
@click.option("--vcfFile", "vcfFile",
              type=click.Path(exists=True, file_okay=True),
              required=False,
              default=None,
              help="VCF file to analyze [Use --vcfFile or --vcfDir]")
@click.option("--vcfDir", "vcfDir",
              type=click.Path(exists=True, file_okay=False),
              required=False,
              default=None,
              help="Directory with VCF files to analyze [Use --vcfFile or --vcfDir]")
@click.option("--hg19", "refGenome",
              type=click.Path(exists=True, file_okay=True),
              required = True,
              help="hg19 reference genome in fasta format")
@click.option("--liftOver", "liftOver",
              is_flag=True,
              required = False,
              help="Use this tag if your VCF is in hg38 coordinates")
@click.option("--keep_input", "keep_input",
              is_flag=True,
              required = False,
              help="Use this tag ito also save DeepTumour input as a csv file")
@click.option("--outDir", "outDir",
              type=click.Path(exists=True, file_okay=False),
              default=os.getcwd(),
              show_default=False,
              help="Directory where save DeepTumour results. Default is the current directory")
def DeepTumour(vcfFile, vcfDir, refGenome, liftOver, keep_input, outDir):
    
    """
    Predict cancer type from a VCF file using DeepTumour
    """

    # Generate the DeepTumour input file from the VCFs
    if vcfFile and not vcfDir:
        input:pd.DataFrame = vcf2input(vcfFile, refGenome, liftOver)
    elif vcfDir and not vcfFile:
        input:pd.DataFrame = pd.DataFrame()
        for file in os.listdir(vcfDir):
            if file.endswith('.vcf'):
                input = pd.concat([input, vcf2input(os.path.join(vcfDir, file), refGenome, liftOver)])
    else:
        raise ValueError('Please provide either a VCF file or a directory with VCF files')
    
    # Save the input file used by DeepTumour
    if keep_input:
        input.to_csv(os.path.join(outDir, 'DeepTumour_preprocess_input.csv'), index=False)
    
    # Load the models
    complete_ensemble = torch.load('/DeepTumour/trained_models/complete_ensemble.pt', map_location=torch.device("cpu"))
    cancer_label:pd.Series = pd.read_csv('/DeepTumour/trained_models/rare_cancer_factors.csv')['Cancer']

    # Separate labels and matrices
    labels:pd.Series = input['index']
    input.drop('index', axis=1, inplace=True)
    matrix:torch.tensor = torch.from_numpy(input.to_numpy()).float()

    # Make predictions
    with torch.no_grad():
        probs = complete_ensemble.predict_proba(matrix)
        prediction = complete_ensemble.predict(matrix)
        entropy = complete_ensemble.get_entropy(matrix)
    
    # Extract the results
    result:dict = {}
    for i, label in enumerate(labels):
        cancer_probs:dict = {}
        for cancer, prob in zip(cancer_label, probs[i]):
            cancer_probs[cancer] = float(prob)
        result[label] = {
            'probs': cancer_probs,
            'prediction': cancer_label[prediction[i]],
            'entropy': float(entropy[i])
        }

    # Save the results
    with open(os.path.join(outDir, 'predictions_DeepTumour.json'), 'w') as file:
        json.dump(result, file, indent=4, sort_keys=True)

if __name__ == '__main__':
    DeepTumour()
