# Import model path
import sys
sys.path.append('/oncoGAN/models')

# Import modules
import os
import time
import click
import traceback
import json
import pickle
import torch
import numpy as np
import pandas as pd

from caloforest.writer import get_writer
from caloforest.forest_diffusion import ForestModel

# CLI options
@click.command(name='trainCaloForest')
@click.option("--csv",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="CSV file to train the model. The column with the predicted labels must be the last column")
@click.option("--config",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="Config file to use for training the model")
@click.option('--prefix',
              type=click.STRING,
              required=True,
              help="Prefix to be used for saving files")
def TrainCaloForest(csv, config, prefix):

    """
    Command to train a Calo-Forest model on tabular data
    """

    # Load configuration file
    with open(config, "r") as f:
        cfg:dict = json.load(f)

    # Check configuration consistency
    if not cfg["ycond"] and cfg["ylabel"] is not None:
        raise ValueError('ylabel must be null when ycond is false in the config.json file')
    if cfg["ycond"] and cfg["ylabel"] is None:
        raise ValueError('ylabel cannot be null when ycond is true in the config.json file')

    # Set up writer functions
    writer = get_writer(os.path.basename(csv).rstrip('.csv'), cfg=cfg)

    # Seeds
    np.random.seed(seed=cfg["seed"])
    torch.manual_seed(cfg["seed"]) # torch is only used for calorimeter evaluation
    torch.cuda.manual_seed(cfg["seed"])

    try:
        # Open the training file
        df:pd.DataFrame = pd.read_csv(csv)

        # Package XGB hyperparameters
        hyper_names:list = ["max_depth", "n_estimators", "eta", "min_child_weight", "gamma", "lambda", "multi_strategy", "early_stopping_rounds", "device"]
        xgb_hypers:dict = {k: v for k, v in cfg.items() if k in hyper_names}
        xgb_hypers["n_jobs"] = cfg["xgb_n_jobs"]

        # Start the model
        print("Starting forest_model")
        forest_model = ForestModel(
            n_t=cfg["n_t"],
            diffusion_type=cfg["diffusion_type"],
            xgb_hypers=xgb_hypers,
            duplicate_K=cfg["duplicate_K"],
            cat_indexes=[],
            bin_indexes=[],
            int_indexes=[],
            eps=cfg["eps"],
            beta_min=cfg["beta_min"],
            beta_max=cfg["beta_max"],
            solver=cfg["solver"],
            scaler=cfg["scaler"],
            n_jobs=cfg["n_jobs"],
            backend=cfg["backend"],
            n_batch=cfg["n_batch"],
            seed=cfg["seed"],
            logdir=writer.logdir)
        
        if cfg["ycond"]:
            # Map prediction labels
            pred_col:str = cfg["ylabel"]
            feature_cols:pd.Index = df.columns[df.columns != pred_col]
            pred_codes, pred_uniques = pd.factorize(df[pred_col])
            code_mapping:dict = dict(enumerate(pred_uniques))
            df[pred_col] = pred_codes

            # Split X and y
            X:np.ndarray = df.loc[:, feature_cols].to_numpy()
            y:np.ndarray|None = df[pred_col].to_numpy()
        else:
            # Split X and y
            X:np.ndarray = df.to_numpy()
            y:np.ndarray|None = None
        
        # Preprocess the data
        prepro_X = forest_model.preprocess(X=X, label_y=y)

        # Save model wrapper as pickle
        writer.write_pickle('forest_model', 
                            {'model': forest_model, 
                             'columns': df.columns.values.tolist(), 
                             'mapping': code_mapping if cfg["ycond"] else None,
                             'cfg': cfg,
                             'y_label': pred_col if cfg["ycond"] else None,
                             'y_values': y,
                             'n': X.shape[0]})
        
        # Training
        t0 = time.time()
        forest_model.train(prepro_X)
        t1 = time.time()
        print(f"Done forest_model training in {t1-t0}s")
        
        # Inference
        forest_model.load_models()
        
        # Generate one set of samples using the labels from the train set
        t2 = time.time()
        Xy_fake:np.ndarray = forest_model.generate(batch_size=X.shape[0], label_y=y)
        t3 = time.time()
        print(f"Generated data in {t3-t2}s")

        # Map back the labels to their original values
        Xy_fake_df:pd.DataFrame = pd.DataFrame.from_records(Xy_fake, columns=df.columns.values.tolist())
        if cfg["ycond"]:
            Xy_fake_df[pred_col] = Xy_fake_df[pred_col].apply(lambda x: code_mapping[x])
        writer.write_pandas(f"{prefix}_calo_forest_simulations", Xy_fake_df)

    except Exception:
        traceback.print_exc()

@click.command(name='useCaloForest')
@click.option("-n",
              type=click.INT,
              required=False,
              help="Number of samples to generate in case of not using a y-label guiding model")
@click.option("--input",
              type=click.Path(exists=True, file_okay=True),
              required=False,
              help="A CSV file with two columns: n and label_y, used to simulate samples. If not provided, the same number of samples as in the training set will be generated")
@click.option("--load-dir", "load_dir",
              type=click.Path(exists=True, file_okay=False),
              required=True,
              help="Directory to load from")
@click.option('--prefix',
              type=click.STRING,
              required=True,
              help="Prefix to be used for saving the generated samples")
@click.option("--out-dir",
              type=click.Path(exists=True, file_okay=False),
              required=False,
              default=os.getcwd(),
              help="Directory to save the generated samples")
def UseCaloForest(n, input, load_dir, prefix, out_dir):
    
    """
    Command to quickly use a trained Calo-Forest model
    """

    # Load the forest model
    model:str = os.path.join(load_dir, 'forest_model.pkl')
    with open(model, 'rb') as file:
        model_dict:dict = pickle.load(file)
    model_dict['model'].set_logdir(load_dir)
    model_dict['model'].set_solver_fn(model_dict['cfg']["solver"])

    # Prepare the number and type of samples to generate
    if input is not None and model_dict['y_label'] is not None:
        df:pd.DataFrame = pd.read_csv(input)
        df = df.loc[df.index.repeat(df['n'])].reset_index(drop=True)

        reverse_mapping:dict = {v: k for k, v in model_dict['mapping'].items()}
        df['y_label'] = df['y_label'].apply(lambda x: reverse_mapping[x])
        
        n2:int = df.shape[0]
        y_values:np.ndarray|None = df['y_label'].to_numpy()
    else:
        n2:int = n if n else model_dict['n']
        y_values:np.ndarray|None = model_dict['y_values']

    # Generate one set of samples using the labels from the train set
    Xy_fake:np.ndarray = model_dict['model'].generate(batch_size=n2, label_y=y_values)

    # Map back the labels to their original values
    Xy_fake_df:pd.DataFrame = pd.DataFrame.from_records(Xy_fake, columns=model_dict['columns'])
    if model_dict['y_label'] is not None:
        pred_col:str = model_dict['y_label']
        Xy_fake_df[pred_col] = Xy_fake_df[pred_col].apply(lambda x: model_dict['mapping'][x])
    Xy_fake_df.to_csv(os.path.join(out_dir,f"{prefix}_calo_forest_simulations.csv"), index=False)
