import click
import pickle
import pandas as pd

@click.command(name="znorm_params")
@click.option("-m", "--matrix",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              default=None,
              help="DeepTumour's preprocess matrix")
@click.option("-o", "--output",
              type=click.STRING,
              default="z-norm",
              help="Name to save the z-norm parameters")
def znorm_params(matrix, output):
    """
    Function to save the z-norm parameters to use during DeepTumour preprocessing step
    """

    # Load the matrix that will be used for training DeepTumour
    dt_matrix:pd.DataFrame = pd.read_csv(matrix)

    # Keep only the columns that are z-norm processed
    dt_matrix = dt_matrix[[col for col in dt_matrix.columns if ".." in col]]

    # Save the z-norm column-wise parameters in a dictionary
    znorm_dict:dict = {}
    for col in dt_matrix.columns:
        col_mean = dt_matrix[col].mean()
        col_std = dt_matrix[col].std()
        znorm_dict[col] = {'mean': col_mean, 'std': col_std}
    
    # Save the params_dict to a pickle file
    with open(f'{output}.pkl', 'wb') as f:
        pickle.dump(znorm_dict, f)

if __name__ == '__main__':
    znorm_params()
