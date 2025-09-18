import pandas as pd
import seaborn as sns 


# write a function to loasd and the data
def load_data():
    """Load the dataset and retirn a pandas Dataframe."""
    df = sns.load_dataset("tips")
    return df 