import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

if __name__ == "__main__":
    path = 'outputs/REDBULL_RAMBO/laps/'    
    all_files = glob.glob(path + "*.csv")
    df_list = [pd.read_csv(f) for f in all_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv('redbull_rambo.csv', index=False)

    path = 'outputs/MONZA_RAMBO/laps/'     
    all_files = glob.glob(path + "*.csv")
    df_list = [pd.read_csv(f) for f in all_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv('monza_rambo.csv', index=False)

    path = 'outputs/BARCELONA_RAMBO/laps/'     
    all_files = glob.glob(path + "*.csv")
    df_list = [pd.read_csv(f) for f in all_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv('barcelona_rambo.csv', index=False)
