import argparse
from pathlib import Path
import re
import numpy as np
###
import pandas as pd


def DFCaution(text, flg_pass=False):
    if flg_pass:
        return

    answer = input( text + ' Continue? [Y/n]:')
    if answer.lower() in ['n', 'no']:
        print('Cancelled.')
        raise SystemExit    



def ExDB_match( args , Base_df):
    raw   = pd.read_excel(args.exdb_path, header=None)
    names = raw.iloc[0]
    units = raw.iloc[1]
    units.index = names
    
    df_ori = raw.iloc[2:].copy()
    df_ori.columns = names
    df_ori.reset_index(drop=True, inplace=True)
    ################################
    ## correction for CASPAR typos #
    df_ori.loc[ (df_ori['Unique Name'] == '2MASS J16085143-3905304'),
                'GAIA EDR3 Soruce ID'] = '5997035416337166976'
    ################################
    df = df_ori.set_index( 'GAIA EDR3 Source ID').copy()              
    df_tem = df.groupby(level=0).first().copy() ## remove the duplications
    return df_tem
    
    

def main(args):
    if args.data_path.exists():
        readF = args.data_path
    else:
        DFCaution(f'{args.data_path} is not found. Sample data will be instead used.',
                  args.yes)
        readF = Path('tests/data/sample.parquet')
    ##########
    Base_df  = pd.read_parquet( readF, engine='pyarrow')
    #############
    ExDB_df = ExDB_match(args, Base_df)
    ####################
    Base_df['GAIA_EDR3_ID'] = Base_df['GAIA_EDR3_ID'].astype('string')
    merged = Base_df.join( ExDB_df[ args.exdb_cols ],
                           on='GAIA_EDR3_ID')
    print(merged)
    #########
    ## save #
    if args.save_path is None:
        DFCaution(f'Overwrite {readF}.')
        SaveFN = readF
    else:
        SaveFN = args.save_path
    #############
    merged.to_parquet( SaveFN, engine='pyarrow')
    

    
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    cwd = Path.cwd()
    parse.add_argument(
        "--data-path",
        type=Path,
        default=cwd / "Aoyama2026.parquet",
        help="JSON file path",
    )
    parse.add_argument(
        "--save-path",
        type=Path,
        default= None,
        help="Directory name saving the updated JSON file",
    )
    
    parse.add_argument(
        '-y', '--yes',
        action='store_true', dest='yes',
        help='Automatically answer to confirmation prompts'
    )
    parse.add_argument(
        '--exdb-path',
        type=Path,
        default=cwd/'CASPAR.csv',
        help='Datapath to external database. Default: CASPAR.csv. If you use another dabatase, pelase modify this script.'
    )
    parse.add_argument(
        '--exdb-cols',
        default=None, nargs="+",
        required=True,
        help="[Required] Variable names to be imported from the external database."
    )    
    args = parse.parse_args()
    main(args)
