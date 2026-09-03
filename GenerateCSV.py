import argparse
from pathlib import Path
import re
import numpy as np
###
import pandas as pd

Lines =  ['Ha','Hb','Hg','H6','H8','H9','Pb','Brg']

def NameNormalize(LNames):
    mapping = {'mass':   'Mass_CA',
               'radius': 'Radius_CA',
               'Mdot':   'Mdot_Msun',
               'area': 'Area_Rsun',
               #'ff': 'ff_percent',
               'Av': 'AV',
               'B': 'BField',
               }
    Lnew = []
    for name in LNames:
        for old, new in mapping.items():
            if re.fullmatch( re.escape(old), name, flags=re.IGNORECASE):
                name = new
        ###########
        Lnew.append(name)
    ######
    return Lnew

def ResolveColNames(cols):
    Strs = [
        ##### IDs
        'ObservationID',
        'ObjectID',
        ##### Name and Date
        'TwoMassName',
        ##### Flags
        'IsMultiDate',
        'IsCalibrated',
    ]
    DirectCols = [
        ##### Object parameters
        "Mass_CA",
        "vff",
        "Radius_CA",
        ##### Literature Values
        #'Lacc_CA',
        #'LaccLin_CA',
    ]
    DirectCols = [ tem2 for tem in DirectCols
                   for tem2 in [tem, f'd{tem}']]
    DirectCols = DirectCols + [
        'Lacc_CA', 'Lacc_CA_range',
    ] ## not mode variables but have range
    ############################
    ## Columnv Names under Mode
    ModeCols = [
        ##### Fitting Parameters
        "v0",
        "n0",
        "ff",
        #"ff_percent",
        "Area_Rsun",
        "AV",
        ##### Combined
        "Mdot_Msun",
        "vRat",
        "BField",
        "Lacc",
        #"LaccRat",
    ]
    ModeCols = [t2 for tem in ModeCols
                for t2 in [tem, f'{tem}_range']]
    #######################    
    LumObsNames = [t2 for line in Lines
                   for t2 in [f'Lum_Obs_{line}', f'dLum_Obs_{line}']]
    LumBCNames  = [t2 for line in Lines[1:] ## no Ha
                   for t2 in [f'Lum_BC_{line}', f'dLum_BC_{line}']]
    ##########
    LumFitNames = [t2 for line in Lines
                   for t2 in [f'Lum_Fit_{line}', f'Lum_Fit_{line}_range']]
    DeRedLumObsNames = [t2 for line in Lines for t2 in
                        [f'DeRedLum_Obs_{line}', f'DeRedLum_Obs_{line}_range']]
    DeRedLumFitNames = [t2 for line in Lines for t2 in
                         [f'DeRedLum_Fit_{line}', f'DeRedLum_Fit_{line}_range']]
    DeRedLumBCNames  = [t2 for line in Lines[1:] for t2 in ## no Ha
                         [f'DeRedLum_BC_{line}', f'DeRedLum_BC_{line}_range']]
    # LObs_LaccNames   = [t2 for line in Lines for t2 in
    #                     [f'LObs_Lacc_{line}', f'LObs_Lacc_{line}_range']]
    # LFit_LaccNames   = [t2 for line in Lines for t2 in
    #                     [f'LFit_Lacc_{line}', f'LFit_Lacc_{line}_range']]
    # LBC_LaccNames    = [t2 for line in Lines for t2 in
    #                     [f'LBC_Lacc_{line}', f'LBC_Lacc_{line}_range']]
    #######################
    DirectCols = DirectCols + LumObsNames
    ModeCols   = (ModeCols +
                  LumFitNames + LumBCNames + 
                  DeRedLumObsNames + DeRedLumFitNames + DeRedLumBCNames
                  # +LObs_LaccNames + LFit_LaccNames + LBC_LaccNames
                  )    
    ###################################
    ## Leave the user-specified names
    if cols is not None:
        ColNames = NameNormalize(cols)
        DC       = [tem for tem in DirectCols if tem in ColNames]
        MC       = [tem for tem in ModeCols   if tem in ColNames]
        return DC, MC
    else:
        return DirectCols, ModeCols
    ##########        

def MakeModeIndex(df, DirectCols, ModeCols, flg_BestMode=False,
                  FidSuff = 'SpecFit_Fid.', BCSSuff='SpecFit_BCS.'):
    #################################
    ## Force to add basic variables #
    ModeCols=ModeCols + ['RefP-value', 'RefRedChi2']
    #############
    tmp1 = df.drop(columns=['Object','Date'], errors='ignore').reset_index()
    ################################
    ## temporally make mode index ##    
    def _ModeIndex_suff(suff):
        modes = sorted( {col.split(".", 2)[1]
                         for col in df.columns
                         if "." in col and col.startswith(suff+'Mode')
                         } )
        tmp2 = pd.concat( [
            tmp1[ ['Object', 'Date'] + DirectCols ].assign(
                _rowID=range(len(tmp1)),
                mode=mode,
                **{ suff + sub : tmp1[suff+f"{mode}.{sub}"]
                    for sub in ModeCols },
            )
            for mode in modes
        ] ,ignore_index=True,)
        ###############
        ref  = suff+'RefP-value'
        if flg_BestMode:
            tmp3 = (tmp2.dropna(subset=[ref])
                    .sort_values(ref, ascending=False)
                    .drop_duplicates('_rowID', keep='first')
                    .drop(columns=['_rowID', 'mode'])
                    .set_index(['Object', 'Date'])
                    )
        else:
            tmp3 = (tmp2.set_index(['Object', 'Date', 'mode'])
                    .drop(columns=['_rowID'])
                    .dropna(subset=[ref]) ## P-value with NaN means blank mode auto-generated via column-expansion in generating pandas dataframe
                    )                    
        return tmp3
    #################
    tmp_Fid = _ModeIndex_suff(FidSuff)
    tmp_BCS = _ModeIndex_suff(BCSSuff)
    #########
    ## merge
    idx    = ['Object', 'Date']
    if not flg_BestMode:
        idx.append('mode')
    ##############
    merged = pd.concat([tmp_Fid,
                        tmp_BCS.loc[:,
                                    tmp_BCS.columns.str.startswith(BCSSuff)
                                    #| tmp_BCS.columns.isin(['Object', 'Date', 'mode'])
                                    ]]
                       , axis=1)
    # merged = tmp_Fid.merge(
    #     tmp_BCS.loc[:,
    #                 tmp_BCS.columns.str.startswith(BCSSuff) |
    #                 tmp_BCS.columns.isin(['Object', 'Date', 'mode'])
    #                 ],
    #     on = idx,
    #     how = 'left')
    ########
    # uni_col = list( {
    #     col.split('.')[-1]: col
    #     for col in reversed(merged.columns)
    # }.values())[::-1]
    # print(uni_col)                    
    return merged

def OpenRangeCell(df, cols):
    RangeCols = [tem for tem in cols if tem.endswith('_range')]
    new_df    = df.copy()
    for col in RangeCols:
        name = col[:-6]
        #print(col, name)
        vals = df[col].apply(
            lambda xx: xx if isinstance(xx, (list, tuple, np.ndarray))
            else [np.nan, np.nan]
        )
        # new_df[ [name+'_low', name+'_up'] ] = pd.DataFrame(
        #     vals.tolist(),
        #     index=new_df.index )
        new_df = pd.concat( [new_df,
                             pd.DataFrame(vals.tolist(),
                                          columns = [ name+'_low', name+'_up'],
                                          index=df.index)
                             ],
                            axis=1)
    ###################
    ## reorder
    ordered = []
    for col in cols:
        if col.endswith('_range'):
            continue
        ###
        ordered.append(col)
        if col+'_range' in RangeCols:
            ordered += [col+'_low', col+'_up']
        #########
    #######
    ordered_df = new_df[ordered].copy()
    return ordered_df, ordered


def DFCaution(text):
    answer = input( text + ' Continue? [Y/n]:')
    if answer.lower() in ['n', 'no']:
        print('Cancelled.')
        raise SystemExit

def main(args):
    if args.data_path.exists():
        Base_df  = pd.read_parquet(args.data_path)
    else:
        DFCaution(f'{args.data_path} is not found. Sample data will be instead used.')
        Base_df  = pd.read_parquet( Path('tests/data/sample.parquet'))
    ####    
    DataPath = args.save_path
    DirectCols, ModeCols = ResolveColNames(args.cols)
    #########################
    PrefixFid = "SpecFit_Fid."
    PrefixBCS = "SpecFit_BCS."
    Clist = []
    Lpf = []
    LMask = []
    Mask_SpecFid = Base_df[PrefixFid + "Global.RefP-value"] > 0.05
    Mask_SpecBCS = Base_df[PrefixBCS + "Global.RefP-value"] > 0.05
    Mask_SpecBCSChi2 = Base_df[PrefixBCS + "Global.RefRedChi2"] < 6
    ###########
    Base_df  = MakeModeIndex(Base_df, DirectCols, ModeCols,
                             flg_BestMode=args.best_mode,
                             FidSuff = PrefixFid,
                             BCSSuff = PrefixBCS,
                             )               
    #########
    OutputCols = DirectCols
    OutputMask  = Mask_SpecFid ### this is the minimum set.
    if args.ShowFid:
        OutputCols = OutputCols + [PrefixFid + tem for tem in ModeCols]
    if args.ShowBCS or args.ShowBCSlike:
        OutputCols = OutputCols + [PrefixBCS + tem for tem in ModeCols]
        if args.ShowBCSlike:
            OutputMask  = Mask_SpecBCSChi2 ## wider
        else:            
            OutputMask  = Mask_SpecBCS
        #######
    ###########
    #######################
    #### additional flags #
    if args.multi_epoch:
        OutputMask = (OutputMask & Base_df["IsMultiDate"])
    #######
    if args.calibrated:
        OutputMask = (OutputMask & Base_df["IsCalibrated"])
    #######################
    if args.hide_error:
        OutputCols = [tem for tem in OutputCols
                      if not ('.d' in tem or tem.endswith('_range'))]
    #########################
    ## open the _range cell #
    #################
    Base_df, OutputCols = OpenRangeCell(Base_df, OutputCols)
    ##########
    ## save ##
    Base_df.loc[OutputMask, OutputCols].to_csv(DataPath/args.file_name,
                                            index=True,na_rep=args.nan,
                                            float_format=args.float_format)
    print( (DataPath/args.file_name).name + ' is succesfully generated')
################################################


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
        default=cwd / "Fig",
        help="Directory name saving the data table",
    )
    parse.add_argument(
        "--file-name",
        type=str,
        default= "data.csv",
        help="File name",
    )
    ########
    ## format
    parse.add_argument(
        "--float-format",
        type=str,
        default= "%.6e",
        help="Output format of float",
    )
    parse.add_argument(
        "--nan",
        type=str,
        default= "NaN",
        help="Output format of nan values",
    )
    
    parse.add_argument(
        "--cols", default=None, nargs="+", help="Variable name. Default=ALL"
    )
    parse.add_argument(
        "--best-mode", action="store_true", help="Show only the best mode"
    )
    parse.add_argument(
        "--hide-fid", action="store_false", dest="ShowFid", help="Hide fiducial results"
    )
    parse.add_argument(
        "--hide-BCS", action="store_false", dest="ShowBCS", help="Hide BCS results. BCS-like samples are also removed."
    )
    parse.add_argument(
        "--hide-BCSlike", action="store_false", dest="ShowBCSlike", help="Hide BCS-like samples"
    )
    parse.add_argument(
        "--hide-error", action="store_true", help="Hide uncertainty range of its standard deviation"
    )
    parse.add_argument(
        "--multi-epoch",
        action="store_true",
        help="Show only samples observed on multiple dates",
    )
    parse.add_argument(
        "--calibrated",
        action="store_true",
        help="Show only the slit-loss calibrated samples",
    )    
    args = parse.parse_args()
    args.save_path.mkdir(parents=False, exist_ok=True)
    main(args)
