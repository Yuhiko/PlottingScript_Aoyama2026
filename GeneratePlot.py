import argparse
from pathlib import Path
from functools import partial
import re
###
import pandas as pd
# import smplotlib # Uncomment to use the classical SuperMongo/IDL style.
######
import PlotLib as PL


class PlotParams:
    def __init__(self):
        self.Color_BCSlike = (0.5, 0, 0.5, 0.6)
        self.Color_BCS = (0, 0, 1, 1)
        self.Color_Fid = (1, 0, 0, 1)
        ####
        self.ColNames = [
            ##### Object parameters
            "Mass_CA",
            "vff",
            "Radius_CA",
            ##### Fitting Parameters
            "v0",
            "n0",
            "ff",
            "Area_Rsun",
            "AV",
            ##### Combined
            "Mdot_Msun",
            "vRat",
            "BField",
            "Lacc",
            ##### Literature Values
            #'LaccLin_CA',
            'Lacc_CA',
            #"LaccRat",
        ]
        Lines =  ['Ha','Hb','Hg','H6','H8','H9','Pb','Brg']
        self.LumObsNames = [f'Lum_Obs_{line}' for line in Lines]
        self.LumFitNames = [f'Lum_Fit_{line}' for line in Lines[:]]
        self.LumBCNames  = [f'Lum_BC_{line}' for line in Lines[1:]]
        self.DeRedLumObsNames  = [f'DeRedLum_Obs_{line}' for line in Lines[:]]
        self.DeRedLumFitNames  = [f'DeRedLum_Fit_{line}' for line in Lines[:]]
        self.DeRedLumBCNames   = [f'DeRedLum_BC_{line}' for line in Lines[1:]]
        # self.LObs_LaccNames  = [f'LObs_Lacc_{line}' for line in Lines[:]]
        # self.LFit_LaccNames  = [f'LFit_Lacc_{line}' for line in Lines[:]]
        # self.LBC_LaccNames   = [f'LBC_Lacc_{line}' for line in Lines[1:]]
        ###
        self.ColNames = (self.ColNames +
                         self.LumObsNames + self.LumFitNames + self.LumBCNames + 
                         self.DeRedLumObsNames + self.DeRedLumFitNames + self.DeRedLumBCNames
                         #+ self.LObs_LaccNames + self.LFit_LaccNames + self.LBC_LaccNames
                         )
    ##########

def DFCaution(text, flg_pass=False):
    if flg_pass:
        return
    answer = input( text + ' Continue? [Y/n]:')
    if answer.lower() in ['n', 'no']:
        print('Cancelled.')
        raise SystemExit
    

def main(args):
    Params = PlotParams()
    if args.data_path.exists():
        Base_df  = pd.read_parquet(args.data_path)
    else:
        DFCaution(f'{args.data_path} is not found. Sample data will be instead used.',
                  args.yes)
        Base_df  = pd.read_parquet( Path('tests/data/sample.parquet'))
    ####    
    FigPath = args.save_path
    xcol = args.x
    ycol = args.y
    if xcol is None and ycol is None:
        DFCaution(" No variable name was specified, so this script will generate about 400 figures.",
                  args.yes)
        flg_ALL = True
    else:
        flg_ALL = False
    ###########
    if xcol is None:
        xcol = Params.ColNames
    ##########
    if ycol is None:
        ycol = Params.ColNames
    ################
    xcol = NameNormalize(xcol)
    ycol = NameNormalize(ycol)
    #########################
    PrefixFid = "SpecFit_Fid."
    PrefixBCS = "SpecFit_BCS."
    Clist = []
    Lpf = []
    LMask = []
    Mask_SpecFid = Base_df[PrefixFid + "Global.RefP-value"] > 0.05
    Mask_SpecBCS = Base_df[PrefixBCS + "Global.RefP-value"] > 0.05
    Mask_SpecBCSChi2 = Base_df[PrefixBCS + "Global.RefRedChi2"] < 6
    if args.ShowBCSlike:
        LMask.append(~Mask_SpecFid & ~Mask_SpecBCS & Mask_SpecBCSChi2)
        Lpf.append(PrefixBCS)
        Clist.append(Params.Color_BCSlike)
    ###########
    if args.ShowBCS:
        LMask.append(~Mask_SpecFid & Mask_SpecBCS)
        Lpf.append(PrefixBCS)
        Clist.append(Params.Color_BCS)
    ###########
    if args.ShowFid:
        LMask.append(Mask_SpecFid)
        Lpf.append(PrefixFid)
        Clist.append(Params.Color_Fid)

    #######################
    #### additional flags #
    if args.multi_epoch:
        LMask = [tmask & Base_df["IsMultiDate"] for tmask in LMask]

    #######################
    ## Plotting Estimates #
    if flg_ALL:
        itotal = int( len(xcol) * (len(ycol)-1) / 2 )
    else:
        itotal = len(xcol) * len(ycol)
    ##################
    ## plot
    i2_0   = 0
    icount = 0
    for ii, name1 in enumerate(xcol):
        if flg_ALL:
            i2_0 = ii + 1
        for name2 in ycol[i2_0:]:
            icount += 1
            print(f'{icount:5d}/{itotal}: plotting {name1} -- {name2}')
            RefFunc = ReferenceFuncs(name1, name2)            
            PL.PlotErrScat(
                Base_df,
                name1,
                name2,
                Masks_in=LMask,
                prefixes_in=Lpf,
                SavePath=FigPath,
                Clist_in=Clist,
                Func_Drow=RefFunc,
                flg_MultiDate=args.multi_epoch,
            )
    ############
    print('Completed')
################################################


def ReferenceFuncs(xx, yy):
    def Lacc_Lline(aa, bb):
        if aa in ["Lacc", "Lacc_CA"]:
            if any([key in bb for key in ["Lum_BC", "Lum_Obs", "Lum_Fit"]]):
                line = bb.split("_")[-1]
                return line
        ####
        return None
    #####
    if {xx, yy} == {"Mass_Msun", "Mdot_Msun"}:
        Func = PL.Plot_EmpRel_MMdot
    elif {xx, yy} == {"Lacc", "Lacc_CA"}:
        Func = PL.Plot_yEQx
    elif (line := Lacc_Lline(xx, yy)) is not None:
        Func = partial(PL.Plot_EmpRel, line=line, flg_xLacc=True)
    elif (line := Lacc_Lline(yy, xx)) is not None:
        Func = partial(PL.Plot_EmpRel, line=line, flg_xLacc=False)
    else:
        Func = None
    ####
    return Func


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
        help="Directory name saving the Fits data",
    )
    parse.add_argument(
        "--x", default=None, nargs="+", help="Variable name in x-axis. Default=ALL"
    )
    parse.add_argument(
        "--y", default=None, nargs="+", help="Variable name in y-axis. Default=ALL"
    )
    parse.add_argument(
        "--hide-fid", action="store_false", dest="ShowFid", help="Hide fiducial samples"
    )
    parse.add_argument(
        "--hide-BCS", action="store_false", dest="ShowBCS", help="Hide BCS samples"
    )
    parse.add_argument(
        "--hide-BCSlike",
        action="store_false",
        dest="ShowBCSlike",
        help="Hide BCS-like samples",
    )
    parse.add_argument(
        "--multi-epoch",
        action="store_true",
        help="Show only samples observed on multiple dates, connected by lines",
    )
    parse.add_argument(
        '-y', '--yes',
        action='store true',
        help='Automatically answer to confirmation prompts'
    )
    # parse.add_argument("--hide-suspicious", action='store_true',
    #                    help='Hide suspicious samples, which are transparent by default')
    args = parse.parse_args()
    args.save_path.mkdir(parents=False, exist_ok=True)
    main(args)
