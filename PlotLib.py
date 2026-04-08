import re
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

############
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

# import smplotlib
import astropy.constants as ac
import astropy.units as au


def Plot_yEQx(ax1):
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    low = max(xmin, ymin)
    high = min(xmax, ymax)
    ax1.plot([low, high], [low, high], color="grey", linestyle="--", lw=0.5)
    return ax1


Lls = ["-", "--", ":", "-."]


### x is Lacc
def Plot_EmpRel(
    ax1,
    line=None,
    flg_xLacc=True,
    ERlist=[
        "Alcala17",  #'Rigliaco12',
        "Aoyama21",
    ],
):
    # Lls = [ '-', '--', ':', '-.']
    for ii, ER in enumerate(ERlist):
        if ER == "Alcala17":
            if line == "Ha":
                aa, bb = (1.13, 1.74)
            elif line == "Hb":
                aa, bb = (1.14, 2.59)
            elif line == "Hg":
                aa, bb = (1.12, 2.69)
            elif line == "H6":
                aa, bb = (1.07, 2.64)
            elif line == "H7":
                aa, bb = (1.06, 2.69)
            elif line == "H8":
                aa, bb = (1.06, 2.73)
            elif line == "H9":
                aa, bb = (1.04, 2.78)
            elif line == "Pb":
                aa, bb = (1.06, 2.76)
            else:
                raise ValueError("PSFE.Plot_EmpRel: Please Specify Line Name")
        elif ER == "Rigliaco12":
            if line == "Ha":
                aa, bb = (1.49, 2.99)
            elif line == "Hb":
                aa, bb = (1.34, 3.01)
            elif line == "Hg":
                aa, bb = (1.30, 3.32)
            elif line == "Pb":
                aa, bb = (1.49, 4.59)
            else:
                continue
        elif ER == "Aoyama21":
            if line == "Ha":
                aa, bb = (0.95, 1.61)
            elif line == "Hb":
                aa, bb = (0.87, 1.47)
            elif line == "Hg":
                aa, bb = (0.85, 1.60)
            elif line == "H6":
                aa, bb = (0.84, 1.77)
            elif line == "H7":
                aa, bb = (0.83, 1.91)
            elif line == "H8":
                aa, bb = (0.83, 2.04)
            elif line == "Pb":
                aa, bb = (0.86, 2.21)
            else:
                continue
        #########
        if flg_xLacc:
            ymin, ymax = ax1.get_ylim()
            yy = np.linspace(ymin, ymax, 100)
            xx = 10 ** (aa * np.log10(yy) + bb)
        else:
            xmin, xmax = ax1.get_xlim()
            xx = np.linspace(xmin, xmax, 100)
            yy = 10 ** (aa * np.log10(xx) + bb)
        ##########
        ax1.plot(xx, yy, color="grey", linestyle=Lls[ii], lw=0.5)
    ######################
    return ax1


### x is Lacc
def Plot_EmpRel_MMdot(ax1, flg_xMdot=False, ERlist=["Betti23", "Betti23_LitHa"]):
    for ii, ER in enumerate(ERlist):
        if ER == "Betti23":
            aa = 2.02
            bb = -8.02
        elif ER == "Betti23_Lit":
            #### To Literature value in Betti+23, continuum measure.
            aa = 1.9
            bb = -8.5
        elif ER == "Betti23_LitLP":
            #### To Literature value in Betti+23, continuum measure.
            aa = 1.0
            bb = -9.9
        elif ER == "Betti23_LitHa":
            #### To Literature value in Betti+23, continuum measure.
            aa = 1.5
            bb = -7.5
        #########
        if flg_xMdot:
            ymin, ymax = ax1.get_ylim()
            yy = np.linspace(ymin, ymax, 100)
            xx = 10 ** (aa * np.log10(yy) + bb)
        else:
            xmin, xmax = ax1.get_xlim()
            xx = np.linspace(xmin, xmax, 100)
            yy = 10 ** (aa * np.log10(xx) + bb)
        ##########
        ax1.plot(xx, yy, color="grey", linestyle=Lls[ii], lw=0.5)
    ######################
    return ax1


def Plot_vff_vRat(ax1, vlim=200):
    xmin, xmax = ax1.get_xlim()
    xtem = np.linspace(xmin, xmax, 100)
    ytem = vlim / xtem
    ax1.plot(xtem, ytem, color="grey", linestyle="-", lw=0.5)
    return ax1


def Plot_Msun_vRat(ax1, vlim=200):
    xmin, xmax = ax1.get_xlim()
    xtem = np.linspace(xmin, xmax, 100)
    LLS = [":", "-.", "--", "-"]
    LR = [0.2, 0.5, 1.0, 2.0]
    for Robj, LS in zip(LR, LLS):
        ytem = (
            (vlim * au.km / au.s)
            * ((Robj * au.Rsun) / (2 * ac.G * xtem * au.Msun)) ** 0.5
        ).to_value(au.dimensionless_unscaled)

        ax1.plot(xtem, ytem, color="grey", linestyle=LS, lw=0.5)
    return ax1


def Plot_Msun_Rt(ax1, vlim=200, flg_Rsun=False):
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    xtem = np.linspace(xmin, xmax, 100)
    LLS = [":", "-.", "--", "-"]
    LR = [0.2, 0.5, 1.0, 2.0]
    for Robj, LS in zip(LR, LLS):
        # ytem = np.zeros_like(xtem)
        ytem = (
            1.0 / (Robj * au.Rsun)
            - (vlim * au.km / au.s) ** 2 / (2 * ac.G * xtem * au.Msun)
        ) ** -1
        if flg_Rsun:
            ytem = ytem.to_value(au.Rsun)
        else:
            ytem = (ytem / (Robj * au.Rsun)).to_value(au.dimensionless_unscaled)
        #########
        ytem[ytem < 0.0] = ymax
        ax1.plot(xtem, ytem, color="grey", linestyle=LS, lw=0.5)
    return ax1


def Plot_vff_Rt(ax1, vlim=200, flg_Rsun=False):
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    xtem = np.linspace(xmin, xmax, 100)
    LLS = [":", "-.", "--", "-"]
    LR = [0.2, 0.5, 1.0, 2.0]
    for Robj, LS in zip(LR, LLS):
        ytem = (1.0 - (vlim / xtem) ** 2) ** -1
        if flg_Rsun:
            ytem = ytem * Robj
        ##########
        ytem[ytem < 0.0] = ymax
        ax1.plot(xtem, ytem, color="grey", linestyle=LS, lw=0.5)
    return ax1


def PlotErrScat(
    Base_df,
    name1,
    name2,
    Clist_in,
    Masks_in=None,
    prefixes_in=None,
    # Modes = ['Global'],
    flg_Modes=True,
    #################
    ax_Range=None,
    flg_MultiDate=False,
    flg_MultiEffMode=True,
    flg_Suspecious=True,
    SolidConnect_in=None,
    PlaceSymbolLimit=None,
    ForceLog=False,
    Func_Drow=None,
    SavePath=Path.cwd(),
):
    suff = ""
    if not flg_Modes:
        flg_Suspecious = False
        flg_MultiEffMode = False
    ##############
    ## make axes #
    fx, fy = (5.35, 5)
    fig = plt.figure(figsize=(fx, fy))
    sfigy = 0.75
    stop = 0.08
    sbot = 1 - sfigy - stop
    ax1 = fig.add_axes((1 - ((sfigy + stop) * fy / fx), sbot, sfigy * fy / fx, sfigy))
    ax1, lab1, range1, log1, FR1 = labeling(ax1, name1, flg_x=True, Force_log=ForceLog)
    ax1, lab2, range2, log2, FR2 = labeling(ax1, name2, flg_x=False, Force_log=ForceLog)
    ### Check whether range plot or symmetric-error plot
    if ax_Range is None:
        ax_Range = FR1 + FR2
    flg_Range1 = "x" in ax_Range
    flg_Range2 = "y" in ax_Range
    ### Drow additional background curve
    if Func_Drow is not None:
        ax1 = Func_Drow(ax1)

    #########################################
    ### Define types of plots              ##
    ### and corresponding Color and Prefix ##
    #########################################
    if Masks_in is None:
        Masks = [pd.Series(True, index=Base_df.index)]
    else:
        Masks = Masks_in.copy()
    ##########
    Clist = Clist_in.copy()
    if SolidConnect_in is not None:
        SolidConnect = SolidConnect_in.copy()
    else:
        SolidConnect = [True] * len(Clist)
    ######
    Fcolor = Clist.copy()
    elw = [0.7] * len(Clist)
    zos = [ii + 1 for ii in range(len(Clist))]
    if prefixes_in is not None:
        prefixes = prefixes_in.copy()
    else:
        prefixes = [None] * len(Masks)
    #########
    if flg_MultiEffMode or flg_Suspecious:
        if flg_Suspecious:
            NewMasks = []
            #### New Mask (new plotting loop) with suspecious
            for mask_tem, pf_tem in zip(Masks, prefixes):
                NewMasks.append(mask_tem & Base_df[pf_tem + "IsSuspecious"])
            #### New Mask (new plotting loop) without suspecious
            for mask_tem, pf_tem in zip(Masks, prefixes):
                NewMasks.append(mask_tem & ~Base_df[pf_tem + "IsSuspecious"])
            ########
            # SolidConnect = [False] * len(prefixes) + SolidConnect
            SolidConnect = SolidConnect * 2  ### make it not dashed
            ##################
            alpha = 0.2
            NewClist = [(*cl[:3], cl[3] * alpha) for cl in Clist] + Clist
            Clist = NewClist.copy()
            elw = elw * 2
            zos = zos * 2
            Ftem = [(*cl[:3], cl[3] * alpha) for cl in Fcolor] + Fcolor
            Fcolor = Ftem.copy()
            ### belows are common
            prefixes = prefixes * 2
            Masks = NewMasks.copy()
        #######################
        if flg_MultiEffMode:
            MDCL = []
            NewMasks = []
            SolidConnect = []
            #### New Mask (new plotting loop) with suspecious
            for mask_tem, pf_tem in zip(Masks, prefixes):
                NewMasks.append(mask_tem & Base_df[pf_tem + "IsMultiEffMode"])
                SolidConnect.append(False)
            #### New Mask (new plotting loop) without suspecious
            for mask_tem, pf_tem in zip(Masks, prefixes):
                NewMasks.append(mask_tem & ~Base_df[pf_tem + "IsMultiEffMode"])
                SolidConnect.append(True)
            ##############
            SolidConnect = [False] * len(prefixes) + SolidConnect
            ###########################################################
            # Fcolor    = [ 'none' ]* len(Fcolor) + Fcolor
            Fcolor = [(1, 1, 1, 1)] * len(Fcolor) + Fcolor
            elw = [0.5] * len(elw) + elw
            zos = zos * 2
            Clist = Clist * 2
            prefixes = prefixes * 2
            Masks = NewMasks.copy()
        #####################
        ####################
    ####
    for ii, (Mask_ori, prefix) in enumerate(zip(Masks, prefixes)):
        if flg_Modes:
            Modes = ModeList(Base_df, prefix.rstrip("."))
        else:
            Modes = ["Global"]
        for mode in Modes:
            PN1, PE1 = naming(name1, prefix, mode, flg_Range1)
            PN2, PE2 = naming(name2, prefix, mode, flg_Range2)
            try:
                Mask_tem = Mask_ori & Base_df[PN1].notna() & Base_df[PN2].notna()
                if (~Mask_tem).all():
                    continue
            except ValueError as ve:
                print(ve)
                print(f"{PN1} or {PN2} not exists")                
                break
            ###################3
            ax1, MDC = DFPlotErrScat(
                fig,
                ax1,
                Base_df[Mask_tem],
                PN1,
                PN2,
                err1=PE1,
                err2=PE2,
                color=Clist[ii],
                elw=elw[ii],
                Fcolor=Fcolor[ii],
                zorder=zos[ii],
                flg_MultiDate=flg_MultiDate,
                ax_Range=ax_Range,
                PlaceSymbolLimit=PlaceSymbolLimit,
            )
            if flg_MultiDate:
                MDC = MDC.drop_duplicates(subset=["Object", "Date", "xx", "yy"])
                MDC["Solid"] = SolidConnect[ii]
                MDCL.append(MDC)
        ### loop modes
    #######################3
    if flg_MultiDate:
        seg_color = [
            mcolors.to_rgb(cn)[:3] + (1,)
            for cn in ["grey", "orange", "darkgreen", "magenta", "cyan", "pink"]
        ]
        suff = "MD_"
        MDC_df = pd.concat(MDCL, ignore_index=True)
        MDC_df = MDC_df.drop_duplicates(subset=["Object", "Date", "xx", "yy"])
        # print(MDC_df)
        for (obj,), gg in MDC_df.groupby(["Object"]):
            gg = gg.sort_values("Date")
            dates = gg["Date"].drop_duplicates().sort_values().tolist()
            points = []
            for dd in dates:
                gd = gg[gg["Date"] == dd]
                pts = list(zip(gd["xx"].to_numpy(), gd["yy"].to_numpy()))
                if len(pts) > 0:
                    points.append(pts)
            ######
            for ii in range(len(points) - 1):
                d1 = points[ii]
                d2 = points[ii + 1]
                if len(d1) > 1 or len(d2) > 1:
                    ls = ":"
                else:
                    ls = "-"
                ######
                for (x1, y1), (x2, y2) in product(d1, d2):
                    ax1.plot(
                        [x1, x2], [y1, y2], ls, color=seg_color[ii], lw=0.5, zorder=0
                    )
    #####################
    mpl.rcParams["savefig.bbox"] = "standard"
    fig.savefig(SavePath / f"{suff}{name1}-{name2}.pdf")
    # bbox_inches=None, dpi=300, pad_inches=0, transparent=False)
    plt.close()
    return


###############################


def naming(name, prefix, mode, flg_range):
    if name.endswith("_CA"):
        PN1 = name
        PE1 = "E" + name
    elif name.startswith("CA_"):  ## use CASPAR but modified
        PN1 = name
        PE1 = name + "_range"
    elif name.startswith(("Mass", "radius", "vff")):
        PN1 = name
        PE1 = name + "_err"
    elif flg_range:  ### In this order, DeRedLum comes to here even with 'Lum'
        PN1 = prefix + f"{mode}." + name
        PE1 = prefix + f"{mode}." + name + "_range"
    elif any([ww in name for ww in ["Ratio", "Lum"]]) and "DeRedLum" not in name:
        PN1 = name
        PE1 = "d" + name
    else:
        PN1 = prefix + f"{mode}." + name
        PE1 = prefix + f"{mode}.d" + name
    #####
    return PN1, PE1
    ######


def labeling(axt, name, flg_x, Force_log=False):
    flg_log = Force_log
    if flg_x:
        axis_xy = "x"
    else:
        axis_xy = "y"
    ##########
    if name == "Mass":
        axis_xy = ""
        label = r"Mass [$M_\mathrm{J}$]"
        vrange = [0, 500]
        if flg_log:
            vrange[0] = 10
    elif name == "radius":
        axis_xy = ""
        label = r"Radius [$R_\mathrm{J}$]"
        vrange = [0, 20]
        if flg_log:
            vrange[0] = 1
    elif name == "Area":
        label = r"Emitting Area [$R^2_\mathrm{J}$]"
        flg_log = True
        vrange = [1e-2, 1e4]
    ###################
    ##### solar normalization
    elif name == "Mass_Msun":
        axis_xy = ""
        label = r"Mass [$M_\odot$]"
        vrange = [0, 0.5]
        flg_log = True  ### for test
        if flg_log:
            vrange[0] = 1e-2
    elif name == "radius_Rsun":
        axis_xy = ""
        label = r"Radius [$R_\odot$]"
        vrange = [0, 2]
        flg_log = True  ## test
        if flg_log:
            vrange[0] = 0.1
    elif name == "Area_Rsun":
        label = r"Emitting Area [$R^2_\odot$]"
        flg_log = True
        vrange = [1e-4, 1e2]
    elif name == "Mdot_Msun":
        label = r"$\dot{M}$ [$M_\odot$ yr$^{-1}$]"
        vrange = [1e-11, 2e-6]
        flg_log = True
    ##############
    elif name == "vff":
        axis_xy = ""
        label = r"$v_\mathrm{ff}$ [km s$^{-1}$]"
        vrange = [150, 350]
    elif name == "v0":
        label = r"$v_\mathrm{0}$ [km s$^{-1}$]"
        vrange = [50, 200]
    elif name == "n0":
        label = r"$n_\mathrm{0}$ [log$_{10}$ cm$^{-3}$]"
        vrange = [9, 15]
    elif name == "vRat":
        label = r"$v_0/v_\mathrm{ff}$"
        vrange = [0, 1]
        if flg_log:
            vrange[0] = 0.1
    elif name == "RtRp":
        label = r"$R_\mathrm{t}/R_\mathrm{P}$"
        vrange = [0, 6]
        if flg_log:
            vrange[0] = 1
    elif name == "RtRobj":
        label = r"$R_\mathrm{t}/R_\mathrm{obj}$"
        vrange = [0, 6]
        if flg_log:
            vrange[0] = 1
    elif name == "RtRj":
        label = r"$R_\mathrm{t}/R_\mathrm{J}$"
        vrange = [0, 40]
        if flg_log:
            vrange[0] = 1
    elif name == "RtRsun":
        label = r"$R_\mathrm{t}/R_\odot$"
        vrange = [0, 4]
        if flg_log:
            vrange[0] = 1
    elif name == "ff":
        label = r"$f_\mathrm{f}$"
        vrange = [1e-6, 1]
        flg_log = True
    elif name == "ff_percent":
        label = r"$f_\mathrm{f}$ [%]"
        vrange = [1e-4, 2e2]
        flg_log = True
    elif name == "Av":
        label = r"$A_\mathrm{V}$ [mag]"
        vrange = [0, 10]
        # if flg_log: vrange[0] = 0.1
    elif name == "Rv":
        label = r"$R_\mathrm{V}$"
        vrange = [2.3, 5.6]
    elif name == "Mdot":
        label = r"$\dot{M}$ [$M_\mathrm{J}$ yr$^{-1}$]"
        vrange = [1e-8, 1e-4]
        flg_log = True
    elif name == "Lacc":
        label = r"$L_\mathrm{acc}$ [$L_\odot$]"
        vrange = [1e-5, 1e2]
        flg_log = True
    elif name == "Bstar" or name == "BFlux":
        label = r"$B_\mathrm{*}$ [G]"
        flg_log = True
        vrange = [10, 1e4]
    elif name.startswith("B_"):
        normal = name.split("_")[-1]
        if normal == "Mobj":
            label = r"$B_\mathrm{*}/M^{1/4}$"
            vrange = [10, 1e6]
        elif normal == "Mdot":
            label = r"$B_\mathrm{*}/\dot{M}^{1/2}$"
            vrange = [1e5, 1e8]
        elif normal == "RtRo":
            label = r"$B_\mathrm{*}/(R_\mathrm{t}/R_*)^{7/4}$"
            vrange = [1e1, 1e4]
        elif normal == "Robj":
            label = r"$B_\mathrm{*}/(R_*)^{-5/4}$"
            vrange = [1e1, 1e4]
        else:
            raise ValueError(f"PSFE.labeling: Check name {name}")
        ###########
        flg_log = True
    ############## CASPAR
    elif name == "LACC_CA":
        label = r"$\log_{10} L_\mathrm{acc, CASPAR}$ [$L_\odot$]"
        vrange = [-5, 2]
        # flg_log = True
    elif name == "MDOT_CA":
        axis_xy = ""
        label = r"$\dot{M}_\mathrm{CASPAR}$ [$M_\odot\,\mathrm{yr}^{-1}$]"
        vrange = [1e-11, 2e-6]
        flg_log = True
    elif name == "CA_LaccLin":
        label = r"$L_\mathrm{acc, CASPAR}$ [$L_\odot$]"
        vrange = [1e-5, 1e2]
        flg_log = True
    elif name == "LaccRat":
        label = r"$L_\mathrm{acc}\,/\,L_\mathrm{acc, CASPAR}$"
        vrange = [1e-3, 2]
        flg_log = True
    #### de-redded luminosity in each mode
    elif "DeRedLum_Obs" in name:
        line = name.split("_")[-1]
        # label   = r'$L_\mathrm{Obs}/L_\odot$' + f' in {line}'
        label = r"$L_\mathrm{Obs,DR," + line + r"}/L_\odot$"
        vrange = [1e-9, 1e-3]
        flg_log = True
    elif "DeRedLum_BC" in name:
        line = name.split("_")[-1]
        label = r"$L_\mathrm{BC,DR," + line + r"}/L_\odot$"
        vrange = [1e-9, 1e-3]
        flg_log = True
    elif "DeRedLum_Fit" in name:
        line = name.split("_")[-1]
        label = r"$L_\mathrm{Shock,DR," + line + r"}/L_\odot$"
        vrange = [1e-9, 1e-3]
        flg_log = True
    #### Now, in each mode and has range
    #### ratio to Lacc
    elif "_Lacc_" in name:
        line = name.split("_")[-1]
        Ltype = name.split("_")[0]
        vrange = [1e-5, 1]
        flg_log = True
        if Ltype == "LObs":
            label = r"$L_\mathrm{Obs," + line + r"}/L_\mathrm{acc}$"
        elif Ltype == "LRes":
            label = r"$L_\mathrm{Res," + line + r"}/L_\mathrm{acc}$"
        elif Ltype == "LBC":
            label = r"$L_\mathrm{BC," + line + r"}/L_\mathrm{acc}$"
            vrange = [1e-5, 1e-1]
        elif Ltype == "LFit":
            label = r"$L_\mathrm{Shock," + line + r"}/L_\mathrm{acc}$"
            vrange = [1e-5, 1e-1]
        else:
            raise ValueError(f"PSFE.labeling: invalid name: {name}")
    elif "Lum_Fit" in name:
        line = name.split("_")[-1]
        label = r"$L_\mathrm{Shock," + line + r"}/L_\odot$"
        # label   = r'$L_\mathrm{fit}/L_\odot$' + f' in {line}'
        vrange = [1e-9, 1e-3]
        flg_log = True
    elif "LRat" in name:
        line = name.split("_")[-1]
        label = r"$L_\mathrm{Shock," + line + r"} /" + r" L_\mathrm{Obs," + line + r"}$"
        # label   = r'$L_\mathrm{fit}/L_\odot$' + f' in {line}'
        vrange = [1e-3, 10]
        flg_log = True
    elif "DeRedLRes" in name:
        line = name.split("_")[-1]
        label = r"$L_\mathrm{Res,DR," + line + r"}/L_\odot$"
        vrange = [1e-9, 1e-3]
        flg_log = True
    elif "LRes" in name:
        line = name.split("_")[-1]
        label = r"$L_\mathrm{Res," + line + r"}/L_\odot$"
        vrange = [1e-9, 1e-3]
        flg_log = True
    ###### luminosity value
    elif "Lum_Obs" in name:
        axis_xy = ""
        line = name.split("_")[-1]
        # label   = r'$L_\mathrm{Obs}/L_\odot$' + f' in {line}'
        label = r"$L_\mathrm{Obs," + line + r"}/L_\odot$"
        vrange = [1e-9, 1e-3]
        flg_log = True
    elif "Lum_BC" in name:
        axis_xy = ""
        line = name.split("_")[-1]
        label = r"$L_\mathrm{BC," + line + r"}/L_\odot$"
        vrange = [1e-9, 1e-3]
        flg_log = True
    ################# luminosity ratio
    elif "Ratio_BC" in name:
        axis_xy = ""
        line = name.split("_")[-1]
        label = r"$L_\mathrm{BC}/L_\mathrm{Obs}$" + f" in {line}"
        vrange = [5e-2, 1]
        flg_log = True
    elif "Ratio_FitObs" in name:
        axis_xy = ""
        line = name.split("_")[-1]
        label = r"$L_\mathrm{Shock}/L_\mathrm{Obs}$" + f" in {line}"
        vrange = [1e-2, 2]
        flg_log = True
    ##########
    ## specific
    # elif 'Lum' in name and name.count('BC')==2:
    elif BCRmatch := re.fullmatch(r".?Lum_(.{2})BC_(.{2})BC", name):
        l1 = BCRmatch.group(1)
        l2 = BCRmatch.group(2)
        axis_xy = ""
        line = name.split("_")[-1]
        label = r"$L_\mathrm{BC," + l1 + r"}/L_\mathrm{BC," + l2 + r"}$"
        vrange = [1e-2, 2]
        flg_log = True
    else:
        raise TypeError(f"PLOT_SpecFitEstiamtes.labeling: {name} is not defined")
    ###########
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    if flg_x:
        axt.set_xlabel(label)
        if vrange is not None:
            axt.set_xlim(vrange)
        if flg_log:
            axt.set_xscale("log")
            axt.xaxis.set_major_locator(
                mticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
            )
            axt.xaxis.set_minor_locator(
                mticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100)
            )
            axt.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
            axt.xaxis.set_minor_formatter(mticker.NullFormatter())
        else:  ### linear
            fmt.set_powerlimits((-3, 3))
            axt.xaxis.set_major_formatter(fmt)
    else:
        axt.set_ylabel(label)
        if vrange is not None:
            axt.set_ylim(vrange)
        if flg_log:
            axt.set_yscale("log")
            axt.yaxis.set_major_locator(
                mticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
            )
            axt.yaxis.set_minor_locator(
                mticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100)
            )
            axt.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
            axt.yaxis.set_minor_formatter(mticker.NullFormatter())
        else:  ###linear
            fmt.set_powerlimits((-3, 3))
            axt.yaxis.set_major_formatter(fmt)
    ##############
    return axt, label, vrange, flg_log, axis_xy


#############


def ModeList(df, ModePrefix_in, ModeIndicator="Mode", SkipCheck="v0"):
    if ModePrefix_in[-1] == ".":
        ModePrefix = ModePrefix_in[:-1]
    else:
        ModePrefix = ModePrefix_in
    #################
    LevKeys = (
        df.columns.to_series().str.split(".", expand=True).reindex(columns=range(4))
    )
    PreF = LevKeys.iloc[:, 0]
    Modes = LevKeys.iloc[:, 1]
    Modes = (
        Modes[PreF.eq(ModePrefix)]
        .dropna()
        .astype(str)
        .loc[lambda xx: xx.str.startswith(ModeIndicator)]
        .drop_duplicates()
        .sort_values(key=lambda xx: xx.str[len(ModeIndicator) :].astype(int))
        .tolist()
    )
    if SkipCheck is not None:
        Modes_use = Modes.copy()
        Modes = []
        for mode in Modes_use:
            PreMode = ModePrefix + "." + mode + "."
            if (PreMode + SkipCheck) not in df.columns:
                continue
            Modes.append(mode)
        #########
    ##################
    return Modes


def DFPlotErrScat(
    fig,
    ax,
    df,
    idx1,
    idx2,
    flg_MultiDate=False,
    err1=None,
    err2=None,
    color="black",
    elw=0.8,
    Fcolor=None,
    zorder=1,
    ax_Range="",
    PlaceSymbolLimit=None,
):
    #######
    if Fcolor is None:
        Fcolor = color
    ####
    flg_PSLx = False
    flg_PSLy = False
    if PlaceSymbolLimit is not None:
        ix = PlaceSymbolLimit.find("x")
        if ix > -1:
            flg_PSLx = True
            PSLx = int(PlaceSymbolLimit[ix + 1])
        ###########
        iy = PlaceSymbolLimit.find("y")
        if iy > -1:
            flg_PSLy = True
            PSLy = int(PlaceSymbolLimit[iy + 1])
    ############3
    capsize = 4
    axt = ax
    xmin, xmax = axt.get_xlim()
    ymin, ymax = axt.get_ylim()

    def _PlotRangeBar(df_masked, flg_MultiDate):
        ylolims = False
        xlolims = False
        yuplims = False
        xuplims = False
        ################
        flg_ylog = ax.get_yscale() == "log"
        y0, y1 = axt.get_ylim()
        fig_yw_pt = fig.get_size_inches()[1] * 72
        if flg_ylog:
            dy = 10 ** (capsize / fig_yw_pt * abs(np.log10(y1) - np.log10(y0)))
        else:
            dy = capsize / fig_yw_pt * (y1 - y0)
        ####################
        yy = df_masked[idx2]
        #############################
        flg_xlog = ax.get_xscale() == "log"
        x0, x1 = axt.get_xlim()
        fig_xw_pt = fig.get_size_inches()[0] * 72
        if flg_xlog:
            dx = 10 ** (capsize / fig_xw_pt * abs(np.log10(x1) - np.log10(x0)))
        else:
            dx = capsize / fig_xw_pt * (x1 - x0)
        ####################
        xx = df_masked[idx1]

        ##############
        def IntoRange(val, vmin, vmax):
            return np.maximum(vmin, np.minimum(vmax, val))

        xx = IntoRange(xx, xmin, xmax)
        yy = IntoRange(yy, ymin, ymax)
        ############################################################
        ### range. This needs to be after determining yy and xx.
        ############################################################
        if "x" in ax_Range:
            xerr = None
            XR = np.asarray(df_masked[err1].to_list())
            if flg_PSLx:
                xori = xx.copy()
                yori = yy.copy()
                xx = IntoRange(XR[:, PSLx], xmin, xmax)
            #### put symbol at lower limit and arrow
            xlolims = pd.isna(XR[:, 1])  ## upper is None, namely only low
            xx[xlolims] = XR[xlolims, 0]
            x_lo = xx - IntoRange(XR[:, 0], xmin, xmax)
            ####
            xuplims = pd.isna(XR[:, 0])
            xx[xuplims] = XR[xuplims, 1]
            x_up = IntoRange(XR[:, 1], xmin, xmax) - xx
            ####
            x_up[xlolims] = xx[xlolims] * dx  ### random value
            x_lo[xuplims] = xx[xuplims] * dx  ### random value
            xerr = [x_lo.to_numpy(), x_up.to_numpy()]
        else:  ## standard deviation is given
            xerr = df_masked[err1].to_numpy()
        if "y" in ax_Range:
            yerr = None
            YR = np.asarray(df_masked[err2].to_list())
            if flg_PSLy:
                xori = xx.copy()
                yori = yy.copy()
                yy = IntoRange(YR[:, PSLy], ymin, ymax)
            ################
            ylolims = pd.isna(YR[:, 1])  ## upper is None, namely only low
            yy[ylolims] = YR[ylolims, 0]
            y_lo = yy - IntoRange(YR[:, 0], ymin, ymax)
            ####
            yuplims = pd.isna(YR[:, 0])
            yy[yuplims] = YR[yuplims, 1]
            y_up = IntoRange(YR[:, 1], ymin, ymax) - yy
            ####
            y_up[ylolims] = yy[ylolims] * dy  ### random value
            y_lo[yuplims] = yy[yuplims] * dy  ### random value
            yerr = [y_lo.to_numpy(), y_up.to_numpy()]
        else:  ## standard deviation is given
            yerr = df_masked[err2].to_numpy()
        ##########
        if isinstance(color, str):
            TransEcolor = list(mcolors.to_rgb(color))
        elif isinstance(color, tuple):
            TransEcolor = color
        elif isinstance(color, list):
            TransEcolor = color.copy()
        ###########
        tem = 0.5
        if len(TransEcolor) == 3:
            TransEcolor.append(tem)
        else:
            TransEcolor = (*TransEcolor[:3], TransEcolor[3] * tem)
        axt.errorbar(
            xx.to_numpy(),
            yy.to_numpy(),
            xerr=xerr,
            yerr=yerr,
            lolims=ylolims,
            uplims=yuplims,
            xlolims=xlolims,
            xuplims=xuplims,
            capsize=capsize,
            fmt="o",
            markersize=5,
            markeredgecolor=color,
            markerfacecolor=Fcolor,
            #######
            elinewidth=elw,
            markeredgewidth=elw,
            capthick=elw,
            ecolor=TransEcolor,
            linestyle="none",
            zorder=zorder,
        )
        if flg_PSLx or flg_PSLy:
            axt.scatter(
                xori.to_numpy(),
                yori.to_numpy(),
                color=color,
                marker="s",
                s=1,
            )
        if flg_MultiDate:
            MMD = df_masked["IsMultiDate"]
            MultiDateCol = pd.DataFrame(
                {
                    "Object": df_masked[MMD].index.get_level_values(0),
                    "Date": df_masked[MMD].index.get_level_values(1),
                    "xx": xx[MMD],
                    "yy": yy[MMD],
                }
            )
            return MultiDateCol
        else:
            return None

    ###################
    MDC = _PlotRangeBar(df, flg_MultiDate)
    #############
    return axt, MDC
