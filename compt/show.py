# v 10
import scipy.stats as stats
# get ANOVA table as R like output
import statsmodels.api as sm
from statsmodels.formula.api import ols
# from pandas.table.plotting import table # EDIT: see deprecation warnings below
from pandas.plotting import table
# import dataframe_image as dfi

from distutils.dir_util import copy_tree
import subprocess
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib
from curses import wrapper
from tabulate import tabulate
import click
import warnings
import itertools
import numpy as np
import statistics as stat
from glob import glob
import six
import debugpy
import os, shutil
import re
import seaborn as sns
from pathlib import Path
import pandas as pd
from attempt.win import *
from mylogs import * 
from datetime import datetime, timedelta
import time
import json
from tqdm import tqdm
# from comet.utils.myutils import *
from attempt.utils.utils import combine_x,combine_y,add_margin
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sklearn
import sklearn.metrics
import attempt.metrics.metrics as mets


warnings.simplefilter(action='ignore', category=Warning)
file_id = "name"

table_mid_template = """
    \\begin{{table*}}
        \centering
        \\begin{{tabular}}{{{}}}
        \hline
        {}
        \end{{tabular}}
        \caption{{{}}}
        \label{{{}}}
    \end{{table*}}
    """
table_fp_template = """
    \\begin{{table*}}
        \\begin{{adjustbox}}{{width=1\\textwidth}}
        \\begin{{tabular}}{{{}}}
        \hline
        {}
        \end{{tabular}}
        \caption{{{}}}
        \label{{{}}}
        \end{{adjustbox}}
    \end{{table*}}
    """
table_env_template = """
    \\begin{{table*}}[h!]
        minipage
        \caption{{{}}}
        \label{{{}}}
    \end{{table*}}
    """
table_hm_template = """\\begin{{minipage}}{{.4\\linewidth}}
    \centering
    \label{{{}}}
    \pgfplotstabletypeset[
    color cells={{min={},max={}}},
    col sep=&,	% specify the column separation character
    row sep=\\\\,	% specify the row separation character
    columns/N/.style={{reset styles,string type}},
    /pgfplots/colormap={{whiteblue}}{{rgb255(0cm)=(255,255,255); rgb255(1cm)=(0,200,200)}},
    ]{{{}}}
    \end{{minipage}}"""
def latex_table(rep, rname, mdf, all_exps, sel_col, category, caption=""):
    maxval = {}
    for ii, exp in enumerate(all_exps): 
        exp = exp.replace("_","-")
        exp = _exp = exp.split("-")[0]
        if not sel_col in rep:
            continue
        if not exp in rep[sel_col]:
            continue
        for rel in mdf['prefix'].unique(): 
            if not rel in rep[sel_col][exp]:
                continue
            val = rep[sel_col][exp][rel]
            if type(val) == list:
                assert val, rel + "|"+ sel_col + "|"+ exp
                val = stat.mean(val)
            if not rel in maxval or val > maxval[rel]:
                maxval[rel] = val

    table_cont2=""
    table_cont2 += "method & "
    head2 = "|r|"
    for rel in mdf['prefix'].unique(): 
        table_cont2 += "\\textbf{" + rel + "} &"
        head2 += "r|"
    table_cont2 += " avg. " 
    head2 += "r|"
    table_cont2 = table_cont2.strip("&")
    table_cont2 += "\\\\\n"
    table_cont2 += "\\hline\n"
    for ii, exp in enumerate(all_exps): 
        exp = exp.replace("_","-")
        exp = _exp = exp.split("-")[0]
        if not sel_col in rep:
            continue
        if not exp in rep[sel_col]:
            continue

        table_cont2 += " \hyperref[fig:" + category + _exp + "]{" + _exp + "} &"
        for rel in mdf['prefix'].unique(): 
            if not rel in rep[sel_col][exp]:
                continue
            val = rep[sel_col][exp][rel]
            if type(val) == list:
                val = stat.mean(val)
            if val == maxval[rel]:
                table_cont2 += "\\textcolor{teal}{" +  f" $ {val:.1f} $ " + "} &"
            else:
                table_cont2 += f" $ {val:.1f} $ &"
        if "avg" in rep[sel_col][exp]:
            avg = rep[sel_col][exp]["avg"]
            if type(avg) == list and avg:
                avg = stat.mean(avg)
            if avg:
                avg = "{:.1f}".format(avg)
            table_cont2 += f" $ \\textcolor{{blue}}{{{avg}}} $ &"
        table_cont2 = table_cont2.strip("&")
        table_cont2 += "\\\\\n"
    table_cont2 += "\\hline \n"
    for head, cont in zip([head2],
            [table_cont2]):
        label = "table:" + rname + sel_col.replace("_","-") 
        capt = caption
        if not capt:
           capt = category + " \hyperref[table:show]{ Main Table } | " + label
        table = """
            \\begin{{table*}}[h!]
                \centering
                \\begin{{tabular}}{{{}}}
                \hline
                {}
                \end{{tabular}}
                \caption{{{}}}
                \label{{{}}}
            \end{{table*}}
            """
        table = table.format(head, cont, capt, label)
    return table


def create_label(row):
    label = ''
    if row['add_target']:
        label += 'A_'
    if row['use_source_prompts']:
        label += 'S'
        if row['load_source_prompts']:
            label += 'I'
        if row['learn_source_prompts']:
            label += 'L'
    if row['use_private_prompts']:
        label += 'P'
        if row['load_private_prompts']:
            label += 'I'
    return label

def save_df_as_image(df, path):
    # Set background to white
    norm = matplotlib.colors.Normalize(-1,1)
    colors = [[norm(-1.0), "white"],
            [norm( 1.0), "white"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    # Make plot
    plot = sns.heatmap(df, annot=True, cmap=cmap, cbar=False)
    fig = plot.get_figure()
    fig.savefig(path)

def clean_file(f):
    with open(f, 'r') as f1:
        _dict = json.load(f1)
        for c in ["tag","log_var","main_vars","full_tag"]:
            if c in _dict:
                del _dict[c]
    with open(f, 'w') as f2:
        json.dump(_dict, f2, indent=3)


def plot_bar(rep, folder, sel_col):
    methods = list(rep[sel_col + "@m_score"].keys()) 
    bar_width = 0.25
    r = np.arange(9)
    ii = -1
    color = ["red","green","blue"]
    for key in [train_num + "@m_score","500@m_score"]: 
        ii += 1
        column = [float(rep[key][met]) for met in methods]
        r = [x + bar_width for x in r]
        plt.bar(r, column, color=color[ii], width=bar_width, edgecolor='white', label=key)

# Add xticks on the middle of the group bars
    plt.xlabel('Methods', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(column))], methods)

# Add y axis label and title
    plt.ylabel('Performance', fontweight='bold')
    plt.title('Performance of Methods')

# Add legend and show the plot
    plt.legend()
    pname = os.path.join(folder, "bar.png")
    plt.savefig(pname)
    return pname

def score_colors(df,row,col, default=None):
    max_val = df[col].max()
    if df.iloc[row][col] == max_val:
        return 136
    return 247

def pivot_colors(df,row,col, default=None):
    rel_col = "n-" + col
    max_val = df[col].max()
    if rel_col in df:
        if df.iloc[row][rel_col] <= 1:
            return 91
        if df.iloc[row][col] == max_val:
            return 136
    return default if default is not None else TEXT_COLOR 

def index_colors(df,row,col, default=None):
    index = df.iloc[row][col]
    return index

def cat_colors(df,row,col, default=None):
    cat = df.iloc[row][col]
    cat = str(cat)
    if "research" in cat or "question" in cat:
        return WARNING_COLOR
    elif "done" in cat:
        return MSG_COLOR
    return default

def time_colors(df,row,col, default=None):
    rel_col = "time" 
    last_hour = datetime.now() - timedelta(hours = 1)
    last_10_minutes = datetime.now() - timedelta(minutes = 10)
    if rel_col in df:
        time = str(datetime.now().year) + "-" + df.iloc[row][rel_col]
        time = datetime.strptime(time, '%Y-%m-%d-%H:%M')
        if time > last_10_minutes:
            return WARNING_COLOR
        elif time > last_hour:
            return 81
    return default if default is not None else TEXT_COLOR 


def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    sd = superitems(data)
    fname = Path(path).stem
    if fname == "results":
        main_df = pd.DataFrame(sd, columns=["exp","model","lang", "wrap","frozen","epochs","stype", "date", "dir", "score"])
    else:
        main_df = pd.DataFrame(sd, columns=["tid","exp","model","lang", "wrap","frozen","epochs","date", "field", "text"])

    out = f"{fname}.tsv"
    df = main_df.pivot(index=list(main_df.columns[~main_df.columns.isin(['field', 'text'])]), columns='field').reset_index()

    #df.columns = list(map("".join, df.columns))
    df.columns = [('_'.join(str(s).strip() for s in col if s)).replace("text_","") for col in df.columns]
    df.to_csv(path.replace("json", "tsv"), sep="\t", index = False)
    return df

def remove_uniques(df, sel_cols, tag_cols = [], keep_cols = []):
    _info_cols = []
    _tag_cols = tag_cols
    _sel_cols = []
    _df = df.nunique()
    items = {k:c for k,c in _df.items()}
    df.columns = df.columns.get_level_values(0)
    for c in sel_cols:
        if not c in items:
            continue
        _count = items[c]
        if c in keep_cols:
            _sel_cols.append(c)
        elif _count > 1: 
           _sel_cols.append(c)
        else:
           _info_cols.append(c) 
    if _sel_cols:
        for _col in tag_cols:
            if not _col in _sel_cols:
                _sel_cols.append(_col)

    _sel_cols = list(dict.fromkeys(_sel_cols))
    return _sel_cols, _info_cols, tag_cols

def list_dfs(df, main_df, s_rows, FID):
    dfs_items = [] 
    dfs = []
    ii = 0
    dfs_val = {}
    for s_row in s_rows:
        exp=df.iloc[s_row]["fid"]
        prefix=df.iloc[s_row]["prefix"]
        dfs_val["exp" + str(ii)] = exp
        mlog.info("%s == %s", FID, exp)
        cond = f"(main_df['{FID}'] == '{exp}') & (main_df['prefix'] == '{prefix}')"
        tdf = main_df[(main_df[FID] == exp) & (main_df['prefix'] == prefix)]
        tdf = tdf[["pred_text1", "exp_name", "id","hscore", "out_score", "bert_score","query", "resp", "template", "rouge_score", "fid","prefix", "input_text","target_text", "sel"]]
        tdf = tdf.sort_values(by="rouge_score", ascending=False)
        sort = "rouge_score"
        dfs.append(tdf)
    return dfs

def find_common(df, main_df, on_col_list, s_rows, FID, char, tag_cols):
    dfs_items = [] 
    dfs = []
    ii = 0
    dfs_val = {}
    for s_row in s_rows:
        exp=df.iloc[s_row]["fid"]
        prefix=df.iloc[s_row]["prefix"]
        dfs_val["exp" + str(ii)] = exp
        mlog.info("%s == %s", FID, exp)
        cond = f"(main_df['{FID}'] == '{exp}') & (main_df['prefix'] == '{prefix}')"
        tdf = main_df[(main_df[FID] == exp) & (main_df['prefix'] == prefix)]
        _cols = tag_cols + ["pred_text1", "top_pred", "top", "exp_name", "id","hscore", "bert_score", "out_score","query", "resp", "template", "rouge_score", "fid","prefix", "input_text","target_text", "sel"]
        _cols = list(dict.fromkeys(_cols))
        _cols2 = [] 
        for col in _cols:
            if col in main_df:
                _cols2.append(col)
        tdf = tdf[_cols2]
        tdf = tdf.sort_values(by="rouge_score", ascending=False)
        sort = "rouge_score"
        if len(tdf) > 1:
            tdf = tdf.groupby(on_col_list).first()
            tdf = tdf.reset_index()
            for on_col in on_col_list:
                tdf[on_col] = tdf[on_col].astype(str).str.strip()
            #tdf = tdf.set_index(on_col_list)
        dfs.append(tdf) #.copy())
        ii += 1
    if char == "i":
        return df, exp, dfs
    if ii > 1:
        intersect = reduce(lambda  left,right: pd.merge(left,right,on=on_col_list,
                                    how='inner'), dfs)
        if char == "k":
            union = reduce(lambda  left,right: pd.merge(left,right,on=on_col_list,
                                    how='outer'), dfs)
            dfs_val["union"] = str(len(union))
            dfs_val["int"] = str(len(intersect))
            dfs_items.append(dfs_val)
            df = pd.DataFrame(dfs_items)
        else:
            df = intersect
    else:
       df = tdf
       df["sum_fid"] = df["id"].sum()
    return df, exp, dfs

def calc_metrics(main_df):
    infos = []
    all_exps = main_df['eid'].unique()
    for exp in all_exps:
        for task in main_df["prefix"].unique():
            cond = ((main_df['eid'] == exp) & (main_df["prefix"] == task))
            tdf = main_df[cond]
            preds = tdf["pred_text1"]
            preds = preds.fillna(0)
            if len(preds) == 0:
                continue
            golds = tdf["target_text"]
            task_metric = mets.TASK_TO_METRICS[task] if task in mets.TASK_TO_METRICS else ["rouge"]
            metrics_list = []
            for mstr in task_metric:
                metric = getattr(mets, mstr)
                met = metric(preds, golds)
                metrics_list.append(met)
            if met: 
                v = list(met.values())[0]
                main_df.loc[cond, "m_score"] = round(float(v),1)
            for met in metrics_list:
                for k,v in met.items():
                    infos.append(exp + ":" + task + ":" + str(k) + ":" + str(v))
                    infos.append("---------------------------------------------")
    return infos


class MyDF:
    context = ""
    df = None
    sel_row = 0
    sel_rows = []
    sel_cols = []
    selected_cols = []
    cur_row = 0
    cur_col = 0
    left = 0
    info_cols = []
    sort = ""
    group_col = ""
    def __init__(self, df, context, sel_cols, cur_col,info_cols, 
            sel_rows, sel_row, left, group_col, selected_cols, sort, **kwargs):
        self.df = df
        self.context = context
        self.sel_cols = sel_cols
        self.cur_col = cur_col
        self.info_cols = info_cols
        self.sel_rows = sel_rows
        self.sel_row = sel_row
        self.left = left
        self.group_col = group_col
        self.selected_cols = selected_cols
        self.sort = sort


def show_df(df):
    global dfname, hotkey, global_cmd

    hk = hotkey
    cmd = global_cmd 
    sel_row = 0
    cur_col = 0
    cur_row = 0
    ROWS, COLS = std.getmaxyx()
    ch = 1
    left = 0
    max_row, max_col= text_win.getmaxyx()
    width = 15
    top = 10
    height = 10
    cond = ""
    sort = ""
    asc = False
    info_cols = load_obj("info_cols", dfname, []) 
    info_cols_back = []
    sel_vals = []
    col_widths = load_obj("widths", "")
    def refresh():
        text_win.refresh(0, left, 0, 0, ROWS-1, COLS-2)
    def fill_text_win(rows):
        text_win.erase()
        for row in rows:
            mprint(row, text_win)
        refresh()

    def save_df(df): 
        return
        s_rows = range(len(df))
        show_msg("Saving ...")
        for s_row in s_rows:
            exp=df.iloc[s_row]["fid"]
            tdf = main_df[main_df["fid"] == exp]
            spath = tdf.iloc[0]["path"]
            tdf.to_csv(spath, sep="\t", index=False)


    if not col_widths:
        col_widths = {"query":50, "model":30, "pred_text1":30, "epochs":30, "date":30, "rouge_score":7, "bert_score":7, "out_score":7, "input_text":50}

    df['id']=df.index
    df = df.reset_index(drop=True)
    if not "tag" in df:
        df["tag"] = np.NaN 

    main_vars = []
    if "main_vars" in df:
        mvars = []
        for var in df["main_vars"].unique():
            dvar = json.loads(var)
            lvar = list(dvar.keys())
            mvars.extend([e for e in lvar 
                    if not e in ['max_train_samples', 'source_prompts',
                        'task_name', 'num_train_epochs']])
        main_vars = list(dict.fromkeys(mvars))

    #if not "word_score" in df:
    #    df['word_score'] = df['pred_text1'].str.split().str.len()

    if not "hscore" in df:
        df["hscore"] = np.NaN 

    if not "pid" in df:
        df["pid"] = 0

    #if not "l1_decoder" in df:
    #    df["l1_decoder"] ="" 
    #    df["l1_encoder"] ="" 
    #    df["cossim_decoder"] ="" 
    #    df["cossim_encoder"] ="" 

    if "gen_route_methods" in df:
        df["gen_norm_method"] = df["gen_route_methods"]
        df["norm_method"] = df["apply_softmax_to"]

    if "gen_norm_methods" in df:
        df["gen_norm_method"] = df["gen_norm_methods"]

    if not "query" in df:
        df["query"] = df["input_text"]
    if not "learning_rate" in df:
        df["learning_rate"] = 1

    if not "prefixed" in df:
        df["prefixed"] = False

    if not "sel" in df:
       df["sel"] = False

    if not "template" in df:
       df["template"] = ""

    if not "bert_score" in df:
       df["bert_score"] = 0

    #if "fid" in df:
    #    df = df.rename(columns={"fid":"expid"})

    if "input_text" in df:
        df['input_text'] = df['input_text'].str.replace('##','')
        df['input_text'] = df['input_text'].str.split('>>').str[0]
        df['input_text'] = df['input_text'].str.strip()

    if not "m_score" in df:
        calc_metrics(df)

    main_df = df
    edit_col = ""
    count_col = ""
    extra = {"filter":[], "inp":""}
    save_obj(dfname, "dfname", "")
    sel_cols = list(df.columns)
    for col in df.columns:
        if col.endswith("score"):
            df[col] = pd.to_numeric(df[col])
    fav_path = os.path.join(base_dir, dfname + "_fav.tsv")
    if Path(fav_path).exists():
        fav_df = pd.read_table(fav_path)
    else:
        fav_df = pd.DataFrame(columns = df.columns)
    sel_path = os.path.join(home, "atomic2020", "new_test.tsv")
    if Path(sel_path).exists():
        sel_df = pd.read_table(sel_path)
        if not "sel" in sel_df:
            sel_df["sel"] = False
    else:
        sel_df = pd.DataFrame(columns = ["prefix","input_text","target_text", "sel"])
        sel_df.to_csv(sel_path, sep="\t", index=False)

    back = []
    cur_df = None
    context = "main"

    def backit(df, sel_cols, cur_df = None):
        if len(sel_cols) < 2:
            mbeep()
            return
        if not cur_df:
            cur_df = MyDF(df, context, sel_cols, cur_col,info_cols, 
                sel_rows, sel_row, left, group_col, selected_cols, sort)
        back.append(cur_df)
        general_keys["b"] = "back"

    filter_df = main_df
    tag_cols = []
    if "taginfo" in df:
        tags = df.loc[0, "ftag"]
        tags = tags.replace("'", "\"")
        tags = json.loads(tags)
        tag_cols = list(tags.keys())
    if "expid" in tag_cols:
        tag_cols.remove("expid")
    if "mask_type" in df:
        df["cur_masking"] = (df["mask_type"].str.split("-").str[1] + "-" 
                + df["mask_type"].str.split("-").str[2]) 
    if "exp_name" in df:
        df["expid"] = df["exp_name"].str.split("-").str[1]
        df["expname"] = df["exp_name"].str.split("-").str[1]
    if False: #"expid" in df:
        df["fexpid"] = df["expid"]
        df["expname"] = df["expid"].str.split("-").str[0]
        df["expname"] = df["expname"].str.split("_").str[0]
        df["expid"] = df["expid"].str.split("-").str[1]
        df["expid"] = df["expid"].str.split(".").str[0]

    #df.loc[df.expid == 'P2-1', 'expid'] = "PI" 
    #tag_cols.insert(1, "expid")
    #if "m_score" in df:
    #    df["m_score"] = np.where((df['m_score']<=0), 0.50, df['m_score'])

    orig_tag_cols = tag_cols.copy()
    src_path = ""
    if "src_path" in df:
        src_path = df.loc[0, "src_path"]
        if not src_path.startswith("/"):
            src_path = os.path.join(home, src_path)
    if "pred_text1" in df:
        br_col = df.loc[: , "bert_score":"rouge_score"]
        df['nr_score'] = df['rouge_score']
        df['nr_score'] = np.where((df['bert_score'] > 0.3) & (df['nr_score'] < 0.1), df['bert_score'], df['rouge_score'])

    #wwwwwwwwww
    colors = ['blue','teal','orange', 'red', 'purple', 'brown', 'pink','gray','olive','cyan']
    context_map = {"g":"main", "G":"main", "X":"view", "r":"main"}
    general_keys = {"l":"latex df"}
    shortkeys = {"main":{"r":"pivot table"}}
    ax = None
    if "Z" in hotkey:
        df["m_score"] = df["rouge_score"]
    context = dfname
    font = ImageFont.truetype("/usr/share/vlc/skins2/fonts/FreeSans.ttf", 30)
    seq = ""
    reset = False
    prev_idea = ""
    pcols = [] #pivot unique cols
    cond_colors = {} # a dictionary of functions
    back_sel_cols = []
    all_sel_cols = []
    main_sel_cols = []
    search = ""
    si = 0
    mode = "main"
    sort = "rouge_score"
    on_col_list = []
    keep_cols = []
    unique_cols = []
    group_sel_cols = []
    group_df = None
    pivot_df = None
    rep_cols = []
    index_cols = []
    dfs = []
    pivot_cols = ['prefix']
    experiment_images = {}

    all_cols = {}
    file_dir = Path(__file__).parent
    #sel_cols =  load_obj("sel_cols", context, [])
    #info_cols = load_obj("info_cols", context, [])
    with open(os.path.join(file_dir, 'cols.json'),'r') as f:
        all_cols = json.load(f)

    if 'sel_cols' in all_cols:
        sel_cols = all_cols['sel_cols'] 
        info_cols = all_cols['info_cols'] 
        rep_cols = all_cols['rep_cols'] if "rep_cols" in all_cols else sel_cols
        index_cols = all_cols['index_cols']
        extra_cols = all_cols['extra_cols']
    if "compose_method" in df:
        rep_cols = rep_cols + extra_cols

    main_sel_cols = sel_cols.copy()

    rels = df["prefix"].unique()
    if "xAttr" in rels:
        score_cols = ['rouge_score','num_preds'] 
    else:
        score_cols = ['m_score', 'num_preds'] 

    rep_cols = main_vars + sel_cols + rep_cols + score_cols
    rep_cols = list(dict.fromkeys(rep_cols))
    back_sel_cols = sel_cols.copy()

    sel_fid = "" 
    df_cond = True
    df_conds = []
    open_dfnames = [dfname]
    dot_cols = {}
    selected_cols = []
    rep_cmp = load_obj("rep_cmp", "gtasks", {})
    settings = load_obj("settings", "gtasks", {})
    capt_pos = settings["capt_pos"] if "capt_pos" in settings else "" 
    rname = settings.setdefault("rname", "rpp")
    task = ""
    if "prefix" in df:
        task = df["prefix"][0]
    #if not "learning_rate" in df:
    #    df[['fid_no_lr', 'learning_rate']] = df['fid'].str.split('_lr_', 1, expand=True)
    if not "plen" in df:
        df["plen"] = 8
    if not "blank" in df:
        df["blank"] = "blank"
    if not "opt_type" in df:
        df["opt_type"] = "na"
    if not "rouge_score" in df:
        df["rouge_score"] = 0
    if not "bert_score" in df:
        df["bert_score"] = 0
    prev_cahr = ""
    FID = "fid"
    sel_exp = ""
    infos = []
    back_rows = []
    back_infos = []
    sel_rows = []
    prev_cmd = ""
    do_wrap = True
    sel_group = 0
    group_col = ""
    keep_uniques = False
    group_rows = []

    def row_print(df, col_widths ={}, _print=False):
        nonlocal group_rows, sel_row
        infos = []
        group_mode = group_col and group_col in sel_cols 
        margin = min(len(df), 5) # if not group_mode else 0
        sel_dict = {}
        g_row = ""
        g = 0
        g_start = -1
        row_color = TEXT_COLOR
        sel_col_color = 102 # HL_COLOR #TITLE_COLOR
        selected_col_color = SEL_COLOR
        cross_color = SEL_COLOR # WARNING_COLOR # HL_COLOR   
        sel_row_color = HL_COLOR if not group_mode else row_color 
        g_color = row_color
        _cur_row = cur_row #-1 if group_mode else sel_row 
        ii = 0 
        gg = 0 # count rows in each group
        pp = 0 # count printed rows
        for idx, row in df.iterrows():
           text = "{:<5}".format(ii)
           _sels = []
           _infs = []
           if (group_mode and group_col in row and row[group_col] != g_row):
               g_row = row[group_col]
               gg = 0
               if not keep_uniques and _print and _cur_row >= 0 and ii >= _cur_row - margin:
                   g_text = "{:^{}}".format(g_row, COLS)
                   # mprint("\n", text_win, color = HL_COLOR) 
                   mprint(g_text, text_win, color = HL_COLOR) 
                   # mprint("\n", text_win, color = HL_COLOR) 
               if g_start >= 0:
                   group_rows = range(g_start, ii)
                   g_start = -1
               if g % 2 == 0:
                  row_color = TEXT_COLOR #INFO_COLOR 
                  sel_col_color = ITEM_COLOR 
                  g_color = row_color
               else:
                  row_color = TEXT_COLOR
                  sel_col_color = TITLE_COLOR
                  g_color = row_color
               if g == sel_row: # sel_group:
                  #_cur_row = ii
                  #row_color = SEL_COLOR
                  #g_color = WARNING_COLOR
                  g_start = ii
               g+=1
           if _cur_row < 0 or ii < _cur_row - margin:
               ii += 1
               pp += 1
               continue
           if group_mode and keep_uniques and gg > 0:
               ii += 1
               continue

           # if group_mode: cross_color = sel_col_color
           _color = row_color
           if cur_col < 0:
               if ii == sel_row:
                  _color = HL_COLOR
               else:
                  _color = sel_col_color
           if pp == _cur_row:
               sel_row = ii
           if pp in sel_rows:
               _color = MSG_COLOR
           if pp == _cur_row and not group_mode:
                _color = cross_color if cur_col < 0 else SEL_COLOR 
           if _print:
               mprint(text, text_win, color = _color, end="") 
           if _print:
               _cols = sel_cols + info_cols
           else:
               _cols = sel_cols
           for sel_col in _cols: 
               if  sel_col in _sels:
                   continue
               if not sel_col in row: 
                   if sel_col in sel_cols:
                       sel_cols.remove(sel_col)
                   continue
               content = str(row[sel_col])
               content = content.strip()
               orig_content = content
               content = "{:<4}".format(content) # min length
               if sel_col in wraps and do_wrap:
                   content = content[:wraps[sel_col]] + ".."
               if "score" in sel_col:
                   try:
                       content = "{:.2f}".format(float(content))
                   except:
                       pass
               _info = sel_col + ":" + orig_content
               if sel_col in info_cols:
                   if pp == _cur_row and not sel_col in _infs:
                      infos.append(_info)
                      _infs.append(sel_col)
               if pp == _cur_row:
                   sel_dict[sel_col] = row[sel_col]
               if not sel_col in col_widths:
                   col_widths[sel_col] = len(content) + 2
               if len(content) > col_widths[sel_col]:
                   col_widths[sel_col] = len(content) + 2
               col_title = map_cols[sel_col] if sel_col in map_cols else sel_col
               min_width = max(5, len(col_title) + 1)
               max_width = 100
               if len(sel_cols) > 2:
                   max_width = int(settings["max_width"]) if "max_width" in settings else 36
               _width = max(col_widths[sel_col], min_width)
               _width = min(_width, max_width)
               col_widths[sel_col] = _width 
               _w = col_widths[sel_col] 
               if sel_col in sel_cols:
                   if (cur_col >=0 and cur_col < len(sel_cols) 
                          and sel_col == sel_cols[cur_col]):
                       if pp == _cur_row: 
                          cell_color = cross_color 
                       #elif sel_col in cond_colors:
                       #    cell_color = cond_colors[sel_col](df, ii, sel_col, 
                       #            default = sel_col_color)
                       elif sel_col in selected_cols:
                          cell_color = selected_col_color
                       else:
                          cell_color = sel_col_color
                   else:
                       if ii in sel_rows:
                          cell_color = MSG_COLOR
                       elif sel_col in selected_cols:
                          cell_color = selected_col_color
                       elif pp == _cur_row:
                          cell_color = sel_row_color
                       elif sel_col in cond_colors:
                           cell_color = cond_colors[sel_col](df, ii, sel_col, 
                                   default = row_color)
                       elif sel_col == group_col:
                          cell_color = g_color
                       else:
                          cell_color = row_color
                   content = textwrap.shorten(content, width=max_width, placeholder="...")
                   text = "{:<{x}}".format(content, x= _w)
                   if _print:
                       mprint(text, text_win, color = cell_color, end="") 
                   _sels.append(sel_col)

           _end = "\n"
           if _print:
               mprint("", text_win, color = _color, end="\n") 
           ii += 1
           gg += 1
           pp += 1
           if pp > _cur_row + ROWS:
               break
        return infos, col_widths

    def get_sel_rows(df, row_id="eid", col="eid", from_main=True):
        values = []
        s_rows = sel_rows
        exprs = []
        if not s_rows:
            s_rows = [sel_row]
        for s_row in s_rows:
            row=df.iloc[s_row][row_id]
            if from_main:
                tdf = main_df[main_df[row_id] == row]
            else:
                tdf = df[df[row_id] == row]
            val=tdf.iloc[0][col]
            values.append(val)
            exp=tdf.iloc[0][row_id]
            exprs.append(exp)
        return exprs, values 


    for _col in ["input_text","pred_text1","target_text"]:
        if _col in df:
            df[_col] = df[_col].astype(str)

    map_cols =  load_obj("map_cols", "atomic", {})
    def get_images(df, exps, fid='eid', merge="vert", image_keys="all"):
        imgs = {}
        dest = ""
        start = "pred"
        fnames = []
        for exp in exps:
            cond = f"(main_df['{fid}'] == '{exp}')"
            tdf = main_df[main_df[fid] == exp]
            if tdf.empty:
                return imgs, fnames
            path=tdf.iloc[0]["path"]
            path = Path(path)
            #_selpath = os.path.join(path.parent, "pred_sel" + path.name) 
            #shutil.copy(path, _selpath)
            # grm = tdf.iloc[0]["gen_route_methods"]
            runid = tdf.iloc[0]["eid"]
            #run = "wandb/offline*" + str(runid) + f"/files/media/images/{start}*.png"
            run = "img_logs/{start}*.png"
            paths = glob(str(path.parent) +"/img_logs/*.png")
            # paths = glob(run)
            spath = "images/exp-" + str(runid)
            if Path(spath).exists():
                shutil.rmtree(spath)
            Path(spath).mkdir(parents=True, exist_ok=True)
            images = []
            kk = 1
            key = exp # "single"
            ii = 0
            for img in paths: 
                fname = Path(img).stem
                # if fname in fnames:
                #    continue
                fnames.append(fname) #.split("_")[0])
                parts = fname.split("_")
                ftype = fname.split("@")[1]
                if kk < 0:
                    _, key = list_values(parts)
                    kk = parts.index(key)
                    key = parts[kk]
                dest = os.path.join(spath, fname + ".png") 
                # if not fname.startswith("pred_sel"):
                #    selimg = str(Path(img).parent) + "/pred_sel" +  fname + ".png"
                #    os.rename(img, selimg)
                #    img = selimg
                shutil.copyfile(img, dest)
                _image = Image.open(dest)
                if key == "single": key = str(ii)
                if not key in imgs:
                    imgs[key] = {} # [_image]
                imgs[key][ftype] = _image
                images.append({"image": dest})
        if imgs:
            fnames = []
            c_imgs = {}
            if Path("temp").exists():
                shutil.rmtree("temp")
            Path("temp").mkdir(parents=True, exist_ok=True)
            for key, img_dict in imgs.items():
                #sorted_keys = (sorted(img_dict.keys()))
                if not image_keys:
                  key_list = ["sim", "rsim", "score", 
                                "effect", "init_router", "router", "mask"] 
                  # image_keys = ["score", "router"] 
                else:
                  key_list = []
                  for prefix in image_keys:
                      key_list.extend([k for k in img_dict.keys() if k.startswith(prefix)])
                # TODO fixed
                img_list = [img_dict[k] for k in key_list if k in img_dict] 
                max_width = 0
                if len(img_list) > 0:
                    if len(img_list) > 1 and merge == "vert":
                        new_im = combine_y(img_list)
                    else:
                        new_im = combine_x(img_list)
                    name = str(key) 
                    dest = os.path.join("temp", name.strip("-") + ".png")
                    new_im.save(dest)
                    c_imgs[key] = [new_im] 
                    if new_im.width > max_width:
                        max_width = new_im.width
                    fnames.append(dest)
            for key, img_list in c_imgs.items():
                for i, img in enumerate(img_list):
                    if img.width < max_width:
                        dif = max_width - img.width // 2
                        _image = add_margin(img, 0, dif, 0, dif, (255, 255, 255))
                        c_imgs[key][i] = _image
            imgs = c_imgs
        return imgs, fnames

    if not map_cols:
        map_cols = {
            "epochs_num":"epn",
            "exp_trial":"exp",
            "pred_text1":"pred",
            "target_text":"tgt",
            "template":"tn",
            "pred_max_num":"pnm",
            "attn_learning_rate":"alr",
            "attn_method":"am",
            "attend_source":"att_src",
            "attend_target":"att_tg",
            "attend_input":"att_inp",
            "add_target":"add_tg",
            "rouge_score":"rg",
            "out_score":"out",
            "bert_score":"bt",
            }
    wraps = {
            "tag":20,
            }
    adjust = True
    show_consts = True
    show_extra = False
    consts = {}
    extra = {"filter":[]}
    orig_df = main_df.copy()
    prev_char = ""
    while prev_char != "q":
        text_win.clear()
        group_rows = []
        left = min(left, max_col  - width)
        left = max(left, 0)
        top = min(top, max_row  - height)
        top = max(top, 0)
        sel_row = min(sel_row, len(df) - 1)
        sel_row = max(sel_row, 0)
        cur_row = max(cur_row, 0)
        cur_row = min(cur_row, len(df) - 1)
        sel_rows = sorted(sel_rows)
        sel_rows = list(dict.fromkeys(sel_rows))
        sel_cols = list(dict.fromkeys(sel_cols))
        sel_group = max(sel_group, 0)
        #sel_group = min(sel_row, sel_group)
        cur_col = min(cur_col, len(sel_cols) - 1)
        cur_col = max(cur_col, -1)
        back_df = back[-1].df if len(back) > 0 else df
        if not hotkey:
            if adjust:
                _, col_widths = row_print(df, col_widths={})
            text = "{:<5}".format(sel_row)
            for i, sel_col in enumerate(sel_cols):
               if not sel_col in df:
                   sel_cols.remove(sel_col)
                   continue
               head = sel_col if not sel_col in map_cols else map_cols[sel_col] 
               #head = textwrap.shorten(f"{i} {head}" , width=15, placeholder=".")
               if not sel_col in col_widths and not adjust:
                    _, col_widths = row_print(df, col_widths={})
                    adjust = True
               if sel_col in col_widths and len(head) > col_widths[sel_col]:
                   col_widths[sel_col] = len(head) 
               _w = col_widths[sel_col] if sel_col in col_widths else 5
               if i == cur_col:
                  #head = inline_colors.INFO2 + head + inline_colors.ENDC 
                  head = head + "*"
                  text += "{:<{x}}".format(head, x=_w) 
               else:
                  text += "{:<{x}}".format(head, x=_w) 
            mprint(text, text_win) 
            #fffff
            infos,_ = row_print(df, col_widths, True)
            refresh()
        if cur_col < len(sel_cols) and len(sel_cols) > 0:
            _sel_col = sel_cols[cur_col]
            if not df.empty:
                _sel_val = df.iloc[sel_row][_sel_col]
                infos.append("{},{}:{}".format(sel_row, _sel_col, _sel_val))
        for c in info_cols:
            if not c in df:
                continue
            if "score" in c:
                mean = df[c].mean()
                _info = f"Mean {c}:" + "{:.2f}".format(mean)
                infos.append(_info)
        infos.append("-------------------------")
        if show_consts:
            consts["len"] = str(len(df))
            consts["context"] = context
            consts["keys"] = general_keys
            if context in shortkeys:
                consts["keys"] = {**general_keys,**shortkeys[context]}
            for key,val in consts.items():
                if type(val) == list:
                    val = "-".join(val)
                infos.append("{:<5}:{}".format(key,val))
        if show_extra:
            show_extra = False
            for key,val in extra.items():
                if type(val) == list:
                    val = "-".join(val)
                infos.append("{:<5}:{}".format(key,val))
        change_info(infos)

        prev_char = chr(ch)
        prev_cmd = cmd
        if global_cmd and not hotkey or hotkey == "q":
            cmd = global_cmd
            global_cmd = ""
        else:
            cmd = ""
        if hotkey == "":
            ch = std.getch()
        else:
            ch, hotkey = ord(hotkey[0]), hotkey[1:]
        char = chr(ch)
        if char != "q" and prev_char == "q": 
            consts["exit"] = ""
            prev_char = ""
        extra["inp"] = char

        seq += char
        vals = []
        get_cmd = False
        adjust = True
        #if char in context_map:
        #    context = contexts_map[char] 
        if ch == cur.KEY_NPAGE:
            left += 20
            adjust = False
            cur_col += 5
            ch = RIGHT
        if ch == cur.KEY_PPAGE:
            left -= 20
            adjust = False
            cur_col -= 5
            ch = LEFT
        if ch == SDOWN:
            info_cols_back = info_cols.copy()
            info_cols = []
        if context == "details": # or context == "notes":
            old_search = search
            pattern = re.compile("[A-Za-z0-9]+")
            if ch == cur.KEY_BACKSPACE:
                if search:
                   search = search[:-1]
                   char == ""
                   ch = 0
                else:
                   context = ""
                if not search:
                   mbeep()
            elif pattern.fullmatch(char) is not None:
                search += char 
            if search and search != old_search:
                col = sel_cols[cur_col]
                df = search_df[search_df[col].astype(str).str.contains(search, na=False)]
                # .first_valid_index()
                # si = min(si, len(mask) - 1)
                # sel_row = df.loc[mask.any(axis=1)].index[si]
                char = ""
                ch = 0
            consts["search"] = "/" + search
        if char == ";":
            # info_cols = info_cols_back.copy()
            backit(df, sel_cols)
            context = "details"
            max_width = 100
            consts["search"] = "/"
            infos = []
            for c in df.columns:
                value = df.iloc[sel_row][c]
                _info = {"col":c, "val":value}
                infos.append(_info)
            df = pd.DataFrame(data=infos)
            df = df.sort_values(by="col", ascending=True)
            search_df = df
            sel_cols = ["col","val"]
        if ch == LEFT:
            cur_col -= 1
            cur_col = max(-1, cur_col)
            #if cur_col < 15 and all_sel_cols:
            #    sel_cols = all_sel_cols[:20]
            cur_col = min(len(sel_cols)-1, cur_col)
            cur_sel_col = sel_cols[cur_col]
            width = len(cur_sel_col) + 2
            if cur_sel_col in col_widths:
                width = col_widths[cur_sel_col]
            _sw = sum([col_widths[x] if x in col_widths else len(x) + 2 
                for x in sel_cols[:cur_col]])
            if _sw < left:
                left = _sw - width - 10 
            adjust = False
        if ch == RIGHT:
            cur_col += 1
            #if cur_col > 15 and len(all_sel_cols) > 10:
            #    sel_cols = all_sel_cols[10:30]
            cur_col = min(len(sel_cols)-1, cur_col)
            cur_col = max(0,cur_col)
            cur_sel_col = sel_cols[cur_col]
            width = len(cur_sel_col) + 2
            if cur_sel_col in col_widths:
                width = col_widths[cur_sel_col]
            _sw = sum([col_widths[x] if x in col_widths else len(x) + 2 
                for x in sel_cols[:cur_col]])
            if _sw >= left + COLS - 10:
                left = _sw - 10 
            adjust = False
        if char in ["+","-","*","/"] and prev_char == "x":
            _inp=df.iloc[sel_row]["input_text"]
            _prefix=df.iloc[sel_row]["prefix"]
            _pred_text=df.iloc[sel_row]["pred_text1"]
            _fid=df.iloc[sel_row]["fid"]
            cond = ((main_df["fid"] == _fid) & (main_df["input_text"] == _inp) &
                    (main_df["prefix"] == _prefix) & (main_df["pred_text1"] == _pred_text))
            if char == "+": _score = 1.
            if char == "-": _score = 0.
            if char == "/": _score = 0.4
            if char == "*": _score = 0.7

            main_df.loc[cond, "hscore"] = _score 
            sel_exp = _fid
            sel_row += 1
            adjust = False
        if ch == DOWN:
            if context == "inp":
                back_rows[-1] += 1
                hotkey = "bp"
            elif False: #TODO group_col and group_col in sel_cols:
                sel_group +=1
            else:
                cur_row += 1
            adjust = False
        elif ch == UP: 
            if context == "inp":
                back_rows[-1] -= 1
                hotkey = "bp"
            elif False: #TODO group_col and group_col in sel_cols:
                sel_group -=1
            else:
                cur_row -= 1
            adjust = False
        elif ch == cur.KEY_SRIGHT:
            cur_row += ROWS - 4
        elif ch == cur.KEY_HOME:
            cur_row = 0 
            sel_group = 0
        elif ch == cur.KEY_SHOME:
            left = 0 
        elif ch == cur.KEY_END:
            cur_row = len(df) -1
        elif ch == cur.KEY_SLEFT:
            cur_row -= ROWS - 4
        elif char == "l" and prev_char == "l":
            seq = ""
        elif char == "s":
            col = sel_cols[cur_col]
            df = df.sort_values(by=col, ascending=asc)
            asc = not asc
        elif char in ["+","\""]:
            col = sel_cols[cur_col]
            if col in selected_cols:
                selected_cols.remove(col)
            else:
                selected_cols.append(col)
            consts["selected_cols"] = selected_cols
            mbeep()
        elif char == "-":
            backit(df, sel_cols)
            col = sel_cols[cur_col]
            val=df.iloc[sel_row][col]
            cond = True
            for o in ["gen_norm_method","norm_method"]:
                vo=df.iloc[sel_row][o]
                cond = cond & (df[o] == vo)
            df = df[cond]
        elif char == ".":
            col = sel_cols[cur_col]
            val=df.iloc[sel_row][col]
            dot_cols[col] = val
            if "sel" in consts:
                consts["sel"] += " " + col + "='" + str(val) + "'"
            else:
                consts["sel"] = col + "='" + str(val) + "'"
        elif char == "=":
            col = sel_cols[cur_col]
            val=df.iloc[sel_row][col]
            if col == "fid": col = FID
            if "filter" in consts:
                consts["filter"] += " " + col + "='" + str(val) + "'"
            else:
                consts["filter"] = col + "='" + str(val) + "'"
            df_conds.append((col, df[col] == val))
        elif char == "=" and prev_char == "x":
            col = info_cols[-1]
            sel_cols.insert(cur_col, col)
        elif char == ">":
            col = info_cols.pop()
            sel_cols.insert(cur_col, col)
        elif char in "01234" and prev_char == "#":
            canceled, col, val = list_df_values(df, get_val=False)
            if not canceled:
                sel_cols = order(sel_cols, [col],int(char))
        elif char in ["E"]:
            if not edit_col or char == "E":
                canceled, col, val = list_df_values(df, get_val=False)
                if not canceled:
                    edit_col = col
                    extra["edit col"] = edit_col
                    refresh()
            if edit_col:
                new_val = rowinput()
                if new_val:
                    df.at[sel_row, edit_col] = new_val
                    char = "SS"
        elif char in ["%"]:
            canceled, col, val = list_df_values(df, get_val=False)
            if not canceled:
                if not col in sel_cols:
                    sel_cols.insert(0, col)
                    save_obj(sel_cols, "sel_cols", context)
        elif char in ["W"] and prev_char == "x":
            save_df(df)
        elif char == "B" and prev_char == "x":
            s_rows = sel_rows
            if not s_rows:
                s_rows = [sel_row]
            if prev_char == "x":
                s_rows = range(len(df))
            for s_row in s_rows:
                exp=df.iloc[s_row]["fid"]
                _score=df.iloc[s_row]["bert_score"]
                #if _score > 0:
                #    continue
                cond = f"(main_df['{FID}'] == '{exp}')"
                tdf = main_df[main_df[FID] == exp]
                #df = tdf[["pred_text1", "id", "bert_score","query", "template", "rouge_score", "fid","prefix", "input_text","target_text"]]
                spath = tdf.iloc[0]["path"]
                spath = str(Path(spath).parent)
                tdf = do_score(tdf, "rouge-bert", spath, reval=True) 
                tdf = tdf.reset_index()
                #main_df.loc[eval(cond), "bert_score"] = tdf["bert_score"]
            df = main_df
            hotkey = hk
        if char == "O":
            sel_exp=df.iloc[sel_row]["eid"]
            tdf = main_df[main_df['eid'] == sel_exp]
            spath = tdf.iloc[0]["path"]
            subprocess.Popen(["nautilus", spath])
        if char in ["o","y","k", "p"]:
            tdf = df #pivot_df if pivot_df is not None and context == "pivot" else df
            images = []
            exprs, _ = get_sel_rows(tdf)
            merge = "vert"
            image_keys = "" 
            if char == "o" and "images" in settings:
                image_keys = settings["images"].split("@")
            elif char == "y":
                image_keys = ["score","sim"]
                merge = "vert"
            elif char == "k" or char == "p":
                image_keys = ["score","router","effect"]
                merge = "vert"

            experiment_images, fnames = get_images(tdf, exprs, 'eid', 
                    merge = merge, image_keys = image_keys)
            dif_cols = ["expid"]
            for col in sel_cols:
                if col in pcols or col in ["eid"]:
                    continue
                vals = []
                for exp in exprs:
                    if col in tdf:
                        v = tdf.loc[tdf.eid == exp, col].iloc[0]
                        vals.append(v)
                if all(str(x).strip() == str(vals[0]).strip() for x in vals):
                    continue
                else:
                    dif_cols.append(col)

            capt_pos = settings["capt_pos"] if "capt_pos" in settings else "" 
            pic = None
            for key, img_list in experiment_images.items(): 
                im = img_list[0]
                images.append(im)

            if images:
                pic = combine_x(images) # if char =="o" else combine_y(images)
                if len(images) > 1:
                    font = ImageFont.truetype("/usr/share/vlc/skins2/fonts/FreeSans.ttf", 30)
                else:
                    font = ImageFont.truetype("/usr/share/vlc/skins2/fonts/FreeSans.ttf", 20)
                im = pic
                if capt_pos and capt_pos != "none" and char != "y":
                    width, height = im.size
                    gap = 150*len(exprs) + 50
                    if capt_pos == "top":
                        _image = add_margin(im, gap, 5, 0, 5, (255, 255, 255))
                        xx = 10
                        yy = 30 
                    elif capt_pos == "below":
                        _image = add_margin(im, 0, 5, gap, 5, (255, 255, 255))
                        xx = 10
                        yy = height + 50 
                    elif capt_pos == "left":
                        _image = add_margin(im, 0, 5, 5, 700, (255, 255, 255))
                        xx = 10
                        yy = 10
                    draw = ImageDraw.Draw(_image)
                    for col_set in [dif_cols, pcols]:
                        for key in exprs:
                            caption_dict = {}
                            if context == "pivot" and not tdf.loc[tdf['eid'] == key].empty:
                                caption_dict = tdf.loc[tdf['eid'] == key, 
                                        col_set].iloc[0].to_dict()
                            for cc, value in caption_dict.items(): 
                                if cc in pcols:
                                    cc = cc.split("_")[0]
                                if cc.endswith("score"):
                                    mm = map_cols[cc] if cc in map_cols else cc
                                    mm = "{}:".format(mm)
                                    draw.text((xx, yy),mm,(150,150,150),font=font)
                                    tw, th = draw.textsize(mm, font)
                                    mm = "{:.2f}".format(value)
                                    xx += tw + 10
                                    draw.text((xx, yy),mm,(250,5,5),font=font)
                                    tw, th = draw.textsize(mm, font)
                                else:
                                    mm = map_cols[cc] if cc in map_cols else cc
                                    mm = "{}:".format(mm)
                                    draw.text((xx, yy),mm,(150,150,150),font=font)
                                    tw, th = draw.textsize(mm, font)
                                    mm = "{}".format(value)
                                    xx += tw + 10
                                    draw.text((xx, yy),mm,(20,25,255),font=font)
                                    tw, th = draw.textsize(mm, font)
                                if capt_pos == "left":
                                    xx = 10
                                    yy += 60
                                else:
                                    xx += tw + 10
                            yy += 40
                            xx = 10
                        yy += 40
                        xx = 10
                    pic = _image
            if pic is not None:
                dest = os.path.join("routers.png")
                if char == "p":
                    # fname = rowinput("prefix:", default="image")
                    ptemp = os.path.join(home, "pictemp", "image.png")
                    if Path(ptemp).is_file():
                        pic2 = Image.open(ptemp)
                        pic = combine_x([pic, pic2])
                    else:
                        pic.save(ptemp)
                pic.save(dest)
                #pname=tdf.iloc[sel_row]["image"]
                subprocess.Popen(["eog", dest])
        elif char == "L":
            s_rows = sel_rows
            if not sel_rows:
                s_rows = group_rows
                if not group_rows:
                    s_rows = [sel_row]
            all_rows = range(len(df))
            Path("temp").mkdir(parents=True, exist_ok=True)
            imgs = []
            for s_row in all_rows:
                exp=df.iloc[s_row]["fid"]
                cond = f"(main_df['{FID}'] == '{exp}')"
                tdf = main_df[main_df[FID] == exp]
                path=tdf.iloc[0]["path"]
                folder = str(Path(path).parent)
                path = os.path.join(folder, "last_attn*.png")
                images = glob(path)
                tdf = pd.DataFrame(data = images, columns = ["image"])
                tdf = tdf.sort_values(by="image", ascending=False)
                pname=tdf.iloc[0]["image"]
                dest = os.path.join("temp", str(s_row) + ".png")
                shutil.copyfile(pname, dest)
                if s_row in s_rows:
                    _image = Image.open(pname)
                    imgs.append(_image)
            if imgs:
                new_im = combine_y(imgs)
                name = "-".join([str(x) for x in s_rows]) 
                pname = os.path.join("temp", name.strip("-") + ".png")
                new_im.save(pname)
            subprocess.run(["eog", pname])
        elif char == "l" and prev_char == "p":
            exp=df.iloc[sel_row]["fid"]
            cond = f"(main_df['{FID}'] == '{exp}')"
            tdf = main_df[main_df[FID] == exp]
            path=tdf.iloc[0]["path"]
            conf = os.path.join(str(Path(path).parent), "exp.json")
            with open(conf,"r") as f:
                infos = f.readlines()
            subwin(infos)
        elif char == "l" and prev_char == "x":
            exp=df.iloc[sel_row]["eid"]
            exp = str(exp)
            logs = glob(str(exp) + "*.log")
            if logs:
                log = logs[0]
                with open(log,"r") as f:
                    infos = f.readlines()
                subwin(infos)

        elif char == "<":
            col = sel_cols[cur_col]
            sel_cols.remove(col)
            info_cols.append(col)
            save_obj(sel_cols, "sel_cols", context)
            save_obj(info_cols, "info_cols", context)
        elif char == "N" and prev_char == "x":
            backit(df,sel_cols)
            sel_cols=["pred_max_num","pred_max", "tag","prefix","rouge_score", "num_preds","bert_score"]
        elif (char == "i" and not prev_char == "x" and hk=="G"):
            backit(df,sel_cols)
            exp=df.iloc[sel_row]["fid"]
            cond = f"(main_df['{FID}'] == '{exp}')"
            df = main_df[main_df[FID] == exp]
            sel_cols=tag_cols + ["bert_score", "out_score", "pred_text1","target_text","input_text","rouge_score","prefix"]
            sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, orig_tag_cols)
            unique_cols = info_cols.copy()
            df = df[sel_cols]
            df = df.sort_values(by="input_text", ascending=False)
        elif char == "I" or char == "#" or char == "3":
            canceled, col, val = list_df_values(df, get_val=False, extra=["All"])
            if not canceled:
                if col == "All":
                    sel_cols = list(df.columns)
                else:
                    if col in sel_cols: 
                        sel_cols.remove(col)
                    if col in info_cols:
                        info_cols.remove(col)
                    if char == "#" or char == "3": 
                        sel_cols.insert(cur_col, col)
                    else:
                        info_cols.append(col)
                    orig_tag_cols.append(col)
            save_obj(sel_cols, "sel_cols", context)
            save_obj(info_cols, "info_cols", context)
        elif char in ["o","O"] and prev_char == "x":
            inp = df.loc[df.index[sel_row],["prefix", "input_text"]]
            df = df[(df.prefix != inp.prefix) | 
                    (df.input_text != inp.input_text)] 
            mbeep()
            sel_df = df.sort_values(by=["prefix","input_text","target_text"]).drop_duplicates()
            sel_df.to_csv(sel_path, sep="\t", index=False)
        elif char in ["w","W"]:
            inp = df.loc[df.index[sel_row],["prefix", "input_text","pred_text1"]]
            df.loc[(df.prefix == inp.prefix) & 
                    (df.input_text == inp.input_text),["sel"]] = True
            _rows = main_df.loc[(main_df.prefix == inp.prefix) & 
                    (main_df.input_text == inp.input_text), 
                    ["prefix","input_text", "target_text","sel"]]
            row = df.loc[(df.prefix == inp.prefix) & 
                    (df.input_text == inp.input_text),:]
            sel_df = sel_df.append(_rows,ignore_index=True)
            #df = df.sort_values(by="sel", ascending=False).reset_index(drop=True)
            #sel_row = row.index[0]
            if char == "W":
                new_row = {"prefix":inp.prefix,
                           "input_text":inp.input_text,
                           "target_text":inp.pred_text1, "sel":False}
                sel_df = sel_df.append(new_row, ignore_index=True)
            consts["sel_path"] = sel_path + "|"+ str(len(sel_df)) + "|" + str(sel_df["input_text"].nunique())
            mbeep()
            sel_df = sel_df.sort_values(by=["prefix","input_text","target_text"]).drop_duplicates()
            sel_df.to_csv(sel_path, sep="\t", index=False)
        elif char == "h" and False:
            backit(df, sel_cols)
            sel_cols = ["prefix", "input_text", "target_text", "sel"]
            df = sel_df
        elif char in ["h","v"] and prev_char == "x":
            _cols = ["template", "model", "prefix"]
            _types = ["l1_decoder", "l1_encoder", "cossim_decoder", "cossim_encoder"]
            canceled, col = list_values(_cols)
            folder = "/home/ahmad/share/comp/"
            if Path(folder).exists():
                shutil.rmtree(folder)
            Path(folder).mkdir(parents=True, exist_ok=True)
            files = []
            for _type in _types:
                g_list = ["template", "model", "prefix"]
                mm = main_df.groupby(g_list, as_index=False).first()
                g_list.remove(col)
                mlog.info("g_list: %s", g_list)
                g_df = mm.groupby(g_list, as_index=False)
                sel_cols = [_type, "template", "model", "prefix"]
                #_agg = {}
                #for _g in g_list:
                #    _agg[_g] ="first"
                #_agg[col] = "count"
                #df = g_df.agg(_agg)
                if True:
                    gg = 1
                    total = len(g_df)
                    for g_name, _nn in g_df:
                        img = []
                        images = []
                        for i, row in _nn.iterrows():
                            if row[_type] is None:
                                continue
                            f_path = row[_type] 
                            if not Path(f_path).is_file(): 
                                continue
                            img.append(row[_type])
                            _image = Image.open(f_path)
                            draw = ImageDraw.Draw(_image)
                            draw.text((0, 0),str(i) +" "+ row[col] ,(20,25,255),font=font)
                            draw.text((0, 70),str(i) +" "+ g_name[0],(20,25,255),font=font)
                            draw.text((0, 140),str(i) +" "+ g_name[1],(20,25,255),font=font)
                            draw.text((250, 0),str(gg) + " of " + str(total),
                                    (20,25,255),font=font)
                            images.append(_image)
                        gg += 1
                        if images:
                            if char == "h":
                                new_im = combine_x(images)
                            else:
                                new_im = combine_y(images)
                            name = _type + "_".join(g_name) + "_" + row[col]
                            pname = os.path.join(folder, name + ".png")
                            new_im.save(pname)
                            files.append({"pname":pname, "name":name})
                if files:
                    df = pd.DataFrame(files, columns=["pname","name"])
                    sel_cols = ["name"]
                else:
                    show_msg("No select")
        elif char == "x" and prev_char == "b" and context == "":
            backit(df, sel_cols)
            df = sel_df
        # png files
        elif char == "l" and prev_char == "x":
            df = main_df.groupby(["l1_decoder", "template", "model", "prefix"], as_index=False).first()
            sel_cols = ["l1_decoder", "template", "model", "prefix"]
        elif char == "z" and prev_char == "x":
            fav_df = fav_df.append(df.iloc[sel_row])
            mbeep()
            fav_df.to_csv(fav_path, sep="\t", index=False)
        elif char == "Z" and prev_char == "x":
            main_df["m_score"] = main_df["rouge_score"]
            df = main_df
            hotkey = "CGR"
            #backit(df, sel_cols)
            #df = fav_df
        elif char == "j" and False:
            canceled, col = list_values(info_cols)
            if not canceled:
                pos = rowinput("pos:","")
                if pos:
                    info_cols.remove(col)
                    if int(pos) > 0:
                        info_cols.insert(int(pos), col)
                    else:
                        sel_cols.insert(0, col)
                    save_obj(info_cols, "info_cols", dfname)
                    save_obj(sel_cols, "sel_cols", dfname)
        elif char in "56789" and prev_char == "\\":
            cmd = "top@" + str(int(char)/10)
        elif char == "BB": 
            sel_rows = []
            for i in range(len(df)):
                sel_rows.append(i)
        elif char == "==": 
            col = sel_cols[cur_col]
            exp=df.iloc[sel_row][col]
            if col == "fid": col = FID
            if col == "fid":
                sel_fid = exp
            mlog.info("%s == %s", col, exp)
            df = main_df[main_df[col] == exp]
            filter_df = df
            hotkey = hk
        elif char  == "a" and prev_char == "a": 
            col = sel_cols[cur_col]
            FID = col 
            extra["FID"] = FID
            df = filter_df
            hotkey=hk
        elif char == "A" and prev_char == "g":
            col = sel_cols[cur_col]
            FID = col 
            extra["FID"] = FID
            df = main_df
            hotkey=hk
        elif char == "AA":
            gdf = filter_df.groupby("input_text")
            rows = []
            for group_name, df_group in gdf:
                for row_index, row in df_group.iterrows():
                    pass
            arr = ["prefix","fid","query","input_text","template"]
            canceled, col = list_values(arr)
            if not canceled:
                FID = col 
                extra["FID"] = FID
                df = filter_df
                hotkey=hk
        elif is_enter(ch) and prev_char == "s": 
            sort = selected_cols + [col] 
            df = df.sort_values(by=sort, ascending=asc)
            selected_cols = []
            asc = not asc
        if context == "grouping":
            if not selected_cols:
                selected_cols = ["label","compose_method","max_train_samples"]
            if char == "m":
                df = back_df
                df = df.groupby(selected_cols).mean(numeric_only=True).reset_index()
            elif char == "s":
                df = back_df
                df = df.groupby(selected_cols).std(numeric_only=True).reset_index()
            df = df.round(2)
        elif char in ["a"] and context != "grouping":
            backit(df, sel_cols)
            context = "grouping"
            shortkeys["grouping"] = {"m":"show mean","s":"show std"}
            if not selected_cols:
                selected_cols = ["label","compose_method","max_train_samples"]
            if len(selected_cols) > 0:
                df = df.groupby(selected_cols).mean(numeric_only=True).reset_index()
                df = df.round(2)
        elif char in ["g", "u"]:
            context = "group_mode"
            if cur_col < len(sel_cols):
                col = sel_cols[cur_col]
                keep_uniques = char == "u"
                if col == group_col:
                    group_col = ""
                    cur_row = 0
                    sel_group = 0
                else:
                    group_col = col
                    cur_row = 0
                    sel_group = 0
                    df = df.sort_values(by=[group_col, sort], ascending=[True, False])
        elif char == "A": 
            consts["options"] = "b: back"
            backit(df, sel_cols)
            if not "eid" in sel_cols:
                sel_cols.insert(1, "eid")
            df = df.groupby(["eid"]).mean(numeric_only=True).reset_index()
            if "m_score" in df:
                df = df.sort_values(by=["m_score"], ascending=False)
            elif "avg" in df:
                df = df.sort_values(by=["avg"], ascending=False)
        elif char == "a" and False: 
            consts["options"] = "b: back"
            backit(df, sel_cols)
            col = sel_cols[cur_col]
            df = df.groupby([col]).mean(numeric_only=True).reset_index()
            df = df.round(2)
            if "m_score" in df:
                df = df.sort_values(by=["m_score"], ascending=False)
                sort = "m_score"
            elif "avg" in df:
                df = df.sort_values(by=["avg"], ascending=False)
                sort = "avg"
        elif char == "u" and False:
            infos = calc_metrics(main_df)
            subwin(infos)
        elif char == "U" and prev_char == "x": 
            if sel_col:
                df = df[sel_col].value_counts(ascending=False).reset_index()
                sel_cols = list(df.columns)
                col_widths["index"]=50
                info_cols = []
        elif char == "C": 
            score_col = "rouge_score"
            # backit(df, sel_cols)
            df["rouge_score"] = df.groupby(['fid','prefix','input_text'])["rouge_score"].transform("max")
            df["bert_score"] = df.groupby(['fid','prefix','input_text'])["bert_score"].transform("max")
            df["hscore"] = df.groupby(['fid','prefix','input_text'])["hscore"].transform("max")
            df["pred_freq"] = df.groupby(['fid','prefix','pred_text1'],
                             sort=False)["pred_text1"].transform("count")
            cols = ['fid', 'prefix']
            tdf = df.groupby(["fid","input_text","prefix"]).first().reset_index()
            df = df.merge(tdf[cols+['pred_text1']]
                 .value_counts().groupby(cols).head(1)
                 .reset_index(name='pred_max_num').rename(columns={'pred_text1': 'pred_max'})
               )



            #temp = (pd
            #       .get_dummies(df, columns = ['pred_text1'], prefix="",prefix_sep="")
            #       .groupby(['fid','prefix'])
            #       .transform('sum'))
            #df = (df
            #.assign(pred_max_num=temp.max(1), pred_max = temp.idxmax(1))
            #)

            extra["filter"].append("group predictions")
        elif char == " ":
            if sel_row in sel_rows:
                sel_rows.remove(sel_row)
            else:
                sel_rows.append(sel_row)
            sel_rows = sorted(sel_rows)
            adjust = False
        elif char == "#" and prev_char == "x": 
            if not sel_rows:
                tinfo=df.iloc[sel_row]["ftag"]
                infos = tinfo.split(",")
                infos.append(main_df.loc[0, "path"])
                subwin(infos)
            else:
                s1 = sel_rows[0]
                s2 = sel_rows[1]
                f1 = df.iloc[s1]["eid"]
                f2 = df.iloc[s2]["eid"]
                infos = []
                for col in main_df.columns:
                    if (
                        col == "ftag"
                        or col == "extra_fields"
                        or col == "path"
                        or col == "folder"
                        or col == "exp_name"
                        or col == "fid"
                        or col == "id"
                        or col == "eid"
                        or col == "full_tag"
                        or col.startswith("test_")
                        or "output_dir" in col
                    ):
                        continue

                    values_f1 = main_df[main_df["eid"] == f1][col]
                    values_f2 = main_df[main_df["eid"] == f2][col]

                    if (pd.notna(values_f1.iloc[0]) and pd.notna(values_f2.iloc[0]) 
                            and values_f1.iloc[0] != values_f2.iloc[0]):
                        if values_f1.iloc[0] != values_f2.iloc[0]:
                            infos.append(f"{col}: {values_f1.iloc[0]}")
                            infos.append(f"{col}: {values_f2.iloc[0]}")

                subwin(infos)                
        elif char == "z" and prev_char == "z":
            consts["context"] = context
            sel_cols =  load_obj("sel_cols", context, [])
            info_cols = load_obj("info_cols", context, [])
        elif char == "G":
            # backit(df, sel_cols)
            context = "main"
            if FID == "input_text":
                context = "inp2"
            col = FID
            left = 0
            col = [col, "prefix"]
            if False: #reset:
                info_cols = ["bert_score", "num_preds"]
            if False: #col == "fid":
                sel_cols = ["eid", "rouge_score"] + tag_cols + ["method", "trial", "prefix","num_preds", "bert_score", "out_score", "pred_max_num","pred_max", "steps","max_acc","best_step", "st_score", "learning_rate",  "num_targets", "num_inps", "train_records", "train_records_nunique", "group_records", "wrap", "frozen", "prefixed"] 
                sel_cols = list(dict.fromkeys(sel_cols))
            reset = False

            _agg = {}
            group_sel_cols = sel_cols.copy()
            if "folder" in group_sel_cols:
                group_sel_cols.remove("folder")
            if "prefix" in group_sel_cols:
                group_sel_cols.remove("prefix")
                group_sel_cols.insert(0, "prefix")

            for c in df.columns:
                if c.endswith("score"):
                    _agg[c] = "mean"
                else:
                    _agg[c] = ["first", "nunique"]
            #df = df.groupby(col).agg(_agg).reset_index(drop=True)
            gb = df.groupby(col)
            counts = gb.size().to_frame(name='group_records')
            counts.columns = counts.columns.to_flat_index()
            gbdf = gb.agg(_agg)
            gbdf.columns = gbdf.columns.to_flat_index()
            df = (counts.join(gbdf))
            df = df.reset_index(drop=True)
            scols = [c for c in df.columns if type(c) != tuple]
            tcols = [c for c in df.columns if type(c) == tuple]
            df.columns = scols + ['_'.join(str(i) for i in col) for col in tcols]
            avg_len = 1 #(df.groupby(col)["pred_text1"]
                        #   .apply(lambda x: np.mean(x.str.len()).round(2)))
            ren = {
                    "target_text_nunique":"num_targets",
                    "pred_text1_nunique":"num_preds",
                    "input_text_nunique":"num_inps",
                    }
            for c in df.columns:
                #if c == FID + "_first":
                #    ren[c] = "fid"
                if c.endswith("_mean"):
                    ren[c] = c.replace("_mean","")
                elif c.endswith("_first"):
                    ren[c] = c.replace("_first","")
            df = df.rename(columns=ren)
            if not "num_preds" in sel_cols:
                sel_cols.append("num_preds")
            df["avg_len"] = avg_len
            df = df.sort_values(by = ["rouge_score"], ascending=False)
            sel_cols = ["expname","eid","prefix","rouge_score","num_preds"]
            group_sel_cols = sel_cols.copy()
            group_df = df.copy()
            exp_num = df["folder"].nunique()
            consts["Number of experiments"] = str(exp_num)
            sort = "rouge_score"
        elif char == "z":
            backit(df, sel_cols)
            exprs, _ = get_sel_rows(df)
            sel_rows = []
            df = df[df['eid'].isin(exprs)]
        elif char == "z" and prev_char == "a": 
            if len(df) > 1:
                sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, 
                        orig_tag_cols, keep_cols)
                unique_cols = info_cols.copy()
            info_cols_back = info_cols.copy()
            info_cols = []
            save_obj(sel_cols, "sel_cols", context)
            save_obj(info_cols, "info_cols", context)
        elif char == "M":
            exp=df.iloc[sel_row]["eid"]
            cond = f"(main_df['eid'] == '{exp}')"
            tdf = main_df[main_df.eid == exp]
            path=tdf.iloc[0]["output_dir"]
            js = os.path.join(path, "exp.json")
            meld = ["meld", js]
            if "conf" in tdf:
                conf = tdf.iloc[0]["conf"]
                if not "/" in conf:
                    conf = os.path.join(home, "results", conf + ".json")
                meld.append(conf)
            subprocess.Popen(meld)
        elif char == "B":
            if "cfg" in df:
                _,files = get_sel_rows(df, row_id="cfg", col="cfg", from_main=False)
                files = [os.path.join(home, "results", c + ".json") for c in files]
                consts["base"] = files[0]
            else:
                _,dirs = get_sel_rows(df, col="output_dir")
                out_dir = dirs[0]
                exp_files = [os.path.join(d, "exp.json") for d in dirs]
                exp_file = exp_files[0]
                if "base" in consts:
                    base_file = consts["base"]
                    src = os.path.join(Path(base_file).parent, Path(base_file).stem)
                    dst = os.path.join(Path(exp_file).parent.parent, 
                            Path(base_file).stem + "_base")
                    shutil.copytree(src, dst)
                    mbeep()
                    #arr = ["meld", base_file, exp_file]
                    #subprocess.run(arr)
        elif char == "t":
            backit(df, sel_cols)
            mode = "cfg"
            files = glob(os.path.join(home,"results","*.json"))
            #for f in files:
            # clean_file(f)
            fnames = [Path(f).stem for f in files]
            rows = []
            for fname,_file in zip(fnames, files):
                ts = os.path.getmtime(_file)
                ctime = datetime.utcfromtimestamp(ts).strftime('%m-%d %H:%M:%S')
                parts = fname.split("@")
                rest = parts
                score = ""
                if len(parts) > 1:
                    rest = parts[0]
                    score = parts[1][:4]
                score = score.replace(".json","")
                score = float(score)
                method, cmm, ep, tn = rest.split("_")
                rows.append({"cfg":fname, "score":score, 
                    "method": method, "at":ctime, "cmm":cmm, "ep":ep, "tn":tn})
            df = pd.DataFrame(data=rows)
            sel_cols = df.columns
            df = df.sort_values("at", ascending=False)
            #with open(, 'r') as f:
            #    lines = f.readlines()
            #subwin(lines)                
        elif char == "T" or char == "U" or char == "Y":
            s_rows = sel_rows
            if not sel_rows:
                s_rows = [sel_row]
            pfix = ""
            ignore_fname = False if char == "T" else True
            if char == "U":
                cmd = "sshpass -p 'a' ssh -t ahmad@10.42.0.2 'rm /home/ahmad/comp/*'"
                os.system(cmd)
            for s_row in s_rows:
                exp=df.iloc[s_row]["eid"]
                score = ""
                if "rouge_score" in df:
                    score=df.iloc[s_row]["rouge_score"]
                elif "All" in df:
                    score=df.iloc[s_row]["All"]
                cond = f"(main_df['eid'] == '{exp}')"
                tdf = main_df[main_df.eid == exp]
                prefix=tdf.iloc[0]["expname"]
                expid=tdf.iloc[0]["expid"]
                compose=tdf.iloc[0]["compose_method"]
                epc=tdf.iloc[0]["num_train_epochs"]
                tn=tdf.iloc[0]["max_train_samples"]
                path=tdf.iloc[0]["output_dir"]
                js = os.path.join(path, "exp.json")
                score = str(round(score,2)) if score else "noscore" 
                fname = prefix + "-" + compose + "-" + str(epc) \
                        + "-" + str(tn) + "@" + score + "@.json"
                if not ignore_fname:
                    fname = rowinput("prefix:", default=fname)
                if fname:
                    parent = Path(path).parent
                    pname = Path(path).parent.name
                    expid = Path(path).name
                    if char == "U":
                        dest = os.path.join(home, "comp", "comp_" + prefix + ".json")
                    elif "conf" in fname:
                        dest = os.path.join(home, "confs", fname)
                    elif "reval" in fname or char == "Y":
                        dest = os.path.join(home, "reval", fname)
                    else:
                        folders = glob(os.path.join(str(parent), "Eval-"+ str(expid) + "*"))
                        results_folder = os.path.join(home,"results",
                                fname.replace(".json",""))
                        for folder in folders:
                            try:
                                shutil.copytree(folder, 
                                        results_folder + "/" + Path(folder).name)
                            except FileExistsError:
                                pass
                        dest = os.path.join(home, "results", fname)
                    shutil.copyfile(js, dest)
                    clean_file(dest)
                    mbeep()
                    to = "ahmad@10.42.0.2:" + dest 
                    cmd = f'sshpass -p "a" rsync -P -ae "ssh" -zarv "{js}" "{to}"'
                    os.system(cmd)
                    # subprocess.run(cmd.split())
        elif char == "p" and False:
            pivot_cols = sel_cols[cur_col]
            consts["pivot col"] = pivot_cols
        elif char == "K":
            folder_path = "/home/ahmad/pictemp"
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    else:
                        show_msg(f"Skipping {file_path} as it is not a file.")
                except Exception as e:
                    show_msg(f"Error while deleting {file_path}: {e}") 
            show_msg("Done!")
            mbeep()
        elif char == "U" and False:
            left = 0
            backit(df, sel_cols)

            s_rows = sel_rows
            if not sel_rows:
                s_rows = [sel_row]
            cond = ""
            for s_row in s_rows:
                exp=df.iloc[s_row]["fid"]
                cond += f"| (main_df['{FID}'] == '{exp}') "
            cond = cond.strip("|")
            filter_df = main_df[eval(cond)]
            df = filter_df.copy()
            sel_rows = []
            FID = "input_text"
            hotkey = hk
        elif char == "e" and context != "notes":
            if sort != "time":
                df = df.sort_values(by="time", ascending=False)
                sort = "time"
            elif "All" in df:
                df = df.sort_values(by="All", ascending=False)
                sort = "All"
        elif (char == "i" or char == "j") and context == "pivot": 
            backit(df, sel_cols)
            context = "prefix"
            col = sel_cols[cur_col]
            s_rows = sel_rows
            if not sel_rows: s_rows = [sel_row]
            dfs = []
            for s_row in s_rows:
                sel_exp=df.iloc[s_row]["eid"]
                if char == "j":
                    tdf = group_df[(group_df['eid'] == sel_exp) 
                            & (group_df["prefix"] == col)]
                else:
                    tdf = group_df[(group_df['eid'] == sel_exp)] 
                dfs.append(tdf)
            df = pd.concat(dfs, ignore_index=True)
            sel_cols = index_cols + ["fid"] + group_sel_cols.copy()
            df = df.sort_values(by=["prefix",score_cols[0]], ascending=False)
            left = 0
            sel_rows = range(len(df))
            char = ""
            if char == "j":
                char = "i"
        if char in ["n", "i"] and "fid" in df: # and prev_cahr != "x" and hk == "gG":
            backit(df, sel_cols)
            left = 0
            context= "comp"
            cur_col = -1
            sel_group = 0
            s_rows = sel_rows
            if not sel_rows:
                s_rows = group_rows
                if not group_rows:
                    s_rows = [sel_row]
            sel_rows = sorted(sel_rows)
            if sel_rows:
                sel_row = sel_rows[-1]
            sel_rows = []
            on_col_list = ["pred_text1"]
            other_col = "target_text"
            if char =="i": 
                pass
            if "xAttr" in pcols:
                group_col = "input_text"
                on_col_list = ["input_text"] 
                other_col = "pred_text1"
            if char =="t": 
                on_col_list = ["target_text"] 
                other_col = "pred_text1"
            on_col_list.extend(["prefix"])
            g_cols = []
            _rows = s_rows
            if char == "n":
                dfs = []
                all_rows = range(len(df))
                for r1 in all_rows:
                    for r2 in all_rows:
                        if r2 > r1:
                            _rows = [r1, r2]
                            _df, sel_exp = find_common(df, filter_df, on_col_list, _rows, FID, char, tag_cols = index_cols)
                            dfs.append(_df)
                df = pd.concat(dfs,ignore_index=True)
                #df = df.sort_values(by="int", ascending=False)
            elif len(s_rows) > 1:
                sel_cols=orig_tag_cols + ["num_preds","eid","bert_score", "out_score","pred_text1","target_text","input_text","rouge_score","prefix"]
                sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, orig_tag_cols)
                unique_cols = info_cols.copy()
                sel_cols = list(dict.fromkeys(sel_cols))
                _cols = tag_cols + index_cols + sel_cols + rep_cols + info_cols
                _cols = list(dict.fromkeys(_cols))
                df, sel_exp, dfs = find_common(df, main_df, on_col_list, _rows, 
                                               FID, char, _cols)
                df = pd.concat(dfs).sort_index(kind='mergesort')
                _all = len(df)
                cdf=df.sort_values(by='input_text').drop_duplicates(subset=['input_text', 'pred_text1',"prefix"], keep=False)
                _common = _all - len(cdf)
                consts["Common"] = str(_common) + "| {:.2f}".format(_common / _all)
            else:
                #path = df.iloc[sel_row]["path"]
                #path = Path(path)
                #_selpath = os.path.join(path.parent, "sel_" + path.name) 
                #shutil.copyfile(path, _selpath)
                exp=df.iloc[sel_row]["fid"]
                sel_exp = exp
                #FID="expid"
                cond = f"(main_df['{FID}'] == '{exp}')"
                df = main_df[main_df[FID] == exp]
                if "prefix" in df:
                    task = df.iloc[0]["prefix"]
                sel_cols=orig_tag_cols + ["num_preds","prefix","bert_score", "out_score","pred_text1","top_pred", "top", "target_text","input_text","rouge_score","prefix"]
                sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, 
                        main_vars, keep_cols=["pred_text1"])
                #unique_cols = info_cols.copy()
                sel_cols = list(dict.fromkeys(sel_cols))
                # df = df[sel_cols]
                df = df.sort_values(by="input_text", ascending=False)
                sort = "input_text"
                info_cols = ["input_text","prefix"]
                df = df.reset_index()
            if len(df) > 1:
                sel_cols=orig_tag_cols + ["eid","prefix", "bert_score","pred_text1", "target_text", "top_pred", "input_text", "rouge_score"]
                ii = 0
                for col in index_cols:
                    if col in sel_cols:
                        sel_cols.remove(col)
                    sel_cols.insert(ii, col)
                    ii += 1
                sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, 
                        main_vars, keep_cols=["fid", "prefix", "pred_text1"])
                if "pred_text1" in sel_cols:
                    sel_cols.remove("pred_text1")
                ii = 0
                _sort = []
                for col in index_cols:
                    if col in sel_cols:
                        _sort.append(col)
                        ii += 1
                sel_cols.insert(ii, "prefix")
                sel_cols.insert(ii + 1, "pred_text1")
                sel_cols = list(dict.fromkeys(sel_cols))
                df = df.sort_values(by=["input_text","prefix"]+_sort, ascending=False)
                unique_cols = info_cols.copy()
                info_cols_back = info_cols.copy()
                info_cols = []

        elif char == "M" and prev_char == "l":
            left = 0
            if sel_exp and on_col_list:
                backit(df, sel_cols)
                _col = on_col_list[0]
                _item=df.iloc[sel_row][_col]
                sel_row = 0
                if sel_fid:
                    df = main_df[(main_df["fid"] == sel_fid) & (main_df[FID] == sel_exp) & (main_df[_col] == _item)]
                else:
                    df = main_df[(main_df[FID] == sel_exp) & (main_df[_col] == _item)]
                sel_cols = ["fid","input_text","pred_text1","target_text","bert_score", "hscore", "rouge_score", "prefix"]
                df = df[sel_cols]
                df = df.sort_values(by="bert_score", ascending=False)
        elif char == "D": 
            s_rows = sel_rows
            if FID == "fid":
                mdf = main_df.groupby("fid", as_index=False).first()
                mdf = mdf.copy()
                _sels = df["fid"]
                for s_row, row in mdf.iterrows():
                    exp=row["fid"]
                    if char == "d":
                        cond = main_df['fid'].isin(_sels) 
                    else:
                        cond = ~main_df['fid'].isin(_sels) 
                    tdf = main_df[cond]
                    if  ch == cur.KEY_SDC:
                        spath = row["path"]
                        os.remove(spath)
                    main_df = main_df.drop(main_df[cond].index)
                df = main_df
                filter_df = main_df
                sel_rows = []
                hotkey = hk
        elif char == "D" and prev_char == "x":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                del main_df[col]
                char = "SS"
                if col in df:
                    del df[col]
        elif char == "o" and prev_char == "x":
            if "pname" in df:
                pname = df.iloc[sel_row]["pname"]
            elif "l1_encoder" in df:
                if not sel_rows: sel_rows = [sel_row]
                sel_rows = sorted(sel_rows)
                pnames = []
                for s_row in sel_rows:
                    pname1 = df.iloc[s_row]["l1_encoder"]
                    pname2 = df.iloc[s_row]["l1_decoder"]
                    pname3 = df.iloc[s_row]["cossim_encoder"]
                    pname4 = df.iloc[s_row]["cossim_decoder"]
                    images = [Image.open(_f) for _f in [pname1, pname2,pname3, pname4]]
                    new_im = combine_y(images)
                    name = "temp_" + str(s_row) 
                    folder = os.path.join(base_dir, "images")
                    Path(folder).mkdir(exist_ok=True, parents=True)
                    pname = os.path.join(folder, name + ".png")
                    draw = ImageDraw.Draw(new_im)
                    draw.text((0, 0), str(s_row) + "  " + df.iloc[s_row]["template"] +  
                                     " " + df.iloc[s_row]["model"] ,(20,25,255),font=font)
                    new_im.save(pname)
                    pnames.append(pname)
                if len(pnames) == 1:
                    pname = pnames[0]
                    sel_rows = []
                else:
                    images = [Image.open(_f) for _f in pnames]
                    new_im = combine_x(images)
                    name = "temp" 
                    folder = os.path.join(base_dir, "images")
                    Path(folder).mkdir(exist_ok=True, parents=True)
                    pname = os.path.join(folder, name + ".png")
                    new_im.save(pname)
            if "ahmad" in home:
                subprocess.run(["eog", pname])
        elif char in ["o","O"] and prev_char=="x":
            files = [Path(f).stem for f in glob(base_dir+"/*.tsv")]
            for i,f in enumerate(files):
                if f in open_dfnames:
                    files[i] = "** " + f

            canceled, _file = list_values(files)
            if not canceled:
                open_dfnames.append(_file)
                _file = os.path.join(base_dir, _file + ".tsv")
                extra["files"] = open_dfnames
                new_df = pd.read_table(_file)
                if char == "o":
                    df = pd.concat([df, new_df], ignore_index=True)
                else:
                    main_df = pd.concat([main_df, new_df], ignore_index=True)
        elif char == "t" and prev_char=="x":
            cols = get_cols(df,5)
            if cols:
                tdf = df[cols].round(2)
                tdf = tdf.pivot(index=cols[0], columns=cols[1], values =cols[2]) 
                fname = rowinput("Table name:", "table_")
                if fname:
                    if char == "t":
                        tname = os.path.join(base_dir, "plots", fname + ".png")
                        wrate = [col_widths[c] for c in cols]
                        tax = render_mpl_table(tdf, wrate = wrate, col_width=4.0)
                        fig = tax.get_figure()
                        fig.savefig(tname)
                    else:
                        latex = tdf.to_latex(index=False)
                        tname = os.path.join(base_dir, "latex", fname + ".tex")
                        with open(tname, "w") as f:
                            f.write(latex)

        elif char == "P" and prev_char == "x":
            cols = get_cols(df,2)
            if cols:
                df = df.sort_values(cols[1])
                x = cols[0]
                y = cols[1]
                #ax = df.plot.scatter(ax=ax, x=x, y=y)
                ax = sns.regplot(df[x],df[y])
        elif (is_enter(ch) or char == "x") and prev_char == ".":
            backit(df, sel_cols)
            if char == "x":
                consts["sel"] += " MAX"
                score_agg = "max"
            else:
                consts["sel"] += " MEAN"
                score_agg = "mean"
            _agg = {}
            for c in df.columns:
                if c.endswith("score"):
                    _agg[c] = score_agg
                else:
                    _agg[c] = "first"
            df = df.groupby(list(dot_cols.keys())).agg(_agg).reset_index(drop=True)
        elif is_enter(ch) and prev_char == "=":
           backit(df, sel_cols)
           df_conds.sort(key=lambda tup: tup[0])
           df_cond = True
           prev_col = -1
           for col, cond in df_conds:
               if col == prev_col: 
                   df_cond = df_cond | cond
               else:
                   df_cond = df_cond & cond
               prev_col = col
           df = df[df_cond]
           df_conds = [] 
           group_col = ""
           keep_uniques = False
        elif is_enter(ch) or char in ["f"]:
            if is_enter(ch) and context == "filter":
               df = back_df
            else:
                backit(df, sel_cols)
            context = "filter"
            is_filtered = True
            col = sel_cols[cur_col]
            if col == "fid": col = FID
            canceled, col, val = list_df_values(df, col, get_val=True)
            cond = ""
            if not canceled:
               if type(val) == str:
                  cond = f"df['{col}'] == '{val}'"
               else:
                  cond = f"df['{col}'] == {val}"
            mlog.info("cond %s, ", cond)
            if cond:
               df = df[eval(cond)]
               #df = df.reset_index()
               filter_df = df
               if not "filter" in extra:
                  extra["filter"] = []
               extra["filter"].append(cond)
               sel_row = 0
               keep_cols.append(col)
        if char == "V":
            backit(df, sel_cols)
            sel_col = sel_cols[cur_col]
            cond = True 
            for col in orig_tag_cols:
                if not col == sel_col and col in main_df:
                    val=df.iloc[sel_row][col]
                    cond = cond & (main_df[col] == val)
            filter_df = main_df
            df = main_df[cond]
            hotkey = hk
        elif char in ["y","Y"] and False: 
            #yyyyyyyy
           cols = get_cols(df, 2)
           backit(df, sel_cols)
           if cols:
               gcol = cols[0]
               y_col = cols[1]
               if char == "Y":
                   cond = get_cond(df, gcol, 10)
                   df = df[eval(cond)]
               gi = 0 
               name = ""
               for key, grp in df.groupby([gcol]):
                     ax = grp.sort_values('steps').plot.line(ax=ax,linestyle="--",marker="o",  x='steps', y=y_col, label=key, color=colors[gi])
                     gi += 1
                     if gi > len(colors) - 1: gi = 0
                     name += key + "_"
               ax.set_xticks(df["steps"].unique())
               ax.set_title(name)
               if not "filter" in extra:
                   extra["filter"] = []
               extra["filter"].append("group by " + name)
               char = "H"
        if char == "H":
            name = ax.get_title()
            pname = rowinput("Plot name:", name[:30])
            if pname:
                folder = ""
                if "/" in pname:
                    folder, pname = pname.split("/")
                ax.set_title(pname)
                if folder:
                    folder = os.path.join(base_dir, "plots", folder)
                else:
                    folder = os.path.join(base_dir, "plots")
                Path(folder).mkdir(exist_ok=True, parents=True)
                pname = pname.replace(" ", "_")
                pname = os.path.join(folder, now + "_" + pname +  ".png")
                fig = ax.get_figure()
                fig.savefig(pname)
                ax = None
                if "ahmad" in home:
                    subprocess.run(["eog", pname])

        elif char == "r" and prev_char == "x":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                new_name = rowinput(f"Rename {col}:")
                main_df = main_df.rename(columns={col:new_name})
                char = "SS"
                if col in df:
                    df = df.rename(columns={col:new_name})



        elif char in ["d"] and prev_char == "x":
            canceled, col, val = list_df_values(main_df)
            if not canceled:
                main_df = main_df.drop(main_df[main_df[col] == val].index)
                char = "SS"
                info_cols = []
                if col in df:
                    df = df.drop(df[df[col] == val].index)
        elif (ch == cur.KEY_DC and context != "notes"): 
            col = sel_cols[cur_col]
            if col in orig_tag_cols:
                orig_tag_cols.remove(col)
            if col in tag_cols:
                tag_cols.remove(col)
            sel_cols.remove(col)
            save_obj(sel_cols, "sel_cols", context)
        elif ch == cur.KEY_DC and context == "notes":
            df = df.drop(df.iloc[sel_row].name)
            doc_dir = "/home/ahmad/findings" #os.getcwd() 
            note_file = os.path.join(doc_dir, "notes.csv")
            df.to_csv(note_file, index=False)
        elif ch == cur.KEY_SDC:
            #col = sel_cols[cur_col]
            #sel_cols.remove(col)
            if info_cols:
                col = info_cols.pop()
            #save_obj(sel_cols, "sel_cols", context)
            save_obj(info_cols, "info_cols", context)
        elif ch == cur.KEY_SDC and prev_char == 'x':
            col = sel_cols[0]
            val = sel_dict[col]
            cmd = rowinput("Are you sure you want to delete {} == {} ".format(col,val))
            if cmd == "y":
                main_df = main_df.drop(main_df[main_df[col] == val].index)
                char = "SS"
                info_cols = []
                if col in df:
                    df = df.drop(df[df[col] == val].index)
        elif char == "v":
            #do_wrap = not do_wrap
            sel_rows = []
            selected_cols = []
            dot_cols = {}
            keep_cols = []
            consts = {}
            if prev_char == "x":
                info_cols = ["bert_score", "num_preds"]
            if prev_char == "x": 
                sel_cols = ["eid", "rouge_score"] + tag_cols + ["method", "trial", "prefix","num_preds", "bert_score", "pred_max_num","pred_max", "steps","max_acc","best_step", "st_score", "learning_rate",  "num_targets", "num_inps", "train_records", "train_records_nunique", "group_records", "wrap", "frozen", "prefixed"] 
                save_obj(sel_cols, "sel_cols", context)
        elif char == "M" and prev_char == "x":
            info_cols = []
            for col in df.columns:
                info_cols.append(col)
        elif char == "m" and "cfg" in df:
            char = ""
            _,files = get_sel_rows(df, row_id="cfg", col="cfg", from_main=False)
            files = [os.path.join(home, "results", c + ".json") for c in files]
            files.insert(0, "meld")
            subprocess.Popen(files)
        elif char == "m":
            _,dirs = get_sel_rows(df, col="output_dir")
            files = [os.path.join(d, "exp.json") for d in dirs]
            files.insert(0, "meld")
            subprocess.Popen(files)
        elif char == "m" and prev_char == "x":
            info_cols = []
            sel_cols = []
            cond = get_cond(df, "model", 2)
            df = main_df[eval(cond)]
            if df.duplicated(['qid','model']).any():
                show_err("There is duplicated rows for qid and model")
                char = "r"
            else:
                df = df.set_index(['qid','model'])[['pred_text1', 'input_text','prefix']].unstack()
                df.columns = list(map("_".join, df.columns))
        elif is_enter(ch) and prev_char == "x":
            col = sel_cols[0]
            val = sel_dict[col]
            if not "filter" in extra:
                extra["filter"] = []
            extra["filter"].append("{} == {}".format(col,val))
            df = filter_df[filter_df[col] == val]
            df = df.reset_index()
            if char == "F":
                sel_cols = order(sel_cols, [col])
            sel_row = 0
            filter_df = df
        elif char == "w" and prev_cahr == "x":
            sel_rows = []
            adjust = True
            tdf = main_df[main_df['fid'] == sel_exp]
            spath = tdf.iloc[0]["path"]
            tdf.to_csv(spath, sep="\t", index=False)
        elif char == "/":
            old_search = search
            search = rowinput("/", search)
            if search and search == old_search:
                si += 1
            else:
                si = 0
            if search:
                mask = np.column_stack([df[col].astype(str).str.contains(search, na=False) for col in df])
                si = min(si, len(mask) - 1)
                sel_row = df.loc[mask.any(axis=1)].index[si]
        elif char == ":":
            cmd = rowinput() #default=prev_cmd)
        elif char == "q":
            save_df(df)
            # if prev_char != "q": mbeep()
            consts["exit"] = "hit q another time to exit"
            prev_char = "q" # temporary line for exit on one key  #comment me
        if cmd.startswith("cp="):
            _, folder, dest = cmd.split("=")
            spath = main_df.iloc[0]["path"]
            dest = os.path.join(home, "logs", folder, dest)
            Path(folder).mkdir(exist_ok=True, parents=True)
            shutil.copyfile(spath, dest)
        if cmd.startswith("w="):
            _,val = cmd.split("=")[1]
            col = sel_cols[cur_col]
            col_widths[col] = int(val)
            adjust = False
        if cmd.startswith("cc"):
            name = cmd.split("=")[-1]
            if not name in rep_cmp:
                rep_cmp[name] = {}
            exp=df.iloc[sel_row]["eid"]
            tdf = main_df[main_df['eid'] == exp]
            _agg = {}
            for c in sel_cols:
                if c in df.columns: 
                    if c.endswith("score"):
                        _agg[c] = "mean"
                    else:
                        _agg[c] = "first"
            gdf = tdf.groupby(["prefix"], as_index=False).agg(_agg).reset_index(drop=True)
            all_rels = gdf['prefix'].unique()
            for rel in all_rels: 
                cond = (gdf['prefix'] == rel)
                val = gdf.loc[cond, "m_score"].iloc[0]
                val = "{:.2f}".format(val)
                if not rel in rep_cmp[name]:
                    rep_cmp[name][rel] = []
                rep_cmp[name][rel].append(val)
            save_obj(rep_cmp, "rep_cmp", "gtasks")
            char = "r"
        if cmd.startswith("conv"):
            to = ""
            if "@" in cmd:
                to = cmd.split("@")[1]
            col = sel_cols[cur_col]
            if to == "num":
                df[col] = df[col].astype(float)
        if char == "x" or cmd.startswith("cross"):
            backit(df, sel_cols)
            eid = df.iloc[sel_row]['eid'] 

            if context == "pivot" or len(sel_rows) > 1:
                prefix = sel_cols[cur_col]
                exprs, scores = get_sel_rows(df, col=prefix, from_main=False) 
                _, mask_types = get_sel_rows(df, col="mask_type", from_main=False) 
                _, labels = get_sel_rows(df, col="label", from_main=False) 
            else:
                prefix = df.iloc[sel_row]['prefix'] 
                exprs = [eid]
                scores = [prefix]
                mask_types = [df.iloc[sel_row]['mask_type']] 
                labels = [df.iloc[sel_row]['label']] 


            info_cols.append(prefix)
            consts["prefix"] = prefix
            _cols = ["pred_text1", "target_text"]
            dfs = []
            for eid, acc, mt, label in zip(exprs, scores, mask_types, labels):
                tdf = main_df.loc[(main_df.eid == eid) & (main_df.prefix == prefix), _cols]
                canceled, val = False, "pred_text1" # list_values(sel_cols)
                if not canceled:
                    treatment = 'target_text' #sel_cols[cur_col]
                    tdf = pd.crosstab(tdf[val], tdf[treatment])
                tdf["preds"] = list(tdf.axes[0])
                tdf["acc"] = acc
                tdf["eid"] = eid
                tdf["label"] = label
                tdf["mask_type"] = mt 
                tdf["uid"] = mt + " " + str(label) + " " + str(eid)
                dfs.append(tdf)
            df = pd.concat(dfs, ignore_index=True)
            all_sel_cols = ["preds"] + list(df.columns)
            sel_cols = all_sel_cols[:20] 
            for col in sel_cols:
               col_widths[col] = len(col) + 2
            #adjust = False
            left = 0
            sel_rows= []
            context = "cross"
            group_col = "uid"
        if cmd == "line" or (char == "h" and context == "pivot"):
             try:
                 cur_col_name = sel_cols[cur_col]
                 if len(selected_cols) == 0:
                     selected_cols = [sel_cols[cur_col]]
                 if len(selected_cols) == 1 and cur_col_name not in selected_cols:
                     selected_cols.append(cur_col_name)
                 if len(selected_cols) == 2:
                     df.plot.line(x=selected_cols[0], y=selected_cols[1])
                 elif len(selected_cols) > 0:
                     tdf = df[selected_cols]
                     tdf.plot.line(subplots=True)
                 plt.show()
             except Exception as e:
                 show_msg("Error:" + repr(e))
                 mbeep()
        if cmd.startswith("anova"):
            to = ""
            canceled, val = False, "pred_text1" # list_values(sel_cols)
            if not canceled:
                treatment = 'target_text' #sel_cols[cur_col]
                df[val] = df[val].astype(float)
                ax = sns.boxplot(x=treatment, y=val, data=df, color='#99c2a2')
                ax = sns.swarmplot(x=treatment, y=val, data=df, color='#7d0013')
                plt.show()
                # Ordinary Least Squares (OLS) model
                model = ols(f'{val} ~ C({treatment})', data=df).fit()
                backit(df, sel_cols)
                sel_cols = ["sum_sq","df","F","PR(>F)"]
                df = sm.stats.anova_lm(model, typ=2)
        if cmd.startswith("banova"):
            to = ""
            canceled, val = False, "target_text" # list_values(sel_cols)
            if not canceled:
                treatment = 'pred_text1' #sel_cols[cur_col]
                df[val] = df[val].astype(float)
                ax = sns.boxplot(x=treatment, y=val, data=df, color='#99c2a2')
                ax = sns.swarmplot(x=treatment, y=val, data=df, color='#7d0013')
                plt.show()
                # Ordinary Least Squares (OLS) model
                model = ols(f'{val} ~ C({treatment})', data=df).fit()
                backit(df, sel_cols)
                sel_cols = ["sum_sq","df","F","PR(>F)"]
                df = sm.stats.anova_lm(model, typ=2)
        if cmd.startswith("rem"):
            if "@" in cmd:
                exp_names = cmd.split("@")[2:]
                _from = cmd.split("@")[1]
            rep = load_obj(_from, "gtasks", {})
            if rep:
                for k, cat in rep.items():
                    for exp in exp_names:
                        if exp in cat:
                            del rep[k][exp]
            save_obj(rep, _from, "gtasks")
        if char in ["1","2"]:
            if not pcols or not all(col in df for col in pcols):
                mbeep()
            else:
                if prev_char not in ["1","2"]:
                    backit(df, sel_cols)
                if score_cols:
                    ss = int(char) - 1
                    if ss < len(score_cols):
                        score_col = score_cols[ss]
                sel_cols = index_cols.copy() + ["All"] # + main_vars
                sort_col = sort if sort else "All"
                if score_col == score_cols[0]:
                    for col in df.columns:
                        if col in pcols:
                            sel_cols.append(col)
                else:
                    for col in df.columns:
                        if col.startswith(score_col[0] + "-"):
                            sel_cols.append(col)
                    sort_col = score_col[0] + "-All"
                    if sort_col in sel_cols:
                        sel_cols.remove(sort_col)
                        sel_cols.insert(len(index_cols), sort_col)
                sort = sort_col
                # df = df.sort_values(by=sort_col, ascending=False)
        if char == "!":
            doc_dir = "/home/ahmad/findings" #os.getcwd() 
            note_file = os.path.join(doc_dir, "notes.csv")
            context = "notes"
            if not "comment" in df:
                backit(df, sel_cols)
                if Path(note_file).is_file():
                    df = pd.read_csv(note_file)
                else:
                    df = pd.DataFrame(columns=["date","cat","comment"])
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                sel_cols = df.columns 
                info_cols = []
                df = df.sort_values("date", ascending=False)
                cond_colors["cat"] = cat_colors
                # search_df = df
        if ch == cur.KEY_IC or char == "e" and context == "notes":
            doc_dir = "/home/ahmad/findings" #os.getcwd() 
            note_file = os.path.join(doc_dir, "notes.csv")
            if not "comment" in df or context != "notes":
                if Path(note_file).is_file():
                    tdf = pd.read_csv(note_file)
                else:
                    tdf = pd.DataFrame(columns=["date","cat","comment"])
                tdf = tdf.loc[:, ~tdf.columns.str.contains('^Unnamed')]
            else:
                tdf = df
            bg_color = HL_COLOR
            win_height = 8
            note_title = ""
            _default = ""
            cat = ""
            if char == "e" and len(df) > 0:
                cat = df.iloc[sel_row]["cat"] 
                _default = cat + "\n" + df.iloc[sel_row]["comment"]
            _comment, ret_ch = biginput("", default=_default)
            if _comment:
                lines = _comment.split("\n")
                comment = _comment
                if len(lines) > 1:
                    cat = lines[0]
                    comment = "\n".join(lines[1:]) 
                new_note = {}
                new_note["date"] = now
                new_note["comment"] = comment
                new_note["cat"] = cat
                if char != "e":
                    tdf = pd.concat([tdf, pd.DataFrame([new_note])], ignore_index=True)
                else:
                    tdf.iloc[sel_row] = new_note 
                if Path(note_file).is_file():
                    shutil.copyfile(note_file, note_file.replace("notes", now + "_notes"))
                tdf.to_csv(note_file)
            if "comment" in df:
                df = tdf
                df = df.sort_values(by=["date"], ascending=False) 
                cond_colors["cat"] = cat_colors
        # rrrrrrrrr
        if cmd.startswith("rep") or char == "Z" or char == "r": 
            mdf = df #main_df
            _agg = {}
            _rep_cols = []
            for c in rep_cols: 
                if c in score_cols: # or c in ["d_seed", "max_train_samples"]: 
                    _agg[c] = "mean"
                elif (c.endswith("score") 
                        or (char == "Z" and (c.endswith("_num") or c.startswith("num_")))): 
                    score_cols.append(c)
                    _agg[c] = "mean"
                elif c not in pivot_cols:
                    _rep_cols.append(c)
                    _agg[c] = "first"
            gcol = _rep_cols
            if not "eid" in rep_cols:
                gcol += ["eid"] 
            mdf[gcol] = mdf[gcol].fillna('none')
            pdf = mdf.pivot_table(index=gcol, columns=pivot_cols, 
                    values=score_cols, aggfunc='mean', margins=True)
            columns = pdf.columns.to_flat_index()
            # pdf["avg"] = pdf.mean(axis=1, skipna=True)
            #pdf['fid'] = mdf.groupby(gcol)['fid'].first()
            # pdf['eid'] = mdf.groupby(gcol)['eid'].first()
            #pdf['cat'] = mdf.groupby(gcol)['cat'].first()
            pdf.reset_index(inplace=True)
            pdf.columns = [col[1] if col[0] == score_cols[0] 
                    else col[0][0] + "-" + col[1] if col[0] in score_cols else col[0]
                    for col in pdf.columns]
            # pdf['cat'] = pdf['cat'].apply(lambda x: x.split('-')[0]) 
            pdf['label'] = pdf.apply(create_label, axis=1)
            pdf['ref'] = pdf.apply(
                    lambda row: f" \\ref{{{'fig:' + str(row['eid'])}}}", axis=1)
            pdf = pdf.round(2)
            #latex_table=tabulate(pdf, #[rep_cols+score_cols], 
            #        headers='keys', tablefmt='latex_raw', showindex=False)
            #latex_table = latex_table.replace("tabular", "longtable")
            #report = report.replace("mytable", latex_table + "\n\n \\newpage mytable")

            avg_col = "All"
            backit(df, sel_cols)
            context = "pivot"
            shortkeys["pivot"] = {"o":"open image", "h": "plot line"}
            score_col = score_cols[0]
            pcols = []
            cond_colors["eid"] = time_colors
            cond_colors["All"] = score_colors
            cond_colors["time"] = time_colors
            cond_colors["expid"] = index_colors
            for col in pivot_cols:
                pcols.extend(df[col].unique())
            for col in pcols:
                cond_colors[col] = pivot_colors

            _sel_cols = [] 
            if score_col == score_cols[0]:
                avg_col = "All"
                for col in pdf.columns:
                    if col in df or col in pcols:
                        _sel_cols.append(col)
            else:
                avg_col = score_col[0] + "-All"
                for col in pdf.columns:
                    if col in df or col.startswith(score_col[0] + "-"):
                        _sel_cols.append(col)
            df = pdf.iloc[:-1]
            df = df.sort_values(by="time", ascending=False)
            sort = "time"
            sel_cols = list(dict.fromkeys(sel_cols + _sel_cols))
            if len(df) > 1:
                sel_cols, info_cols_back, tag_cols = remove_uniques(df, sel_cols, 
                        keep_cols=pivot_cols + info_cols + pcols)
            for col in ["folder", "output_dir"]:
                if col in sel_cols:
                    sel_cols.remove(col)
                if not col in info_cols_back:
                    info_cols_back.append(col)

            for i, col in enumerate(info_cols + [avg_col]):
                if col in sel_cols:
                    sel_cols.remove(col)
                sel_cols.insert(i, col)
            if "time" in sel_cols:
                sel_cols.remove("time")
                sel_cols.append("time")

            pivot_df = df
            info_cols = []
            #df.columns = [map_cols[col].replace("_","-") if col in map_cols else col 
            #              for col in pdf.columns]
        if char == "l" or char == "Z" or cmd.startswith("rep"):
            _dir = Path(__file__).parent
            doc_dir = "/home/ahmad/logs" #os.getcwd() 
            if len(score_cols) > 1:
                # m_report = f"{_dir}/report_templates/report_colored_template.tex"
                m_report = f"{_dir}/report_templates/report_template.tex"
            else:
                m_report = f"{_dir}/report_templates/report_template.tex"
            with open(m_report, "r") as f:
                report = f.read()
            with open(os.path.join(doc_dir, "report.tex"), "w") as f:
                f.write("")
            latex_table=tabulate(df[sel_cols],  #[rep_cols+score_cols], 
                    headers='keys', tablefmt='latex_raw', showindex=False)
            latex_table = latex_table.replace("tabular", "longtable")
            latex_table = latex_table.replace("_", "-")
            report = report.replace("mytable", latex_table + "\n\n \\newpage mytable")
            report = report.replace("mytable", "\n \\newpage mytable")
            # df = pdf
            # iiiiiiiiiiiii
            report = report.replace("mytable","")
            tex = f"{doc_dir}/report.tex"
            pdf = f"{doc_dir}/report.pdf"
            with open(tex, "w") as f:
                f.write(report)
            #with open(m_report, "w") as f:
            #    f.write(main_report)
            mbeep()
            #subprocess.run(["pdflatex", tex])
            #subprocess.run(["okular", pdf])
        if "getimage" in cmd or char == "Z":
            show_msg("Generating images ...", bottom=True, delay=2000)
            _dir = Path(__file__).parent
            doc_dir = "/home/ahmad/logs" #os.getcwd() 
            m_report = os.path.join(doc_dir, "report.tex")
            with open(m_report, "r") as f:
                report = f.read()

            image = """
                \\begin{{figure}}
                    \centering
                    \includegraphics[width=\\textwidth]{{{}}}
                    \caption[image]{{{}}}
                    \label{{{}}}
                \end{{figure}}
            """
            multi_image = """
                \\begin{figure}
                    \centering
                    \caption[image]{mycaption}
                    mypicture 
                    \label{fig:all}
                \end{figure}
            """
            graphic = "\includegraphics[width=\\textwidth]{{{}}}"
            pics_dir = doc_dir + "/pics"
            #ii = image.format(havg, "havg", "fig:havg")
            #report = report.replace("myimage", ii +"\n\n" + "myimage")
            Path(pics_dir).mkdir(parents=True, exist_ok=True)
            #pname = plot_bar(pics_dir, train_num)
            #ii = image.format(pname, "bar", "fig:bar")
            #report = report.replace("myimage", ii +"\n\n" + "myimage")
            all_exps = df["eid"].unique()
            experiment_images, fnames = get_images(df, all_exps, 'eid')
            all_images = {}
            kk = 0
            id = "other"
            images_str = ""
            cols = ["eid"] + rep_cols + score_cols
            img_string = ""
            for key, img_list in experiment_images.items():
                mkey = key
                caption_dict = {}
                if not df.loc[df['eid'] == key].empty:
                    caption_dict = df.loc[df['eid'] == key, sel_cols].iloc[0].to_dict()
                caption = ""
                name = key
                key = str(key)
                for new_im in img_list:
                    name = key + str(name)
                    _exp = key.replace("_","-")
                    _exp = _exp.split("-")[0]
                    fname = fnames[kk]
                    for k,v in caption_dict.items():
                        if k in map_cols:
                            k = map_cols[k]
                        if type(v) == float:
                            v = f"{v:.2f}"
                        if k == "cat":
                            v = v.split("-")[0]
                        caption += " \\textcolor{gray}{" + str(k).replace("_","-") \
                            + "}: \\textcolor{blue}{" + str(v).replace("_","-")+ "}" 
                    ss = "_scores" if "score" in fname else "_sim"
                    if "@" in fname:
                        ss = "_" + fname.split("@")[1]
                    pname = doc_dir + "/pics/" + id + name.strip("-") + ss + ".png" 
                    dest = os.path.join(doc_dir, pname) 
                    new_im.save(dest)
                    label = "fig:" + key
                    ii = image.format(pname, caption, label)
                    if kk % 10 == 0:
                        tex = f"{doc_dir}/hm_img_{kk}.tex"
                        with open(tex, "w") as f:
                            f.write(img_string)
                        img_string = ""
                        report = report.replace("myimage", 
                                f"\clearpage \n \input{{hm_img_{kk}.tex}} \n myimage") 
                    img_string +=  ii +"\n\n" 
                    if not _exp in all_images:
                        all_images[_exp] = {}
                    all_images[_exp][ss] = pname
                    kk += 1

            if img_string: 
                tex = f"{doc_dir}/hm_img_{kk}.tex"
                with open(tex, "w") as f:
                    f.write(img_string)
                img_string = ""
                report = report.replace("myimage", 
                        f"\clearpage \n \input{{hm_img_{kk}.tex}} \n myimage") 

            g1 = ["SIL","SILP","SIP"] 
            g2 = ["SILPI","SLPI","SLP", "SL"] 
            ii = 0
            multi_image3 = multi_image
            for k,v in all_images.items():
                if ii % 2 == 0:
                    multi_image3 += f" \\newpage \n \\subsection{{{k}}}"
                ii += 1
                for p,q in v.items():
                    multi_image3 += multi_image.replace("mypicture", 
                            graphic.format(q) + "\n").replace("mycaption",
                                    str(p) + ":" + str(q))

            multi_image3 = multi_image3.replace("mypicture","")
            tex = f"{doc_dir}/scores_img.tex"
            with open(tex, "w") as f:
                f.write(multi_image3)
            #report = report.replace("myimage", 
            #        "\n\n \input{scores_img.tex} \n\n myimage") 
            #report = report.replace("myimage", "\n\n \input{sim_img.tex} \n\n myimage") 
            #report = report.replace("myimage", "\n\n \input{other_img.tex} \n\n") 
            ####################
            report = report.replace("mytable","")
            report = report.replace("myimage","")
            tex = f"{doc_dir}/report.tex"
            pdf = f"{doc_dir}/report.pdf"
            with open(tex, "w") as f:
                f.write(report)
            #with open(m_report, "w") as f:
            #    f.write(main_report)
            show_msg(pdf)
            mbeep()
            #subprocess.run(["pdflatex", tex])
            subprocess.run(["okular", pdf])

        if cmd == "fix_types":
            for col in ["target_text", "pred_text1"]: 
                main_df[col] = main_df[col].astype(str)
                for col in ["steps", "epochs", "val_steps"]: 
                    main_df[col] = main_df[col].astype(int)
                char = "SS"
        if cmd == "clean":
            main_df = main_df.replace(r'\n',' ', regex=True)
            char = "SS"
        if cmd == "fix_template":
            main_df.loc[(df["template"] == "unsup-tokens") & 
                    (main_df["wrap"] == "wrapped-lstm"), "template"] = "unsup-tokens-wrap"
            main_df.loc[(main_df["template"] == "sup-tokens") & 
                    (main_df["wrap"] == "wrapped-lstm"), "template"] = "sup-tokens-wrap"
        
        if cmd == "ren":
            sel_col = sel_cols[cur_col]
            new_name = rowinput("Rename " + sel_col + " to:", default="")
            map_cols[sel_col] = new_name
            save_obj(map_cols, "map_cols", "atomic")
            cur_col += 1
        if cmd == "copy" or char == "\\":
            exp=df.iloc[sel_row]["eid"]
            exp = str(exp)
            spath = tdf.iloc[0]["path"]
            oldpath = Path(spath).parent.parent
            pred_file = os.path.join(oldpath, "images", "pred_router_" + exp + ".png") 
            oldpath = os.path.join(oldpath, exp)
            newpath = rowinput(f"copy {oldpath} to:", default=oldpath)
            new_pred_file = os.path.join(newpath, "images", "pred_router_" + exp + ".png") 
            shutil.copyfile(pred_file, new_pred_file)
            copy_tree(oldpath, newpath)
        if cmd == "repall":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                _a = rowinput("from")
                _b = rowinput("to")
                main_df[col] = main_df[col].str.replace(_a,_b)
                char = "SS"
        if cmd == "replace" or cmd == "replace@":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                vals = df[col].unique()
                d = {}
                for v in vals:
                    rep = rowinput(str(v) + "=" ,v)
                    if not rep:
                        break
                    if type(v) == int:
                        d[v] = int(rep)
                    else:
                        d[v] = rep
                if rowinput("Apply?") == "y":
                    if "@" in cmd:
                        df = df.replace(d)
                    else:
                        df = df.replace(d)
                        main_df = main_df.replace(d)
                        char = "SS"
        if cmd in ["set", "set@", "add", "add@", "setcond"]:
            if "add" in cmd:
                col = rowinput("New col name:")
            col = sel_cols[cur_col]
            cond = ""
            if "cond" in cmd:
                cond = get_cond(df, for_col=col, num=5, op="&")
            if cond:
                val = rowinput(f"Set {col} under {cond} to:")
            else:
                val = rowinput("Set " + col + " to:")
            if val:
                if cond:
                    if "@" in cmd:
                        main_df.loc[eval(cond), col] = val
                        char = "SS"
                    else:
                        df.loc[eval(cond), col] =val
                else:
                    if "@" in cmd:
                        main_df[col] =val
                        char = "SS"
                    else:
                        df[col] = val
        if ":=" in cmd:
            var, val = cmd.split(":=")
            settings[var] = val
            save_obj(settings, "settings", "gtasks")
        elif "==" in cmd:
            col, val = cmd.split("==")
            df = df[df[col] == val]
        elif "top@" in cmd:
            backit(df, sel_cols)
            tresh = float(cmd.split("@")[1])
            df = df[df["bert_score"] > tresh]
            df = df[["prefix","input_text","target_text", "pred_text1"]] 
        if cmd == "cp" or cmd == "cp@":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                copy = rowinput("Copy " + col + " to:", col)
                if copy:
                    if "@" in cmd:
                        df[copy] = df[col]
                    else:
                        main_df[copy] = main_df[col]
                        char = "SS"
        if cmd.isnumeric():
            sel_row = int(cmd)
        elif cmd == "q" or cmd == "wq":
            save_df(df)
            prev_char = "q" 
        elif not char in ["q", "S","R"]:
            pass
            #mbeep()
        if char in ["S", "}"]:
            _name = "main_df" if char == "S" else "df"
            _dfname = dfname
            if dfname == "merged":
                _dfname = "test"
            cmd, _ = minput(cmd_win, 0, 1, f"File Name for {_name} (without extension)=", default=_dfname, all_chars=True)
            cmd = cmd.split(".")[0]
            if cmd != "<ESC>":
                if char == "}":
                    df.to_csv(os.path.join(base_dir, cmd+".tsv"), sep="\t", index=False)
                else:
                    dfname = cmd
                    char = "SS"
        if char == "SS":
                df = main_df[["prefix","input_text","target_text"]]
                df = df.groupby(['input_text','prefix','target_text'],as_index=False).first()

                save_path = os.path.join(base_dir, dfname+".tsv")
                sel_cols = ["prefix", "input_text", "target_text"]
                Path(base_dir).mkdir(parents = True, exist_ok=True)
                df.to_csv(save_path, sep="\t", index=False)

                save_obj(dfname, "dfname", dfname)
        if char == "R" and prev_char != "x":
            filter_df = orig_df
            df = filter_df
            FID = "fid" 
            reset = True
            #sel_cols = group_sel_cols 
            #save_obj([], "sel_cols", context)
            #save_obj([], "info_cols", context)
            hotkey = hk
        if char == "R" and prev_char == "x":
            df = main_df
            sel_cols = list(df.columns)
            save_obj(sel_cols,"sel_cols",dfname)
            extra["filter"] = []
            info_cols = []
        if (ch == cur.KEY_BACKSPACE or char == "b") and back:
            if back:
                cur_df = back.pop()
                df = cur_df.df
                sel_cols = cur_df.sel_cols 
                sel_row = cur_df.sel_row
                sel_rows = cur_df.sel_rows
                info_cols = cur_df.info_cols
                context = cur_df.context
                cur_col = cur_df.cur_col
                left = cur_df.left
                group_col = cur_df.group_col
                if back:
                    back_df = back[-1].df
                else:
                    if "b" in general_keys:
                        del general_keys["b"]
            else:
                if "b" in general_keys:
                    del general_keys["b"]
                mbeep()
            if extra["filter"]:
                extra["filter"].pop()

def render_mpl_table(data, wrate, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        mlog.info("Size %s", size)
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    mpl_table.auto_set_column_width(col=list(range(len(data.columns)))) # Provide integer list of columns to adjust

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax
def get_cond(df, for_col = "", num = 1, op="|"):
    canceled = False
    sels = []
    cond = ""
    while not canceled and len(sels) < num:
        canceled, col, val = list_df_values(df, col=for_col, get_val=True,sels=sels)
        if not canceled:
            cond += f"{op} (df['{col}'] == '{val}') "
            sels.append(val)
    cond = cond.strip(op)
    return cond

def get_cols(df, num = 1):
    canceled = False
    sels = []
    while not canceled and len(sels) < num:
        canceled, col,_ = list_df_values(df, get_val=False, sels = sels)
        if not canceled:
            sels.append(col)
    return sels

def biginput(prompt=":", default=""):
    rows, cols = std.getmaxyx()
    win = cur.newwin(12, cols, 5, 0)
    win.bkgd(' ', cur.color_pair(CUR_ITEM_COLOR))  # | cur.A_REVERSE)
    _comment, ret_ch = minput(win, 0, 0, "Enter text", 
            default=default, mode =MULTI_LINE)
    if _comment == "<ESC>":
        _comment = ""
    return _comment, ret_ch

def rowinput(prompt=":", default=""):
    prompt = str(prompt)
    default = str(default)
    ch = UP
    history = load_obj("history","", ["test1", "test2", "test3"])
    ii = len(history) - 1
    hh = history.copy()
    while ch == UP or ch == DOWN:
        cmd, ch = minput(cmd_win, 0, 1, prompt, default=default, all_chars=True)
        if ch == UP:
            if cmd != "" and cmd != default:
                jj = ii -1
                while jj > 0: 
                    if hh[jj].startswith(cmd):
                      ii = jj
                      break
                    jj -= 1
            elif ii > 0: 
                ii -= 1 
            else: 
                ii = 0
                mbeep()
        elif ch == DOWN:
            if cmd != "" and cmd != default:
                jj = ii + 1
                while jj < len(hh) - 1: 
                    if hh[jj].startswith(cmd):
                      ii = jj
                      break
                    jj += 1
            elif ii < len(hh) - 1: 
               ii += 1 
            else:
               ii = len(hh) - 1
               mbeep()
        if hh:
            ii = max(ii, 0)
            ii = min(ii, len(hh) -1)
            default = hh[ii]
    if cmd == "<ESC>":
        cmd = ""
    if cmd:
        history.append(cmd)
    save_obj(history, "history", "")
    return cmd

def order(sel_cols, cols, pos=0):
    z = [item for item in sel_cols if item not in cols] 
    z[pos:pos] = cols
    save_obj(z, "sel_cols",dfname)
    return z

def subwin(infos):
    ii = 0
    infos.append("[OK]")
    inf = infos[ii:ii+30]
    change_info(inf)
    cc = std.getch()
    while not is_enter(cc): 
        if cc == DOWN:
            ii += 1
        if cc == UP:
            ii -= 1
        if cc == cur.KEY_NPAGE:
            ii += 10
        if cc == cur.KEY_PPAGE:
            ii -= 10
        if cc == cur.KEY_HOME:
            ii = 0
        if cc == cur.KEY_END:
            ii = len(infos) - 20 
        ii = max(ii, 0)
        ii = min(ii, len(infos)-10)
        inf = infos[ii:ii+30]
        change_info(inf)
        cc = std.getch()
                
def change_info(infos):
    info_bar.erase()
    h,w = info_bar.getmaxyx()
    w = 80
    lnum = 0
    for msg in infos:
        lines = textwrap.wrap(msg, width=w, placeholder=".")
        for line in lines: 
            mprint(str(line).replace("@","   "), info_bar, color=HL_COLOR)
            lnum += 1
    rows,cols = std.getmaxyx()
    info_bar.refresh(0,0, rows -lnum - 1,0, rows-1, cols - 2)
si_hash = {}

def list_values(vals,si=0, sels=[], is_combo=False):
    tag_win = cur.newwin(15, 70, 3, 5)
    tag_win.bkgd(' ', cur.color_pair(TEXT_COLOR))  # | cur.A_REVERSE)
    tag_win.border()
    vals = sorted(vals)
    key = "_".join([str(x) for x in vals[:4]])
    if si == 0:
        if key in si_hash:
            si = si_hash[key]
    opts = {"items":{"sels":sels, "range":["Done!"] + vals}}
    if is_combo: opts["items"]["type"] = "combo-box"
    is_cancled = True
    si,canceled, st = open_submenu(tag_win, opts, "items", si, "Select a value", std)
    val = st
    if not canceled and si > 0: 
        val = vals[si - 1]
        si_hash[key] = si
        is_cancled = False
    return is_cancled, val

def list_df_values(df, col ="", get_val=True,si=0,vi=0, sels=[], extra=[]):
    is_cancled = False
    if not col:
        cols = extra + list(df.columns) 
        is_cancled, col = list_values(cols,si, sels)
    val = ""
    if col in df and col and get_val and not is_cancled:
        df[col] = df[col].astype(str)
        vals = sorted(list(df[col].unique()))
        is_cancled, val = list_values(vals,vi, sels)
    return is_cancled, col, val 


text_win = None
info_bar = None
cmd_win = None
main_win = None
text_width = 60
std = None
check_time = False
hotkey = ""
dfname = ""
global_cmd = ""
global_search = ""
base_dir = os.path.join(home, "mt5-comet", "comet", "data", "atomic2020")
data_frame = None

def get_files(dfpath, dfname, dftype, limit, file_id):
    if not dfname:
        mlog.info("No file name provided")
    else:
        path = os.path.join(dfpath, *dfname)
        if Path(path).is_file():
            files = [path]
            dfname = Path(dfname).stem
        else:
            files = []
            ii = 0
            for root, dirs, _files in os.walk(dfpath):
                for _file in _files:
                    root_file = os.path.join(root,_file)
                    cond = all(s.strip() in root_file for s in dfname)
                    if check_time:
                        ts = os.path.getctime(root_file)
                        ctime = datetime.fromtimestamp(ts)
                        last_hour = datetime.now() - timedelta(hours = 5)
                        cond = cond and ctime > last_hour
                    if dftype in _file and cond: 
                        files.append(root_file)
                        ii += 1
                if limit > 0 and ii > limit:
                    break
        # mlog.info("files: %s",files)
        if not files:
            print("No file was selected")
            return
        dfs = []
        ii = 0
        ff = 0
        folders = {}
        for f in tqdm(files):
            if f.endswith(".tsv"):
                df = pd.read_table(f, low_memory=False)
            elif f.endswith(".json"):
                df = load_results(f)
            force_fid = False
            sfid = file_id.split("@")
            fid = sfid[0]
            if global_search: 
                col = "pred_text1"
                val = global_search
                if "@" in global_search:
                    val, col = global_search.split("@")
                values = df[col].unique()
                if val in values:
                    print("path:", f)
                    print("values:", values)
                    assert False, "found!" + f
                continue
            if len(sfid) > 1:
                force_fid = sfid[1] == "force"
            if True: #force_fid:
                df["path"] = f
                df["fid"] = ii
                _dir = str(Path(f).parent)
                folder = str(Path(f).parent) 
                if not folder in folders:
                    folders[folder] = ff
                    ff += 1
                df["folder"] = folder 
                df["eid"] = folders[folder]
                _pp = _dir + "/*.png"
                png_files = glob(_pp)
                if not png_files:
                    _pp = str(Path(_dir).parent) + "/hf*/*.png"
                    png_files = glob(_pp)
                for i,png in enumerate(png_files):
                    key = Path(png).stem
                    if not key in df:
                       df[key] = png
                if fid == "parent":
                    _ff = "@".join(f.split("/")[5:]) 
                    df["exp_name"] = Path(f).parent.stem #.replace("=","+").replace("_","+")
                else:
                    df["exp_name"] =  "_" + Path(f).stem
            dfs.append(df)
            ii += 1
        if len(dfs) > 0:
            df = pd.concat(dfs, ignore_index=True)
            return df
        return None

def start(stdscr):
    global info_bar, text_win, cmd_win, std, main_win, colors, dfname, STD_ROWS, STD_COLS
    stdscr.refresh()
    std = stdscr
    now = datetime.now()
    rows, cols = std.getmaxyx()
    set_max_rows_cols(rows, cols) 
    height = rows - 1
    width = cols
    # mouse = cur.mousemask(cur.ALL_MOUSE_EVENTS)
    text_win = cur.newpad(rows * 50, cols * 20)
    text_win.bkgd(' ', cur.color_pair(TEXT_COLOR)) # | cur.A_REVERSE)
    cmd_win = cur.newwin(1, cols, rows - 1, 0)

    info_bar = cur.newpad(rows*10, cols*20)
    info_bar.bkgd(' ', cur.color_pair(HL_COLOR)) # | cur.A_REVERSE)

    cur.start_color()
    cur.curs_set(0)
    # std.keypad(1)
    cur.use_default_colors()

    colors = [str(y) for y in range(-1, cur.COLORS)]
    if cur.COLORS > 100:
        colors = [str(y) for y in range(-1, 100)] + [str(y) for y in range(107, cur.COLORS)]


    theme = {'preset': 'default', "sep1": "colors", 'text-color': '247', 'back-color': '233', 'item-color': '71','cur-item-color': '251', 'sel-item-color': '33', 'title-color': '28', "sep2": "reading mode",           "dim-color": '241', 'bright-color':"251", "highlight-color": '236', "hl-text-color": "250", "inverse-highlight": "True", "bold-highlight": "True", "bold-text": "False", "input-color":"234", "sep5": "Feedback Colors"}

    reset_colors(theme)
    show_df(data_frame)


@click.command(context_settings=dict(
            ignore_unknown_options=True,
            allow_extra_args=True,))
@click.argument("fname", nargs=-1, type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--fid",
    "-fid",
    default="parent",
    type=str,
    help=""
)
@click.option(
    "--ftype",
    "-ft",
    default="tsv",
    type=str,
    help=""
)
@click.option(
    "--dpy",
    "-d",
    is_flag=True,
    help=""
)
@click.option(
    "--chk_time",
    "-t",
    is_flag=True,
    help=""
)
@click.option(
    "--hkey",
    "-h",
    default="CG",
    type=str,
    help=""
)
@click.option(
    "--cmd",
    "-c",
    default="",
    type=str,
    help=""
)
@click.option(
    "--search",
    "-s",
    default="",
    type=str,
    help=""
)
@click.option(
    "--limit",
    "-l",
    default=-1,
    type=int,
    help="Limit of datasets to load"
)
@click.pass_context
def main(ctx, fname, path, fid, ftype, dpy, hkey, cmd, search, limit, chk_time):
    if dpy:
        port = 1234
        debugpy.listen(('0.0.0.0', int(port)))
        print("Waiting for client at run...port:", port)
        debugpy.wait_for_client()  # blocks execution until client is attached
    global dfname, hotkey, global_cmd, global_search,check_time, data_frame
    check_time = chk_time
    global_search = search
    hotkey = hkey 
    global_cmd = cmd
    dfname = fname
    if not fname:
        fname = [ftype]
    set_app("showdf")
    data_frame = get_files(path, fname, ftype, limit=limit, file_id= fid)
    if data_frame is not None:
        dfname = "merged"
        wrapper(start)
    else:
        mlog.info("No tsv or json file was found")

if __name__ == "__main__":
    main()
