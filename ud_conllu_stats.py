from multiprocessing import Pool, cpu_count

import copy
import gzip
import math
import os
import sys
import orjson as json
# import json

from pathlib import Path

import pyconll
import pyconll.util
from pycountry import languages


from utils import *
from preprocess_conllu import *

import pandas as pd
import numpy as np
from scipy import stats

import bokeh
from bokeh.plotting import figure, show
# from bokeh.palettes import Spectral4
# from bokeh.io import output_file
from bokeh.models import LinearAxis, Range1d, HoverTool, ColumnDataSource, DataTable, TableColumn, Label
from bokeh.models.layouts import Column, Panel, Tabs
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import gridplot, column, row, Spacer
from bokeh.resources import CDN
from bokeh.embed import components, file_html, json_item, autoload_static


UD_VERSION = "2.6"
BASEPATH = ""
CONLLU_BASEPATH = os.path.join(BASEPATH, 'UniversalDependencies/ud-treebanks-v{}'.format(UD_VERSION))

#
DISTRIBUTIONS = {"norm": stats.norm,
                 "skewnorm": stats.skewnorm,
                 "gennorm": stats.gennorm,
                 "beta": stats.beta,
                 "betaprime": stats.betaprime,
                 "lognorm": stats.lognorm,
                 }

rootdir = CONLLU_BASEPATH
blacklist = []  # BLACKLIST
allconll = get_all_files_recurse(rootdir)
train, test, dev = filter_conllu_files(allconll, blacklist)


def conllu_get_fields(fname):
    """
    Processes one conllu file
    :param fname: absolute path to the conllu file
    :return:
    """
    conll = pyconll.load_from_file(fname)
    upos = []
    xpos = []
    # deprel = []
    sentences = []
    forms = []

    src_lang = path_leaf(fname).split('_')[0]
    for sen in conll:
        sentences.append((src_lang, sen.text))
        try:
            forms.extend([t.form for t in sen._tokens])
        except:
            pass
        try:
            sen_upos = [t.upos for t in sen._tokens]
            upos.append((src_lang, sen.text, tuple(sen_upos)))
        except:
            pass
        try:
            sen_xpos = [t.xpos for t in sen._tokens]
            xpos.append((src_lang, sen.text, tuple(sen_xpos)))
        except:
            pass
        # try:
        #     sen_deprel = [t.deprel for t in sen._tokens]
        #     deprel.append((src_lang, sen.text, tuple(sen_deprel)))
        # except:
        #     pass

    # return (set(upos), len(upos)), (set(xpos), len(xpos)), (set(deprel), len(deprel)), (
    #     set(sentences), len(sentences)), (set(forms), len(forms))
    return (set(upos), len(upos)), (set(xpos), len(xpos)), (set(sentences), len(sentences)), (set(forms), len(forms))


def _try_get_2list(fname):
    try:
        return conllu_get_fields(fname)
    except Exception as e:
        print("Error processing file: {} \nWith error: {}".format(fname, e))


def conllu_process_get_2list(rootdir=CONLLU_BASEPATH, blacklist=BLACKLIST):
    allconll = get_all_files_recurse(rootdir)
    train, test, dev = filter_conllu_files(allconll, blacklist)
    all_files = train + test + dev
    #     print(all_files)

    with Pool(processes=cpu_count()) as pool:
        res = pool.map(_try_get_2list, all_files)
        return res


def extract_data_from_fields(results):
    upos_data = []
    # deprel_data = []
    sentences_data = []
    forms_data = []

    for r in results:
        # upos_val, xpos_val, deprel_val, sentences_val, forms_val = r
        upos_val, xpos_val, sentences_val, forms_val = r
        #     print("lala 1")
        forms_data.extend(forms_val[0])
        for val in upos_val[0]:
            #         print(val)
            lang1, txt1, upos = val
            upos_data.append((lang1, txt1, upos, len(upos)))
        # for lang3, txt3, deprel in deprel_val[0]:
        #     deprel_data.append((lang3, txt3, deprel, len(deprel)))
        for lang4, txt4 in sentences_val[0]:
            sentences_data.append((lang4, txt4, len(txt4)))

    return upos_data, sentences_data, forms_data
    # return upos_data, deprel_data, sentences_data, forms_data


def get_best_distribution(data, distributions=DISTRIBUTIONS):
    dist_results = []
    params = {}
    for dist_name, dist in distributions.items():
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = stats.kstest(data, dist_name, args=param)
        #         print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, D, p))

    # select the best fitted distribution
    # store the name of the best fit and its p value
    best_dist, D, best_p = max(dist_results, key=lambda item: item[2])
    #     print("Best fitting distribution: "+str(best_dist))
    #     print("Best p value: "+ str(best_p))
    #     print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]


# def compute_distributions(upos_data, deprel_data, sentences_data, langs=None):
def compute_distributions(upos_data, sentences_data, langs=None):
    df_upos = pd.DataFrame(upos_data, columns=["lang", "text", "upos", "upos_len"])
    # df_deprel = pd.DataFrame(deprel_data, columns=["lang", "text", "deprel", "deprel_len"])
    df_txt = pd.DataFrame(sentences_data, columns=["lang", "text", "text_len"])

    if langs is None:
        langs = sorted(df_upos['lang'].unique())

    langs_data = {}

    for lang in langs:
        try:
            #         fig, ax = plt.subplots()
            dest_lang = languages.get(alpha_2=lang) if len(lang) == 2 else languages.get(alpha_3=lang)
            dest_lang = dest_lang.name
            lng_upos_len = df_upos.loc[df_upos['lang'] == lang]['upos_len']
            # lng_deprel_len = df_deprel.loc[df_deprel['lang'] == lang]['deprel_len']
            lng_text_len = df_txt.loc[df_txt['lang'] == lang]['text_len']

            langs_data[lang] = {
                'lang': dest_lang,
                'upos_len': lng_upos_len,
                'upos_distrib': get_best_distribution(lng_upos_len),
                # 'deprel_len': lng_deprel_len,
                # 'deprel_distrib': get_best_distribution(lng_deprel_len),
                'text_len': lng_text_len,
                'text_distrib': get_best_distribution(lng_text_len),
            }

        except Exception as e:
            print("Error processing lang {} with Exception {}".format(lang, e))
            pass
    return langs_data


def _try_compute_distributions(upos_data, sentence_data):
    try:
        return compute_distributions(upos_data=upos_data, sentences_data=sentence_data)
    except Exception as e:
        print("Error computing distributions With error: {}".format(e))


# TODO parallelization of compute_distributions with starmap


def _get_stats(distrib, distrib_params, data, n_bins=100, n_samples=100):
    """
    
    :param distrib: distribution function (scipy.stats.[beta|norm|....]) 
    :param distrib_params: parameters of the distribution
    :param data:
    :param n_bins: number of bins to compute for the histograms.
    :param n_samples: number of samples for the CDF and PDF functions
    :return: (stats, {cdf,pdf})
    """
    try:  # if data is a pandas dataframe (which it is) TODO cleanup these dirty things
        data = data.to_numpy()
    except:
        pass
    mskv = [None, None, None, None]
    t_mskv = distrib.stats(*distrib_params)
    for i in range(len(t_mskv)):  # mean, variance, skew, kurtosis -> variable length
        mskv[i] = t_mskv[i]
    ret_stats = {
        'mean': mskv[0],  # mean, variance, skew, kurtosis -> variable length
        'variance': mskv[1],
        'skew': mskv[2],
        'kurtosis': mskv[3],
        'median': distrib.median(*distrib_params),
        'std': distrib.std(*distrib_params),
        'intervals': {'99': distrib.interval(0.99, *distrib_params),
                      '98': distrib.interval(0.98, *distrib_params),
                      '95': distrib.interval(0.95, *distrib_params),
                      '90': distrib.interval(0.90, *distrib_params),
                      '85': distrib.interval(0.85, *distrib_params),
                      '80': distrib.interval(0.8, *distrib_params),
                      }
    }
    max_len = max(data)
    x = np.linspace(0, max_len, 100)
    hist, bin_edges = np.histogram(data, bins=n_bins)  # (hist, bin_edges)
    # the function computation is to make life easy when drawing with bokeh ... some points still to clarify
    # for this n_samples needs to be the same as n_bins
    ret_foo = {'x': x,
               'hist': hist,
               'bin_edges': bin_edges,
               # 'bin_edges_left': bin_edges[:-1],
               # 'bin_edges_right': bin_edges[1:],
               'cdf': distrib.cdf(x, *distrib_params),
               'pdf': distrib.pdf(x, *distrib_params)
               }
    return ret_stats, ret_foo


def _get_lang_stats(lang_data, distributions=DISTRIBUTIONS):
    upos_distrib = distributions[lang_data['upos_distrib'][0]]
    upos_distrib_params = lang_data['upos_distrib'][2]
    #     print('upos', upos_distrib, upos_distrib_params)
    upos_data = lang_data['upos_len']
    upos_stats, upos_functions = _get_stats(upos_distrib, upos_distrib_params, upos_data)
    #
    # deprel_distrib = distributions[lang_data['deprel_distrib'][0]]
    # deprel_distrib_params = lang_data['deprel_distrib'][2]
    # #     print('deprel', deprel_distrib, deprel_distrib_params)
    # deprel_data = lang_data['deprel_len']
    # deprel_stats, deprel_functions = _get_stats(deprel_distrib, deprel_distrib_params, deprel_data)
    #
    text_distrib = distributions[lang_data['text_distrib'][0]]
    text_distrib_params = lang_data['text_distrib'][2]
    #     print('text', text_distrib, text_distrib_params)
    text_data = lang_data['text_len']
    text_stats, text_functions = _get_stats(text_distrib, text_distrib_params, text_data)

    lang_data['upos_stats'] = upos_stats
    # lang_data['deprel_stats'] = deprel_stats
    lang_data['text_stats'] = text_stats

    lang_data['upos_functions'] = upos_functions
    # lang_data['deprel_functions'] = deprel_functions
    lang_data['text_functions'] = text_functions

    return lang_data


def flatten_dict(lang, d, sep="_"):
    import collections

    obj = collections.OrderedDict()
    obj['lang_code'] = lang
    lang_name = languages.get(alpha_2=lang) if len(lang) == 2 else languages.get(alpha_3=lang)
    obj['lang_name'] = lang_name.name

    def recurse(t, parent_key=""):

        if isinstance(t, list):
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)

    return obj


def make_plot(title, data_source):
    # TODO make it even better being able to change the Sizing mode from a dropdown menu ?
    hover = HoverTool(
        names=["CDF"],
        tooltips=[
            ("length", '$x{int}'),
            ("Count", "@hist{int}"),
            ("pdf", "@pdf{1.111}"),
            ("cdf", "@cdf{1.111}"),
        ],
        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline',
    )

    p = figure(title=title, background_fill_color="#fafafa",
               plot_height=500, sizing_mode="stretch_width",
               tools="crosshair,pan,wheel_zoom,box_zoom,zoom_in,zoom_out,undo,redo,reset",
               toolbar_location="left",
               output_backend="webgl")
    p.add_tools(hover)
    # p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.xaxis.axis_label = 'Length'
    p.yaxis.axis_label = 'Count'
    # second axe, probability
    p.extra_y_ranges = {"cdf(x)": Range1d(start=0., end=1.02),
                        "Pr(x)": Range1d(start=0., end=max(data_source.data['pdf']) * 1.02)
                        }

    p.add_layout(LinearAxis(y_range_name="Pr(x)", axis_label='Pr(x)'), 'right')
    p.quad(name='hist', top='hist', bottom=0, left='bin_edges_left', right='bin_edges_right',
           fill_color="blue", line_color="white", alpha=0.5, legend_label="Freq.", source=data_source)
    p.line(name='PDF', x='x', y='pdf', line_color="green", line_width=4, alpha=0.7, legend_label="PDF",
           y_range_name="Pr(x)", source=data_source)
    p.line(name='CDF', x='x', y='cdf', line_color="red", line_width=2, alpha=0.7, legend_label="CDF",
           y_range_name="cdf(x)", source=data_source)

    p.y_range.start = 0

    p.title.align = 'center'
    p.legend.location = "center_right"
    #     p.legend.location = "bottom_right"
    p.legend.background_fill_color = "#fefefe"
    p.grid.grid_line_color = "grey"
    #     p.legend.click_policy="mute"
    p.legend.click_policy = "hide"
    leo_label = Label(x=0, y=10, text='leomrocha.github.io')
    p.add_layout(leo_label)
    return p


def _make_data_sources(lang_data):
    lang_name = lang_data['lang']
    ##
    upos_plot_name = lang_name + ' - Text Length by UPOS Count'
    upos_stats = lang_data['upos_stats']
    upos_functions = lang_data['upos_functions']

    upos_data_source = ColumnDataSource({'hist': upos_functions['hist'],
                                         'bin_edges_left': upos_functions['bin_edges'][:-1],
                                         'bin_edges_right': upos_functions['bin_edges'][1:],
                                         'x': upos_functions['x'],
                                         'pdf': upos_functions['pdf'],
                                         'cdf': upos_functions['cdf']
                                         }
                                        )
    # TODO refactor this to make it better ...
    text_plot_name = lang_name + ' - Text Length by Character Count'
    text_stats = lang_data['text_stats']
    text_functions = lang_data['text_functions']

    text_data_source = ColumnDataSource({'hist': text_functions['hist'],
                                         'bin_edges_left': text_functions['bin_edges'][:-1],
                                         'bin_edges_right': text_functions['bin_edges'][1:],
                                         'x': text_functions['x'],
                                         'pdf': text_functions['pdf'],
                                         'cdf': text_functions['cdf']
                                         }
                                        )
    return (upos_plot_name, upos_data_source, upos_stats), (text_plot_name, text_data_source, text_stats)


def _make_stats_tables(stats):
    name_col = [k for k in stats.keys() if k != 'intervals' and stats[k] is not None]
    value_col = [round(float(stats[k]), 3) for k in name_col if stats[k] is not None]

    interval_names = list(stats['intervals'].keys())
    interval_values = [round(float(i[1]), 1) for i in stats['intervals'].values()]

    data = dict(
        names=name_col,
        values=value_col,
    )
    source = ColumnDataSource(data)

    columns = [
        TableColumn(field="names", title="Name"),
        TableColumn(field="values", title="Value"),
    ]
    data_table = DataTable(source=source, columns=columns, width=150, fit_columns=True, index_position=None)

    int_data = dict(
        names=interval_names,
        values=interval_values,
    )
    int_source = ColumnDataSource(int_data)

    int_columns = [
        TableColumn(field="names", title="interval"),
        TableColumn(field="values", title="Max Value"),
    ]

    interval_table = DataTable(source=int_source, columns=int_columns, width=120, fit_columns=True, index_position=None)

    return data_table, interval_table


# def _make_grid_datasources(lang_data):
#     upos_plt_info, txt_plt_info = _make_data_sources(lang_data)
#     upos_plot = make_plot(*upos_plt_info[:2])
#     text_plot = make_plot(*txt_plt_info[:2])
#
#     upos_stats_table, upos_interval_table = _make_stats_tables(upos_plt_info[2])
#     text_stats_table, text_interval_table = _make_stats_tables(txt_plt_info[2])
#     pass


def _make_grid_plot(lang_data):
    upos_plt_info, txt_plt_info = _make_data_sources(lang_data)
    upos_plot = make_plot(*upos_plt_info[:2])
    text_plot = make_plot(*txt_plt_info[:2])

    upos_stats_table, upos_interval_table = _make_stats_tables(upos_plt_info[2])
    text_stats_table, text_interval_table = _make_stats_tables(txt_plt_info[2])

    upos_stats = Column(upos_stats_table, sizing_mode="fixed", height=350, width=150)
    upos_interval = Column(upos_interval_table, sizing_mode="fixed", height=350, width=150, margin=(0, 0, 0, 10))
    text_stats = Column(text_stats_table, sizing_mode="fixed", height=350, width=150)
    text_interval = Column(text_interval_table, sizing_mode="fixed", height=350, width=150, margin=(0, 0, 0, 10))

    gp = gridplot([upos_stats, upos_interval, upos_plot,
                   text_stats, text_interval, text_plot],
                  ncols=3, sizing_mode="stretch_width", plot_height=350)
    return gp


def stats_dict2table(all_lang_stats):
    upos_stats = []
    # deprel_stats = []
    text_stats = []
    for lang, lang_data in all_lang_stats.items():
        # upos_row, deprel_row, text_row = stats_dict2rows(lang, lang_data)
        upos_row, text_row = stats_dict2rows(lang, lang_data)
        upos_stats.append(upos_row)
        # deprel_stats.append(deprel_row)
        text_stats.append(text_row)

    upos_df = pd.DataFrame(upos_stats)
    # deprel_df = pd.DataFrame(deprel_stats)
    text_df = pd.DataFrame(text_stats)

    # return upos_df, deprel_df, text_df
    return upos_df, text_df


def stats_dict2rows(lang, lang_data):
    upos_data = flatten_dict(lang, lang_data['upos_stats'])
    # deprel_data = flatten(lang, lang_data['deprel_stats'])
    text_data = flatten_dict(lang, lang_data['text_stats'])
    return upos_data, text_data
    # return upos_data, deprel_data, text_data


# complete tables showing stats for all the available languages
def _make_complete_stats_tables(all_lang_stats):
    upos_df, text_df = stats_dict2table(all_lang_stats)
    df_tables = (upos_df.round(2), text_df.round(2))
    intervals = ['intervals_99', 'intervals_98', 'intervals_95', 'intervals_90', 'intervals_85', 'intervals_80']
    cols_to_drop = intervals + ['intervals_99_low', 'intervals_98_low',
                                'intervals_95_low', 'intervals_90_low',
                                'intervals_85_low', 'intervals_80_low',
                                'skew', 'kurtosis']
    # separate and clean the data
    df_clean_tables = []
    for df in df_tables:
        for interval in intervals:
            df[[interval + '_low', interval + '_high']] = pd.DataFrame(df[interval].tolist(), index=df.index)
        df = df.drop(columns=cols_to_drop).round(2)
        df_clean_tables.append(df)

    bk_tables = []

    def _get_title(col_name):
        if col_name == 'lang_code':
            return 'Code'
        elif col_name == 'lang_name':
            return 'Language'
        else:
            return col_name.replace('intervals_', '').replace('_high', '').replace('_low', '')

    def _get_width(col_name):
        if col_name == 'lang_code':
            return 60
        elif col_name == 'lang_name':
            return 140
        else:
            return 50

    for table in df_clean_tables:
        columns = [TableColumn(field=Ci, title=_get_title(Ci), width=_get_width(Ci)) for Ci in
                   table.columns]  # bokeh columns
        data_table = DataTable(columns=columns, source=ColumnDataSource(table), sizing_mode='stretch_width',
                               fit_columns=False)  # bokeh table
        bk_tables.append(data_table)

    return bk_tables


# convert to json  TODO this must be improved and all sent to the NumpyEncoder ...
# This solution is modified from:
# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
# https://github.com/mpld3/mpld3/issues/434

# class NumpyEncoder(json.JSONEncoder):
#     """ Special json encoder for numpy types """
#
#     def default(self, obj):
#         if isinstance(obj, (tuple, set)):
#             return list(obj)
#         elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
#                               np.int16, np.int32, np.int64, np.uint8,
#                               np.uint16, np.uint32, np.uint64)):
#             return int(obj)
#         elif isinstance(obj, (np.float_, np.float16, np.float32,
#                               np.float64)):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, pd.Series):
#             obj = obj.to_list()
#         return json.JSONEncoder.default(self, obj)


def _recursive_jsonify(dict_data):
    new_dict = {}
    for k, v in dict_data.items():
        k = str(k)  # always convert,
        if isinstance(v, (tuple, set)):
            ov = []
            for t in v:
                if isinstance(t, str):
                    ov.append(t)
                elif isinstance(t, tuple):
                    ov.append([float(i) for i in t])
                else:
                    ov.append(float(t))
            v = ov
        if isinstance(v, pd.Series):
            v = v.to_list()
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if np.issubdtype(type(v), np.number):
            v = float(v)
        if isinstance(v, dict):
            new_dict[k] = _recursive_jsonify(v)
        else:
            new_dict[k] = v
    return new_dict
###


def _save_file(obj, fpath):
    with open(fpath, 'w') as f:
        f.write(obj)
        f.flush()


def _generate_html_plots(all_stats, path='./docs/plots{}/', fname="{}_plot{}"):
    all_grids = {}
    all_grids_html = {}
    all_grids_json = {}

    for lang, data in all_stats.items():
        grid = _make_grid_plot(data)
        all_grids[lang] = grid
        plt_name = lang + " stats plot"
        # complete html page
        html = file_html(grid, CDN, plt_name)
        all_grids_html[lang] = html
        # grid as json information
        # jsn_grid = json.dumps(json_item(grid, plt_name))
        # all_grids_json[lang] = jsn_grid

        # save html
        Path(path.format('_html')).mkdir(parents=True, exist_ok=True)
        html_fname = os.path.join(path.format('_html'), fname.format(lang, '.html'))
        _save_file(html, html_fname)
        # save json
        # Path(path.format('_json')).mkdir(parents=True, exist_ok=True)
        # json_fname = os.path.join(path.format('_json'), fname.format(lang, '.json'))
        # _save_file(jsn_grid, json_fname)

    # all_components = (all_components_script, all_components_div) = components(all_grids)
    # # save all grids as component
    # Path(path.format('_components')).mkdir(parents=True, exist_ok=True)
    # comp_fname_script = os.path.join(path.format('_components'), fname.format('all_components', '_script.html'))
    # comp_fname_div = os.path.join(path.format('_components'), fname.format('all_components', '_div.html'))
    # _save_file(all_components_div, comp_fname_div)
    # _save_file(all_components_script, comp_fname_script)

    return all_grids, all_grids_html
    # return all_grids, all_grids_html, all_components


def _generate_html_table(table, name, path='./docs/tables_{}/', fname="{}_table{}"):

    Path(path.format('html')).mkdir(parents=True, exist_ok=True)

    html_fname = os.path.join(path.format('html'), fname.format(name, '.html'))
    html = file_html(table, CDN, name)
    _save_file(html, html_fname)

    # Path(path.format('json')).mkdir(parents=True, exist_ok=True)
    # jsn_fname = os.path.join(path.format('json'), fname.format(name, '.json'))
    # jsn = json.dumps(json_item(table, name))
    # _save_file(json, jsn_fname)

    # Path(path.format('components')).mkdir(parents=True, exist_ok=True)
    # cmp_script_fname = os.path.join(path.format('components'), fname.format(name, '_script.html'))
    # cmp_div_fname = os.path.join(path.format('components'), fname.format(name, '_div.html'))
    # cmp_script, cmp_div = components(table)
    # # _save_file(cmp_script, cmp_script_fname)
    # # _save_file(cmp_div, cmp_div_fname)

    return html
    # return html, (cmp_script, cmp_div)
    # return html, jsn, (cmp_script, cmp_div)


def generate_files(blacklist=[], saveto='conllu_stats.json.gz'):
    res = conllu_process_get_2list(blacklist=blacklist)
    # upos_data, deprel_data, sentences_data, forms_data = extract_data_from_fields(res)
    # langs_data = compute_distributions(upos_data, deprel_data, sentences_data)
    upos_data, sentences_data, forms_data = extract_data_from_fields(res)
    langs_data = compute_distributions(upos_data, sentences_data)

    all_stats = {}

    for lang, lang_data in langs_data.items():
        print('processing {}'.format(lang))
        all_stats[lang] = _get_lang_stats(lang_data)

    all_stats_copy = copy.deepcopy(all_stats)
    all_stats_copy = _recursive_jsonify(all_stats_copy)
    jsn = json.dumps(all_stats_copy)
    # this is with default json lib
    # jsn = json.dumps(all_stats_copy, cls=NumpyEncoder)
    # for non compressed file -> too big, not worth it
    # with open('conllu_stats.json', 'w') as f:
    #     f.write(jsn)
    #     f.flush()

    with gzip.open(saveto, 'wb') as f:
        print("Saving to {}".format(saveto))
        # f.write(jsn.encode('utf-8'))
        f.write(jsn)
        f.flush()

    return all_stats


def generate_html(all_stats):
    all_stats_copy = copy.deepcopy(all_stats)
    all_grids, all_grids_html, all_grid_components = _generate_html_plots(all_stats_copy)

    all_stats_copy = copy.deepcopy(all_stats)
    upos_table, text_table = _make_complete_stats_tables(all_stats_copy)

    _generate_html_table(upos_table, 'upos_table')
    _generate_html_table(text_table, 'text_table')


def main():
    all_stats = generate_files()
    generate_html(all_stats)


if __name__ == '__main__':
    main()
