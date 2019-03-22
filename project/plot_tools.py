from plotly.offline import plot, iplot
import pandas as pd
import numbers
import numpy as np
import plotly.graph_objs as go

def plot_scatter(xs, ys, label, title='', xtitle='', ytitle='', wpath='', save = False):
    c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 270, len(ys))]
    data_plot = []
    shareXs = (len(xs)==1)&(len(ys)>1)
    for i in range(len(ys)):
        trace = go.Scatter(
            x = xs[shareXs*0+(1-shareXs)*i],
            y = ys[i],
            mode = 'markers+lines',
            name = str(label[i]),
            marker = dict(color = c[i])

        )

        data_plot = data_plot + [trace]
    layout = go.Layout(
        title = title,
        xaxis = dict(
            title = xtitle,
            showline = True,
            zeroline = True,
        ),
        yaxis = dict(
            title = ytitle,
            showline = True,
            zeroline = True,
        )
    )
    fig = go.Figure(data=data_plot, layout=layout)
    iplot(fig)
    if save:
        plot(fig, filename = os.path.join(wpath, title+'.html'))

def plot_by_group(df, bys, ons , f, q, title = '', xtitle = '', ytitle = '', wpath = '', save = False):
    res = apply_by_group(df, bys, ons , f, q)
    if len(bys)>1:
        y_buckets = res.index.get_level_values('by_{}'.format(bys[1])).unique()
        y = [res.xs(y_bucket, level = 'by_{}'.format(bys[1]))[on].values for on in ons for y_bucket in y_buckets]
        x = [res.xs(y_bucket, level = 'by_{}'.format(bys[1]))[on].index for on in ons for y_bucket in y_buckets]

        plot_scatter(x, y, list(itertools.product(ons, y_buckets)), title, xtitle, ytitle, wpath, save)
    else:
        x_buckets = res.index.get_level_values('by_{}'.format(bys[0])).unique()
        y = [res[on].values for on in ons]
        nqtls = len(ons)
        plot_scatter(nqtls*[x_buckets], y, ons, title, xtitle, ytitle, wpath, save)

def apply_by_group(df, bys, ons, f, qs):
    def aux_apply_by_group(df, by, q):
        if isinstance(df[by].iloc[0], numbers.Real):
            bins_mapping, bins = pd.qcut(df[by], q, labels=None, retbins=True, precision=3, duplicates='drop')
            df['by_{}'.format(by)] = bins_mapping
            bin_means = df.groupby('by_{}'.format(by))[by].mean().round(3)
            df['by_{}'.format(by)] = df['by_{}'.format(by)].map(bin_means)
        else:
            df['by_{}'.format(by)] = df[by]
        return df
    df = df.copy()
    if not isinstance(bys,list):
        bys = [bys]
    if not isinstance(qs,list):
        qs = [qs]
    if (len(qs) == 1) & (len(bys)==2):
        qs = [qs[0], qs[0]]
    if len(bys)==1:
        df = aux_apply_by_group(df, bys[0], qs[0])
        return df.groupby(['by_{}'.format(by) for by in bys])[ons].apply(f)
    elif len(bys)==2:
        df = aux_apply_by_group(df, bys[1], qs[1])
        df = df.groupby(f'by_{bys[1]}').apply(aux_apply_by_group, by = bys[0], q = qs[0])
        return df.groupby(['by_{}'.format(by) for by in bys])[ons].apply(f)
    else:
        raise("NotImplementedError: Groupby not implemented for 3 axis and more")
