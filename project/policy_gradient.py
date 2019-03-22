import tools as ut
import numpy as np
import pandas as pd

#Backtest with a threshold t and a trading cost g
def backtest(data, t, g):
    t = max(t, 0)
    d = data.copy()
    d.loc[d.predictor > t, 'position'] = 1
    d.loc[d.predictor < -t, 'position'] = -1
    d.position = d.position.fillna(method='ffill')
    d.position = d.position.fillna(0)
    d['turnover'] = np.abs(d.position - d.position.shift())
    d.loc[d.turnover.isnull(), 'turnover'] = d.loc[d.turnover.isnull(), 'position']
    d['pnl'] = d.eval('position*fwd_ret1')
    return d[:-1].pnl.sum() - d[:-1].turnover.sum()*g

def policy_gradient(d, g):
    d['day'] = ut.get(d, 'date')
    t = g
    rate = 0.00002
    delta = 0.05
    res = pd.DataFrame(index = d.day.unique(), columns = ['pnl_naive', 'pnl_dynamic', 'q'])
    for day in d.day.unique():
        batch = d[d.day == day]
        pnl = backtest(batch, t, g)
        grad = 0.5*(backtest(batch, t+delta, g) - backtest(batch, t-delta, g))/delta
        t = max(t + rate*grad, 0)
        res.loc[day] = [backtest(batch, g, g), pnl, t]
        #print('Day: ' + day.strftime(format='%Y-%m-%d') + ' PNL: ' + str(pnl) + ' t:'+str(t))
    return res
