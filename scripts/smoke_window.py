import sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
from aptg.agents import simulate_episode
from aptg.graph import build_graph, graph_features, aptg_score
def run(a):
    mech,cond,seed,w=a
    ep=simulate_episode(mech,cond,seed)
    g=build_graph(ep,None,window=w,stride=max(1,w//2),n_perm=50,correction="maxT")
    f=graph_features(g)
    return mech,w,f["persistence"],aptg_score(f)
if __name__=="__main__":
    jobs=[(m,c,s,w) for (m,c) in [("leader_follower","C4"),("complementary","C4"),("independent","C0"),("shared_model","C0")] for s in range(6) for w in (10,20)]
    with ProcessPoolExecutor(6) as ex: res=list(ex.map(run,jobs))
    import collections
    d=collections.defaultdict(list)
    for m,w,p,a in res: d[(m,w)].append((p,a))
    for k in sorted(d): print(k,"mean persistence %.2f  mean aptg %.3f  frac aptg>0 %.2f"%(np.mean([x[0] for x in d[k]]),np.mean([x[1] for x in d[k]]),np.mean([x[1]>0 for x in d[k]])))
