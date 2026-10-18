"""Statistics for the free-text run (no tagged line).

Run after the score_*.py scripts. Used for the read-out comparison in the paper.
"""
import json
from collections import defaultdict, Counter
from scipy.stats import chi2_contingency, fisher_exact, binomtest

def load(p): return json.load(open(p))
bench = load("data/benchmark.json")

# variance classification: low-variance issue = all 4 parties share one gold bucket
variance = {}
for c in ["japan","south_korea"]:
    for iss in bench[c]["issues"]:
        buckets = {cell["bucket"] for cell in iss["cells"]}
        variance[(c, iss["issue_code"])] = "low" if len(buckets)==1 else "high"

SP = [r for r in load("scored/outputs_scored_singleparty.json")
      if r["issue_code"]!="per408" and r["verdict"] not in ("TRUNCATED_EXCLUDE","AMBIGUOUS_NEEDS_MANUAL_READ")]
SHORT={"cohere-command-a":"Cohere-A","groq-gpt-oss-120b":"gpt-oss-120b","groq-gpt-oss-20b":"gpt-oss-20b",
       "mistral-large":"Mistral-L","zhipu-glm5":"GLM-5","anthropic-sonnet":"Claude-Sonnet",
       "deepseek-v3":"DeepSeek-V3","gemini-2.5-flash":"Gemini-Flash"}
VERD=["correct","partial","misattributed","hallucinated","no_knowledge"]

print("="*70)
print("Single-party verdict distribution (%), free-text run")
print("="*70)
by=defaultdict(lambda:Counter())
for r in SP: by[r["model"]][r["verdict"]]+=1
tot=Counter()
for m in sorted(by, key=lambda m:-by[m]["correct"]/sum(by[m].values())):
    n=sum(by[m].values()); tot+=by[m]
    print(f"  {SHORT[m]:14} "+" ".join(f"{100*by[m][v]/n:5.1f}" for v in VERD)+f"   n={n}")
N=sum(tot.values())
print(f"  {'OVERALL':14} "+" ".join(f"{100*tot[v]/N:5.1f}" for v in VERD)+f"   n={N}")
print("  (cols: Corr Part Misat Hall NoKn)")

# calibration chi2: model x (no_knowledge vs not)
rows=[]
for m in by:
    nk=by[m]["no_knowledge"]; rows.append([nk, sum(by[m].values())-nk])
chi2,p,_,_=chi2_contingency(rows)
nvar=min(len(rows),2)-1  # cramers V for 2-col table -> phi-like
import math
cv=math.sqrt(chi2/(N*(min(len(rows),2)-1)))
print(f"\n  calibration (no_knowledge across models): chi2={chi2:.2f}, p={p:.2e}, CramerV={cv:.2f}")

print("\n"+"="*70)
print("Single-party exact-match accuracy")
print("="*70)
print(f"  overall exact-match faithfulness: {tot['correct']}/{N} = {100*tot['correct']/N:.1f}%")

print("\n"+"="*70)
print("Error type vs cross-party variance")
print("="*70)
def acc(sel):
    s=[r for r in SP if sel(r)]; c=sum(1 for r in s if r["verdict"]=="correct")
    return c,len(s)
c,n=acc(lambda r: variance[(r["country"],r["issue_code"])]=="low");   print(f"  low-variance accuracy : {c}/{n} = {100*c/n:.1f}%")
c,n=acc(lambda r: variance[(r["country"],r["issue_code"])]=="high");  print(f"  high-variance accuracy: {c}/{n} = {100*c/n:.1f}%")
c,n=acc(lambda r: r["issue_code"]=="per504"); print(f"  welfare per504 accuracy: {c}/{n} = {100*c/n:.1f}%")
c,n=acc(lambda r: variance[(r["country"],r["issue_code"])]=="low" and r["issue_code"]!="per504"); print(f"  low-var non-welfare acc: {c}/{n} = {100*c/n:.1f}%")
# misattribution low vs high variance, Fisher
def mis(v):
    s=[r for r in SP if variance[(r["country"],r["issue_code"])]==v]
    m=sum(1 for r in s if r["verdict"]=="misattributed"); return m,len(s)
ml,nl=mis("low"); mh,nh=mis("high")
_,pf=fisher_exact([[mh,nh-mh],[ml,nl-ml]])
print(f"  misattribution low-var: {ml}/{nl}={100*ml/nl:.1f}%  high-var: {mh}/{nh}={100*mh/nh:.1f}%  Fisher p={pf:.2e}")

print("\n"+"="*70)
print("Multi-party ranking accuracy (tie-aware) + binomial vs 0.25")
print("="*70)
MP=[r for r in load("scored/outputs_scored_multiparty.json") if r["issue_code"]!="per408"]
for lab,key in [("highest","highest_correct"),("lowest","lowest_correct")]:
    v=[r[key] for r in MP if r[key] is not None]; c=sum(v)
    bp=binomtest(c,len(v),0.25,alternative="greater").pvalue
    print(f"  {lab}: {c}/{len(v)} = {100*c/len(v):.1f}%   binomial p={bp:.2e}")

print("\n"+"="*70)
print("Self-consistency (%) by model, with omnibus chi2")
print("="*70)
SC=load("scored/outputs_selfconsistency.json")
agg=defaultdict(lambda:{"hi":[0,0],"lo":[0,0]})
for r in SC:
    if r["highest_agree"] is not None: agg[r["model"]]["hi"][0]+=r["highest_agree"]; agg[r["model"]]["hi"][1]+=1
    if r["lowest_agree"] is not None: agg[r["model"]]["lo"][0]+=r["lowest_agree"]; agg[r["model"]]["lo"][1]+=1
H=[0,0]; L=[0,0]
for m in sorted(agg):
    hi=agg[m]["hi"]; lo=agg[m]["lo"]; H[0]+=hi[0];H[1]+=hi[1];L[0]+=lo[0];L[1]+=lo[1]
    hs=f"{100*hi[0]/hi[1]:.1f} (n={hi[1]})" if hi[1] else "n/a"
    ls=f"{100*lo[0]/lo[1]:.1f} (n={lo[1]})" if lo[1] else "n/a"
    print(f"  {SHORT[m]:14} highest={hs:16} lowest={ls}")
print(f"  {'OVERALL':14} highest={100*H[0]/H[1]:.1f} (n={H[1]})   lowest={100*L[0]/L[1]:.1f} (n={L[1]})")
# omnibus chi2 across models on highest-agree
rows=[[agg[m]['hi'][0], agg[m]['hi'][1]-agg[m]['hi'][0]] for m in agg if agg[m]['hi'][1]>0]
try:
    chi2,p,_,_=chi2_contingency(rows); print(f"  omnibus chi2 (highest, across models): chi2={chi2:.2f}, p={p:.3f}")
except Exception as e: print("  omnibus chi2 error:",e)

print("\n"+"="*70)
print("Grounding: self-consistency ungrounded vs grounded (GLM-5 excluded)")
print("="*70)
scg=load("scored/outputs_selfconsistency_grounded.json")
gmodels={r["model"] for r in scg}
un=[r for r in SC if r["model"] in gmodels]
def rate(recs,key):
    v=[r[key] for r in recs if r[key] is not None]; return sum(v),len(v)
for lab,key in [("highest","highest_agree"),("lowest","lowest_agree")]:
    ua,un_=rate(un,key); ga,gn=rate(scg,key)
    _,pf=fisher_exact([[ga,gn-ga],[ua,un_-ua]])
    print(f"  {lab}: ungrounded {ua}/{un_}={100*ua/un_:.1f}%  grounded {ga}/{gn}={100*ga/gn:.1f}%  Fisher p={pf:.3f}")
# grounded single-party accuracy (7 models) vs ungrounded on same models
SPg=[r for r in load("scored/outputs_scored_singleparty_grounded.json")
     if r["issue_code"]!="per408" and r["verdict"] not in ("TRUNCATED_EXCLUDE","AMBIGUOUS_NEEDS_MANUAL_READ")]
def facc(recs,models):
    s=[r for r in recs if r["model"] in models]; c=sum(1 for r in s if r["verdict"]=="correct"); return c,len(s)
cu,nu=facc(SP,gmodels); cg,ng=facc(SPg,gmodels)
print(f"  single-party accuracy: ungrounded {100*cu/nu:.1f}%  grounded {100*cg/ng:.1f}%  (n_ung={nu}, n_gr={ng})")
