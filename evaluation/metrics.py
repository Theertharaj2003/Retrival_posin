import math

def precision_recall_f1(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; f=2*p*r/(p+r) if p+r else 0; return p,r,f

def attack_success(records,key='poisoned_topk'):
    return sum(r[key]>0 for r in records)/len(records) if records else 0

def retrieval_recall(records,key='poisoned_topk'):
    # Fraction of records where no poisoned item remains in selected top-k.
    return sum(r[key]==0 for r in records)/len(records) if records else 0

def healing_success(records):
    eligible=[r for r in records if r.get('baseline_poisoned_topk',0)>0]
    return sum(r.get('healed_poisoned_topk',1)==0 for r in eligible)/len(eligible) if eligible else 0

def mean(records,key):
    vals=[r.get(key,0) for r in records]; return sum(vals)/len(vals) if vals else 0
