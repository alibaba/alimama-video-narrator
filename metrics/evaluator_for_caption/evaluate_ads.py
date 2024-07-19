import numpy as np
from cap_eval.bleu.bleu import Bleu
from cap_eval.meteor.meteor import Meteor
from cap_eval.cider.cider import Cider
#from .tokenizer.tokenizer_13a import Tokenizer13a
import pdb
import json
import sys

#tokenizer =Tokenizer13a()

meteor_scorer = Meteor()
cider_scorer = Cider()
bleu_scorer = Bleu(4)

def bleu_eval(refs, cands):
  print ("calculating bleu_4 score...")
  bleu, _ = bleu_scorer.compute_score(refs, cands)
  return bleu

def cider_eval(refs, cands):
  print ("calculating cider score...")
  cider, ciders = cider_scorer.compute_score(refs, cands)
  return cider,ciders

def meteor_eval(refs, cands):
  print ("calculating meteor score...")
  meteor, _ = meteor_scorer.compute_score(refs, cands)
  return meteor

def compute(preds, refs,keys):
  refcaps = {}
  candcaps = {}
  for i in range(len(preds)):
    candcaps[str(i)] = [preds[i]]
    refcaps[str(i)] = [refs[i]]   # [ref1, ref2, ref3] for multiple references, and [ref] for single reference
  bleu = bleu_eval(refcaps, candcaps)
  cider,ciders = cider_eval(refcaps, candcaps)
  meteor = meteor_eval(refcaps, candcaps)
  scores = {'meteor':meteor,'cider':cider,'bleu_4':bleu[3],'bleu_3':bleu[2],'bleu_2':bleu[1],'bleu_1':bleu[0]}
  return scores

def formats(t_str):
    out_str = []
    for l in t_str:
        #print(l)
        l = " ".join([str(tok) for tok in l])

        out_str.append(l)
    #out_str = tokenizer.tokenize(out_str)
    return out_str
fin = json.load(open(sys.argv[1]))
refs = []
pred_1 = []
pred_2 = []
keys = []
for key in fin.keys():
    k=fin[key]
    if "0.1" not in k or "gt" not in k:continue
    keys.append(key)
    refs.append(k["gt"])
    pred_1.append(k["0.1"])
    #pred_1.append(k["0.2"])

print(compute(formats(pred_1),formats(refs),keys))
#print(compute(formats(pred_2),formats(refs)))
