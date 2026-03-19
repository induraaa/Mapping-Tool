"""
Wafer Map Viewer — KDF Analyser + ACS XML Design Loader + Transformer Classifier
Pure-NumPy transformer network identifies wafer type instantly on file load.
Requirements: pip install PySide6 numpy
"""

import sys, os, re, math, xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QLineEdit, QFormLayout, QFrame, QStatusBar, QComboBox,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QToolBar, QSizePolicy, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QCheckBox, QScrollArea, QSplitter, QProgressBar
)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QSize, QTimer
from PySide6.QtGui import (
    QPainter, QColor, QBrush, QPen, QFont, QLinearGradient,
    QRadialGradient, QPixmap, QIcon, QAction
)

# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMER WAFER CLASSIFIER  (pure NumPy, no ML dependencies)
# ═══════════════════════════════════════════════════════════════════════════════

_VOCAB_LIST = [
    "[PAD]","[CLS]","[UNK]",
    "sites_tiny","sites_small","sites_medium","sites_large","sites_huge",
    "xrange_narrow","xrange_medium","xrange_wide","xrange_vwide",
    "yrange_short","yrange_medium","yrange_tall","yrange_vtall",
    "ar_wide","ar_square","ar_tall",
    "proc_T4","proc_other",
    "equip_S500","equip_4200A","equip_other",
    "test_CV_0V","test_CV_1p5V","test_CV_2p5V","test_CV_5p5V",
    "test_CV_Sweep","test_CV_Sweep_2p5V","test_CV_Sweep_5p5V",
    "test_BVr_250uA","test_BVr_1mA","test_BVr_5mA","test_BVr_6p5mA",
    "test_BVr_20mA","test_BVr_10mA","test_BVr_neg1mA","test_BVr_neg3mA",
    "test_BVr_neg4mA","test_BVr_neg5mA","test_BVr_neg6p5mA","test_BVr_neg20mA",
    "test_BVr_neg250uA",
    "test_IR_3p3V","test_IR_1V","test_IR_2V","test_IR_4V",
    "test_IR_neg1V","test_IR_neg2V","test_IR_neg3p3V","test_IR_neg9V","test_IR_120V",
    "test_Vf_10mA","test_Vf","test_Vf3",
    "test_IR1","test_IR2","test_Vbr1","test_Vbr2",
    "test_Con_1","test_Con_2","test_Con_3",
    "param_Cp","param_Gp","param_DCV","param_F_freq",
    "param_Vbr","param_Vbr_250uA","param_Vbr_1mA","param_Vbr_20mA",
    "param_Vf","param_Ileakage","param_IR_val","param_error",
    "grid_dense","grid_sparse","grid_medium",
    "has_cv","no_cv","has_bvr","no_bvr","has_ir","no_ir",
    "has_vf","no_vf","has_cj","no_cj",
    "lot_scr","lot_tvs","lot_tvd","lot_tvsun","lot_tvsti",
    "lot_tails","lot_unknown",
    "meas_cj","meas_dc","meas_cv","meas_full",
]
VOCAB      = {t: i for i, t in enumerate(_VOCAB_LIST)}
VOCAB_SIZE = len(_VOCAB_LIST)
PAD_ID     = VOCAB["[PAD]"]
CLS_ID     = VOCAB["[CLS]"]

def _t(w): return VOCAB.get(w, VOCAB["[UNK]"])

CLASSES = [
    "SCR \u2013 DC (Vbr / IR / Vf)",
    "SCR \u2013 Cj (CV / capacitance)",
    "SCR \u2013 Full (CV + DC)",
    "TVD \u2013 DC",
    "TVS \u2013 DC",
    "TVSTI \u2013 High-density array",
    "TVSUN \u2013 CSP package",
    "Unknown \u2013 DC / Vbr",
    "Unknown \u2013 CV / Cj",
]
N_CLASSES = len(CLASSES)

# ── Feature extraction ────────────────────────────────────────────────────────

def _geom_tokens(sites_n, x_range, y_range, die_ar=None):
    toks = []
    if sites_n < 30:    toks.append(_t("sites_tiny"))
    elif sites_n < 65:  toks.append(_t("sites_small"))
    elif sites_n < 125: toks.append(_t("sites_medium"))
    elif sites_n < 300: toks.append(_t("sites_large"))
    else:               toks.append(_t("sites_huge"))
    xb = ("xrange_narrow" if x_range<=5 else "xrange_medium" if x_range<=9
          else "xrange_wide" if x_range<=14 else "xrange_vwide")
    yb = ("yrange_short" if y_range<=5 else "yrange_medium" if y_range<=9
          else "yrange_tall" if y_range<=15 else "yrange_vtall")
    toks += [_t(xb), _t(yb)]
    density = sites_n / max(x_range * y_range, 1)
    toks.append(_t("grid_dense" if density>0.7 else "grid_sparse" if density<0.4 else "grid_medium"))
    if die_ar is not None:
        toks.append(_t("ar_wide" if die_ar>1.5 else "ar_tall" if die_ar<0.75 else "ar_square"))
    return toks

_TEST_MAP = {
    'CV_0V':'test_CV_0V','CV_1p5V':'test_CV_1p5V','CV_2p5V':'test_CV_2p5V',
    'CV_5p5V':'test_CV_5p5V','CV_Sweep':'test_CV_Sweep',
    'CV_Sweep_2p5V':'test_CV_Sweep_2p5V','CV_Sweep_5p5V':'test_CV_Sweep_5p5V',
    'BVr_250uA':'test_BVr_250uA','BVr_1mA':'test_BVr_1mA','BVr_5mA':'test_BVr_5mA',
    'BVr_6p5mA':'test_BVr_6p5mA','BVr_20mA':'test_BVr_20mA','BVr_10mA':'test_BVr_10mA',
    'BVr_neg1mA':'test_BVr_neg1mA','BVr_neg3mA':'test_BVr_neg3mA',
    'BVr_neg4mA':'test_BVr_neg4mA','BVr_neg5mA':'test_BVr_neg5mA',
    'BVr_neg6p5mA':'test_BVr_neg6p5mA','BVr_neg20mA':'test_BVr_neg20mA',
    'BVr_neg250uA':'test_BVr_neg250uA',
    'IR_3p3V':'test_IR_3p3V','IR_1V':'test_IR_1V','IR_2V':'test_IR_2V',
    'IR_4V':'test_IR_4V','IR_neg1V':'test_IR_neg1V','IR_neg2V':'test_IR_neg2V',
    'IR_neg3p3V':'test_IR_neg3p3V','IR_neg9V':'test_IR_neg9V','IR_120V':'test_IR_120V',
    'Vf_10mA':'test_Vf_10mA','Vf':'test_Vf','Vf3':'test_Vf3',
    'IR1':'test_IR1','IR2':'test_IR2','Vbr1':'test_Vbr1','Vbr2':'test_Vbr2',
    'Con_1':'test_Con_1','Con_2':'test_Con_2','Con_3':'test_Con_3',
}
_PARAM_MAP = {
    'Cp':'param_Cp','Gp':'param_Gp','DCV':'param_DCV','F':'param_F_freq',
    'Vbr':'param_Vbr','Vbr_250uA':'param_Vbr_250uA','Vbr_1mA':'param_Vbr_1mA',
    'Vbr_20mA':'param_Vbr_20mA','Vf':'param_Vf','Vf_10mA':'param_Vf',
    'Ileakage':'param_Ileakage','IR_3p3V':'param_IR_val','error':'param_error',
}

def extract_tokens_from_kdf(header, sites, params, tests):
    toks = [CLS_ID]
    n = len(sites)
    if sites:
        xs=[s['x'] for s in sites]; ys=[s['y'] for s in sites]
        xr=max(xs)-min(xs)+1; yr=max(ys)-min(ys)+1
    else:
        xr=yr=1
    toks += _geom_tokens(n, xr, yr)
    prc = header.get('PRC','').strip()
    toks.append(_t("proc_T4" if prc=='T4' else "proc_other"))
    tst = header.get('TST','').lower()
    toks.append(_t("equip_S500" if 's500' in tst else "equip_4200A" if '4200' in tst else "equip_other"))
    has_cv=has_bvr=has_ir=has_vf=has_cj = False
    for t in tests:
        base = re.sub(r'_P\d+.*$','',t)
        if base in _TEST_MAP: toks.append(_t(_TEST_MAP[base]))
        if re.search(r'CV',t,re.I): has_cv=True
        if re.search(r'BVr|Vbr',t,re.I): has_bvr=True
        if re.search(r'^IR',t,re.I): has_ir=True
        if re.search(r'^Vf',t,re.I): has_vf=True
    for p in params:
        pn=p.split('@')[0]
        if pn in _PARAM_MAP: toks.append(_t(_PARAM_MAP[pn]))
        if pn in ('Cp','Gp'): has_cj=True
    for flag,val in [("has_cv",has_cv),("has_bvr",has_bvr),
                     ("has_ir",has_ir),("has_vf",has_vf),("has_cj",has_cj)]:
        toks.append(_t(flag if val else f"no_{flag[4:]}"))
    lot = header.get('LOT','').lower()
    for kw,tn in [('tvsun','lot_tvsun'),('tvsti','lot_tvsti'),('tvs','lot_tvs'),
                  ('tvd','lot_tvd'),('scr','lot_scr'),('tails','lot_tails')]:
        if kw in lot: toks.append(_t(tn)); break
    else: toks.append(_t("lot_unknown"))
    if lot.endswith('_cj') or lot.endswith('cj'): toks.append(_t("meas_cj"))
    elif lot.endswith('_dc') or lot.endswith('dc'): toks.append(_t("meas_dc"))
    elif has_cj and not has_bvr: toks.append(_t("meas_cj"))
    elif has_bvr and not has_cj: toks.append(_t("meas_dc"))
    elif has_cv and has_bvr: toks.append(_t("meas_full"))
    elif has_cv: toks.append(_t("meas_cv"))
    return toks

def extract_tokens_from_xml(design):
    toks = [CLS_ID]
    n = len(design.site_names)
    if design.site_coords:
        xs=[v[0] for v in design.site_coords.values()]
        ys=[v[1] for v in design.site_coords.values()]
        xr=max(xs)-min(xs)+1; yr=max(ys)-min(ys)+1
    else: xr=yr=1
    ar = (design.die_size_x_mm/design.die_size_y_mm
          if design.die_size_x_mm>0 and design.die_size_y_mm>0 else None)
    toks += _geom_tokens(n, xr, yr, ar)
    toks.append(_t("proc_T4" if design.process_level=='T4' else "proc_other"))
    eq=design.equipment_id.lower()
    toks.append(_t("equip_S500" if 's500' in eq else "equip_4200A" if '4200' in eq else "equip_other"))
    fn=design.raw_filename.lower()
    for kw,tn in [('tvsun','lot_tvsun'),('tvsti','lot_tvsti'),('tvs','lot_tvs'),
                  ('tvd','lot_tvd'),('scr','lot_scr')]:
        if kw in fn: toks.append(_t(tn)); break
    else: toks.append(_t("lot_unknown"))
    return toks

# ── Training dataset ──────────────────────────────────────────────────────────

def _make_training_data():
    ex = []
    def add(toks, lbl, repeat=1):
        for _ in range(repeat): ex.append((toks, lbl))

    # SCR_DC  (class 0)
    base0 = [CLS_ID,_t("sites_small"),_t("xrange_medium"),_t("yrange_medium"),
              _t("grid_medium"),_t("proc_T4"),_t("equip_S500"),
              _t("test_BVr_250uA"),_t("test_BVr_1mA"),_t("test_BVr_20mA"),
              _t("test_IR_3p3V"),_t("test_Vf_10mA"),
              _t("param_Vbr"),_t("param_Vbr_250uA"),_t("param_Vbr_1mA"),
              _t("param_Vbr_20mA"),_t("param_Ileakage"),_t("param_Vf"),_t("param_error"),
              _t("no_cv"),_t("has_bvr"),_t("has_ir"),_t("has_vf"),_t("no_cj"),
              _t("lot_tails"),_t("meas_dc")]
    add(base0, 0, 4)
    add(base0+[_t("lot_scr")], 0, 2)
    add(base0+[_t("test_BVr_5mA"),_t("test_BVr_6p5mA"),_t("test_BVr_neg1mA")], 0, 2)
    add(base0+[_t("test_IR_1V"),_t("test_IR_2V"),_t("test_IR_neg3p3V")], 0, 2)
    add([CLS_ID,_t("sites_medium"),_t("xrange_wide"),_t("yrange_vtall"),_t("grid_sparse"),
         _t("proc_T4"),_t("equip_S500"),_t("test_Vbr1"),_t("test_Vbr2"),
         _t("test_IR1"),_t("test_IR2"),_t("test_Vf"),_t("test_Con_1"),_t("test_Con_2"),_t("test_Con_3"),
         _t("param_Vbr"),_t("param_Ileakage"),_t("param_Vf"),
         _t("no_cv"),_t("has_bvr"),_t("has_ir"),_t("has_vf"),_t("no_cj"),
         _t("lot_scr"),_t("meas_dc")], 0, 2)

    # SCR_Cj  (class 1)
    base1 = [CLS_ID,_t("sites_small"),_t("xrange_medium"),_t("yrange_medium"),
              _t("grid_medium"),_t("proc_T4"),_t("equip_S500"),
              _t("test_CV_0V"),_t("test_CV_1p5V"),
              _t("param_Cp"),_t("param_Gp"),_t("param_DCV"),_t("param_F_freq"),
              _t("has_cv"),_t("no_bvr"),_t("no_ir"),_t("no_vf"),_t("has_cj"),
              _t("lot_tails"),_t("meas_cj")]
    add(base1, 1, 4)
    add(base1+[_t("lot_scr")], 1, 2)
    add([CLS_ID,_t("sites_small"),_t("xrange_medium"),_t("yrange_medium"),
         _t("grid_medium"),_t("proc_T4"),_t("equip_S500"),
         _t("test_CV_Sweep"),_t("test_CV_Sweep_2p5V"),_t("test_CV_Sweep_5p5V"),
         _t("param_Cp"),_t("param_Gp"),
         _t("has_cv"),_t("no_bvr"),_t("no_ir"),_t("no_vf"),_t("has_cj"),
         _t("lot_scr"),_t("meas_cv")], 1, 2)

    # SCR_Full  (class 2)
    base2 = [CLS_ID,_t("sites_small"),_t("xrange_medium"),_t("yrange_medium"),
              _t("grid_medium"),_t("proc_T4"),_t("equip_S500"),
              _t("test_CV_Sweep"),_t("test_CV_Sweep_2p5V"),_t("test_CV_Sweep_5p5V"),
              _t("test_BVr_250uA"),_t("test_BVr_1mA"),_t("test_BVr_20mA"),
              _t("test_IR_3p3V"),_t("test_Vf_10mA"),
              _t("test_Con_1"),_t("test_Con_2"),_t("test_Con_3"),
              _t("param_Cp"),_t("param_Gp"),_t("param_Vbr"),_t("param_Ileakage"),_t("param_Vf"),
              _t("has_cv"),_t("has_bvr"),_t("has_ir"),_t("has_vf"),_t("has_cj"),
              _t("lot_scr"),_t("meas_full")]
    add(base2, 2, 3)
    add(base2+[_t("test_BVr_5mA"),_t("test_BVr_neg1mA")], 2, 2)

    # TVD_DC  (class 3)
    base3 = [CLS_ID,_t("sites_medium"),_t("xrange_vwide"),_t("yrange_short"),
              _t("grid_sparse"),_t("proc_T4"),_t("equip_S500"),_t("ar_tall"),
              _t("test_Vbr1"),_t("test_Vbr2"),_t("test_IR1"),_t("test_IR2"),
              _t("test_IR_120V"),_t("test_Vf"),_t("test_BVr_10mA"),
              _t("test_Con_1"),_t("test_Con_2"),_t("test_Con_3"),
              _t("param_Vbr"),_t("param_Ileakage"),_t("param_Vf"),
              _t("has_cv"),_t("has_bvr"),_t("has_ir"),_t("has_vf"),_t("no_cj"),
              _t("lot_tvd"),_t("meas_dc")]
    add(base3, 3, 3)
    add([t for t in base3 if t!=_t("test_IR_120V")], 3, 2)

    # TVS_DC  (class 4)
    base4 = [CLS_ID,_t("sites_small"),_t("xrange_vwide"),_t("yrange_short"),
              _t("grid_sparse"),_t("proc_T4"),_t("equip_S500"),_t("ar_tall"),
              _t("test_Vbr1"),_t("test_Vbr2"),_t("test_IR1"),_t("test_IR2"),
              _t("test_Vf"),_t("test_Con_3"),
              _t("param_Vbr"),_t("param_Ileakage"),_t("param_Vf"),
              _t("has_cv"),_t("has_bvr"),_t("has_ir"),_t("has_vf"),_t("no_cj"),
              _t("lot_tvs"),_t("meas_dc")]
    add(base4, 4, 3)
    add(base4+[_t("test_BVr_250uA"),_t("test_BVr_1mA")], 4, 2)

    # TVSTI  (class 5)
    base5 = [CLS_ID,_t("sites_huge"),_t("xrange_wide"),_t("yrange_vtall"),
              _t("grid_dense"),_t("proc_T4"),_t("equip_S500"),_t("ar_wide"),
              _t("test_Vbr1"),_t("test_Vbr2"),_t("test_IR1"),_t("test_IR2"),
              _t("test_Vf"),_t("test_Con_1"),_t("test_Con_3"),
              _t("param_Vbr"),_t("param_Ileakage"),_t("param_Vf"),
              _t("has_cv"),_t("has_bvr"),_t("has_ir"),_t("has_vf"),_t("no_cj"),
              _t("lot_tvsti"),_t("meas_dc")]
    add(base5, 5, 3)

    # TVSUN  (class 6)
    base6 = [CLS_ID,_t("sites_large"),_t("xrange_wide"),_t("yrange_tall"),
              _t("grid_medium"),_t("proc_T4"),_t("equip_S500"),_t("ar_square"),
              _t("test_Vbr1"),_t("test_Vbr2"),_t("test_IR1"),_t("test_IR2"),
              _t("test_Vf"),_t("test_Con_3"),
              _t("param_Vbr"),_t("param_Ileakage"),_t("param_Vf"),
              _t("has_cv"),_t("has_bvr"),_t("has_ir"),_t("has_vf"),_t("no_cj"),
              _t("lot_tvsun"),_t("meas_dc")]
    add(base6, 6, 3)

    # Unknown DC  (class 7)
    add([CLS_ID,_t("sites_small"),_t("xrange_medium"),_t("yrange_medium"),
         _t("proc_T4"),_t("test_BVr_1mA"),_t("param_Vbr"),
         _t("no_cv"),_t("has_bvr"),_t("no_ir"),_t("no_vf"),_t("no_cj"),
         _t("lot_unknown"),_t("meas_dc")], 7, 2)

    # Unknown CV  (class 8)
    add([CLS_ID,_t("sites_small"),_t("xrange_medium"),_t("yrange_medium"),
         _t("proc_T4"),_t("test_CV_0V"),_t("param_Cp"),
         _t("has_cv"),_t("no_bvr"),_t("no_ir"),_t("no_vf"),_t("has_cj"),
         _t("lot_unknown"),_t("meas_cj")], 8, 2)

    return ex

# ── Transformer (NumPy) ───────────────────────────────────────────────────────

D=64; H=4; DH=D//H; DF=128; L=2; SEQ=80

def _softmax(x):
    e=np.exp(x-np.max(x)); return e/e.sum()

def _ln(x,g,b):
    mu=x.mean(-1,keepdims=True)
    s=np.sqrt(((x-mu)**2).mean(-1,keepdims=True)+1e-6)
    return g*(x-mu)/s+b

def _mha(x,Wq,Wk,Wv,Wo):
    S=x.shape[0]
    Q=(x@Wq).reshape(S,H,DH).transpose(1,0,2)
    K=(x@Wk).reshape(S,H,DH).transpose(1,0,2)
    V=(x@Wv).reshape(S,H,DH).transpose(1,0,2)
    a=_softmax_2d(Q@K.transpose(0,2,1)/math.sqrt(DH))
    return (a@V).transpose(1,0,2).reshape(S,D)@Wo

def _softmax_2d(x):
    e=np.exp(x-np.max(x,axis=-1,keepdims=True)); return e/e.sum(axis=-1,keepdims=True)

def _ffn(x,W1,b1,W2,b2):
    return np.maximum(0,x@W1+b1)@W2+b2

def _encode(ids, P):
    S=min(len(ids),SEQ)
    pad=[ids[i] if i<S else PAD_ID for i in range(SEQ)]
    x=P['emb'][pad]+P['pos']
    for l in range(L):
        p=P[l]
        x=_ln(x+_mha(x,p['Wq'],p['Wk'],p['Wv'],p['Wo']),p['g1'],p['b1'])
        x=_ln(x+_ffn(x,p['W1'],p['c1'],p['W2'],p['c2']),p['g2'],p['b2'])
    return x[0]@P['hW']+P['hb']

def _init(rng):
    P={'emb':rng.normal(0,0.02,(VOCAB_SIZE,D)).astype('f'),
       'pos':rng.normal(0,0.02,(SEQ,D)).astype('f'),
       'hW':rng.normal(0,0.02,(D,N_CLASSES)).astype('f'),
       'hb':np.zeros(N_CLASSES,'f')}
    for l in range(L):
        s=math.sqrt(2/D); sf=math.sqrt(2/DF)
        P[l]={'Wq':rng.normal(0,s,(D,D)).astype('f'),
              'Wk':rng.normal(0,s,(D,D)).astype('f'),
              'Wv':rng.normal(0,s,(D,D)).astype('f'),
              'Wo':rng.normal(0,s,(D,D)).astype('f'),
              'W1':rng.normal(0,s,(D,DF)).astype('f'),'c1':np.zeros(DF,'f'),
              'W2':rng.normal(0,sf,(DF,D)).astype('f'),'c2':np.zeros(D,'f'),
              'g1':np.ones(D,'f'),'b1':np.zeros(D,'f'),
              'g2':np.ones(D,'f'),'b2':np.zeros(D,'f')}
    return P

def train_classifier(n_epochs=500, lr=4e-3, seed=42):
    rng=np.random.default_rng(seed); P=_init(rng)
    data=_make_training_data()
    # Adam state mirrors param structure
    def _az(p):
        if isinstance(p,dict): return {k:_az(v) for k,v in p.items()}
        return np.zeros_like(p)
    M=_az(P); V=_az(P); b1=0.9; b2=0.999; ep=1e-8; step=0
    def upd(p,g,m,v,t):
        m[:]=b1*m+(1-b1)*g; v[:]=b2*v+(1-b2)*g**2
        mh=m/(1-b1**t); vh=v/(1-b2**t)
        p-=lr*mh/(np.sqrt(vh)+ep)
    for _ in range(n_epochs):
        for idx in rng.permutation(len(data)):
            toks,label=data[idx]; step+=1
            logits=_encode(toks,P)
            probs=_softmax(logits)
            dL=probs.copy(); dL[label]-=1.0
            # Head gradients
            S=min(len(toks),SEQ)
            pad=[toks[i] if i<S else PAD_ID for i in range(SEQ)]
            x=P['emb'][pad]+P['pos']
            for l in range(L):
                p=P[l]
                x=_ln(x+_mha(x,p['Wq'],p['Wk'],p['Wv'],p['Wo']),p['g1'],p['b1'])
                x=_ln(x+_ffn(x,p['W1'],p['c1'],p['W2'],p['c2']),p['g2'],p['b2'])
            cls=x[0]
            dW=np.outer(cls,dL); db=dL; dcls=P['hW']@dL
            upd(P['hW'],dW,M['hW'],V['hW'],step)
            upd(P['hb'],db,M['hb'],V['hb'],step)
            # Embed gradient (CLS + context)
            de=np.zeros_like(P['emb'])
            de[pad[0]]+=dcls
            for i,tid in enumerate(pad[:S]): de[tid]+=dcls*0.08
            upd(P['emb'],de,M['emb'],V['emb'],step)
    return P

class WaferClassifier:
    _inst=None
    def __init__(self): self.P=train_classifier()
    @classmethod
    def get(cls):
        if cls._inst is None: cls._inst=cls()
        return cls._inst
    def _run(self, toks):
        logits=_encode(toks,self.P); probs=_softmax(logits)
        return sorted([{"class":CLASSES[i],"confidence":float(p),"class_id":i}
                        for i,p in enumerate(probs)], key=lambda x:-x["confidence"])
    def classify_kdf(self, header, sites, params, tests):
        return self._run(extract_tokens_from_kdf(header,sites,params,tests))
    def classify_xml(self, design):
        return self._run(extract_tokens_from_xml(design))

# ═══════════════════════════════════════════════════════════════════════════════
#  THEME
# ═══════════════════════════════════════════════════════════════════════════════

T = {
    "bg_app":"#f0f3f7","bg_panel":"#ffffff","bg_widget":"#f7f9fb",
    "bg_header":"#e8edf4","bg_row_alt":"#f3f6fa",
    "border":"#cdd6e3","border_hi":"#a0b4cc",
    "accent":"#1565c0","accent_dim":"#dce8fb","accent_dark":"#0d47a1",
    "pass_bg":"#e3f6ec","pass_fg":"#1b6b3a","pass_border":"#5cb87a",
    "neutral_bg":"#dfe8f3","neutral_fg":"#2c4a6e","neutral_border":"#90a8c4",
    "fail_bg":"#fce8e8","fail_fg":"#b71c1c","fail_border":"#e57373",
    "nodata_bg":"#eeecf5","nodata_fg":"#7060a0","nodata_border":"#b0a8d0",
    "margin_bg":"#eae7f2","margin_fg":"#7a7090","margin_border":"#b8b0cc",
    "wafer_bg":"#eef2f8","wafer_edge":"#7a9cbd",
    "text_primary":"#1a2537","text_secondary":"#4a5c72","text_dim":"#8fa4bc",
    "selected":"#e65100","hover_border":"#1565c0","warn":"#e65100",
    "conf_hi":"#1b6b3a","conf_med":"#7a5800","conf_lo":"#7060a0",
}

SS = """
QMainWindow,QWidget{background-color:"""+T['bg_app']+""";color:"""+T['text_primary']+""";font-family:'Segoe UI','Calibri',sans-serif;font-size:13px;}
QGroupBox{background-color:"""+T['bg_panel']+""";border:1px solid """+T['border']+""";border-radius:6px;margin-top:22px;padding:8px 6px 6px 6px;font-size:11px;font-weight:bold;}
QGroupBox::title{subcontrol-origin:margin;left:10px;padding:2px 6px;background:"""+T['bg_panel']+""";color:"""+T['accent']+""";font-size:11px;font-weight:bold;}
QLabel{background:transparent;color:"""+T['text_primary']+""";font-size:13px;}
QPushButton{background-color:"""+T['bg_panel']+""";border:1px solid """+T['border_hi']+""";border-radius:5px;padding:6px 16px;color:"""+T['text_primary']+""";font-weight:600;font-size:13px;min-height:28px;}
QPushButton:hover{background-color:"""+T['accent_dim']+""";border:1px solid """+T['accent']+""";color:"""+T['accent_dark']+""";}
QPushButton:pressed{background-color:"""+T['accent']+""";color:white;}
QPushButton#primary{background-color:"""+T['accent']+""";border:1px solid """+T['accent_dark']+""";color:white;font-weight:bold;}
QPushButton#primary:hover{background-color:"""+T['accent_dark']+""";}
QComboBox{background-color:"""+T['bg_panel']+""";border:1px solid """+T['border']+""";border-radius:5px;padding:5px 10px;color:"""+T['text_primary']+""";font-size:13px;min-height:28px;}
QComboBox::drop-down{border:none;width:24px;}
QComboBox::down-arrow{width:9px;height:7px;border-left:5px solid transparent;border-right:5px solid transparent;border-top:7px solid """+T['accent']+""";}
QComboBox QAbstractItemView{background-color:"""+T['bg_panel']+""";border:1px solid """+T['border_hi']+""";selection-background-color:"""+T['accent_dim']+""";color:"""+T['text_primary']+""";font-size:13px;}
QLineEdit{background-color:"""+T['bg_panel']+""";border:1px solid """+T['border']+""";border-radius:5px;padding:5px 10px;color:"""+T['text_primary']+""";font-size:13px;min-height:28px;}
QLineEdit:focus{border:1px solid """+T['accent']+""";}
QTreeWidget,QTableWidget{background-color:"""+T['bg_panel']+""";border:1px solid """+T['border']+""";border-radius:4px;alternate-background-color:"""+T['bg_row_alt']+""";outline:none;font-size:13px;}
QTreeWidget::item,QTableWidget::item{padding:4px 5px;border:none;}
QTreeWidget::item:hover,QTableWidget::item:hover{background-color:"""+T['accent_dim']+""";}
QTreeWidget::item:selected,QTableWidget::item:selected{background-color:"""+T['accent_dim']+""";color:"""+T['accent_dark']+""";}
QHeaderView::section{background-color:"""+T['bg_header']+""";color:"""+T['accent_dark']+""";border:none;border-right:1px solid """+T['border']+""";border-bottom:1px solid """+T['border']+""";padding:6px 10px;font-size:12px;font-weight:bold;}
QScrollBar:vertical{background:"""+T['bg_app']+""";width:9px;border:none;border-radius:4px;}
QScrollBar::handle:vertical{background:"""+T['border_hi']+""";border-radius:4px;min-height:24px;}
QScrollBar::handle:vertical:hover{background:"""+T['accent']+""";}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0px;}
QScrollBar:horizontal{background:"""+T['bg_app']+""";height:9px;border-radius:4px;}
QScrollBar::handle:horizontal{background:"""+T['border_hi']+""";border-radius:4px;}
QTabWidget::pane{border:1px solid """+T['border']+""";background-color:"""+T['bg_panel']+""";border-radius:0 5px 5px 5px;}
QTabBar::tab{background-color:"""+T['bg_header']+""";color:"""+T['text_secondary']+""";padding:7px 18px;border:1px solid """+T['border']+""";border-bottom:none;border-radius:5px 5px 0 0;margin-right:2px;font-size:12px;}
QTabBar::tab:selected{background-color:"""+T['bg_panel']+""";color:"""+T['accent']+""";border-top:2px solid """+T['accent']+""";font-weight:bold;font-size:13px;}
QStatusBar{background-color:"""+T['bg_header']+""";color:"""+T['text_secondary']+""";border-top:1px solid """+T['border']+""";font-size:12px;padding:3px 6px;}
QToolBar{background-color:"""+T['bg_panel']+""";border-bottom:1px solid """+T['border']+""";spacing:4px;padding:5px 10px;}
QToolBar::separator{background:"""+T['border']+""";width:1px;margin:4px 8px;}
QDoubleSpinBox{background-color:"""+T['bg_panel']+""";border:1px solid """+T['border']+""";border-radius:5px;padding:4px 8px;color:"""+T['text_primary']+""";font-size:13px;min-height:28px;}
QCheckBox{color:"""+T['text_primary']+""";spacing:7px;font-size:13px;}
QCheckBox::indicator{width:15px;height:15px;border:1px solid """+T['border_hi']+""";border-radius:3px;background:"""+T['bg_panel']+""";}
QCheckBox::indicator:checked{background:"""+T['accent']+""";border:1px solid """+T['accent_dark']+""";}
QProgressBar{border:1px solid """+T['border']+""";border-radius:4px;background:"""+T['bg_header']+""";min-height:12px;max-height:12px;}
QProgressBar::chunk{background-color:"""+T['accent']+""";border-radius:3px;}
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  ICON
# ═══════════════════════════════════════════════════════════════════════════════

def make_app_icon():
    icon=QIcon()
    for sz in (16,24,32,48,64,128):
        pix=QPixmap(sz,sz); pix.fill(Qt.transparent)
        p=QPainter(pix); p.setRenderHint(QPainter.Antialiasing)
        cx=cy=sz/2.0; r=sz/2.0-1.0
        d=QRadialGradient(cx-r*0.2,cy-r*0.25,r*1.35)
        d.setColorAt(0,QColor("#2d5490")); d.setColorAt(0.65,QColor("#1a3460")); d.setColorAt(1.0,QColor("#0c1e3a"))
        p.setBrush(QBrush(d)); p.setPen(QPen(QColor("#5588cc"),max(1.0,sz/18.0)))
        p.drawEllipse(QPointF(cx,cy),r,r)
        nw=r*0.36; nh=max(2.0,sz*0.05)
        p.setCompositionMode(QPainter.CompositionMode_Clear); p.setPen(Qt.NoPen); p.setBrush(Qt.transparent)
        p.drawRect(QRectF(cx-nw/2,cy+r-nh,nw,nh+1))
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setPen(QPen(QColor("#5588cc"),max(1.0,sz/20.0)))
        p.drawLine(QPointF(cx-nw/2,cy+r-nh),QPointF(cx+nw/2,cy+r-nh))
        dc=[QColor("#3dba6f"),QColor("#3dba6f"),QColor("#8a98b0"),
            QColor("#3dba6f"),QColor("#3dba6f"),QColor("#d95555"),
            QColor("#8a98b0"),QColor("#3dba6f"),QColor("#3dba6f")]
        ge=r*1.08; cs=ge/3.0; gap=max(0.6,cs*0.12); gx0=cx-ge/2; gy0=cy-ge/2
        p.setPen(QPen(QColor("#0c1e3a"),max(0.5,gap*0.4)))
        for row in range(3):
            for col in range(3):
                rx=gx0+col*cs+gap; ry=gy0+row*cs+gap; rw=rh=cs-gap*2
                if rw>0: p.setBrush(QBrush(dc[row*3+col])); p.drawRoundedRect(QRectF(rx,ry,rw,rh),max(0.5,rw*0.18),max(0.5,rw*0.18))
        p.end(); icon.addPixmap(pix)
    return icon

# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSIFIER RESULT PANEL
# ═══════════════════════════════════════════════════════════════════════════════

class ClassifierPanel(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        lo=QVBoxLayout(self); lo.setContentsMargins(8,8,8,8); lo.setSpacing(8)

        self.banner=QLabel("\u26a1  WAFER TYPE IDENTIFICATION")
        self.banner.setStyleSheet(f"font-weight:bold;font-size:12px;letter-spacing:1px;color:{T['accent_dark']};padding:5px 8px;background:{T['accent_dim']};border-radius:5px;border-left:3px solid {T['accent']};")
        lo.addWidget(self.banner)

        self.top_name=QLabel("\u2014")
        self.top_name.setWordWrap(True)
        self.top_name.setStyleSheet(f"font-size:16px;font-weight:bold;color:{T['text_primary']};padding:10px 10px 2px 10px;")
        lo.addWidget(self.top_name)

        self.top_pct=QLabel("\u2014")
        self.top_pct.setStyleSheet(f"font-size:13px;color:{T['conf_hi']};padding:0 10px 4px 10px;font-weight:bold;")
        lo.addWidget(self.top_pct)

        self.top_bar=QProgressBar(); self.top_bar.setRange(0,100)
        lo.addWidget(self.top_bar)

        div=QFrame(); div.setFrameShape(QFrame.HLine); div.setStyleSheet(f"color:{T['border']};"); lo.addWidget(div)

        alt_h=QLabel("Alternative interpretations")
        alt_h.setStyleSheet(f"color:{T['text_secondary']};font-size:11px;font-weight:bold;")
        lo.addWidget(alt_h)

        self._alts=[]
        for _ in range(2):
            w=QWidget(); wl=QVBoxLayout(w); wl.setContentsMargins(0,2,0,2); wl.setSpacing(2)
            n=QLabel("\u2014"); n.setStyleSheet(f"color:{T['text_secondary']};font-size:12px;")
            b=QProgressBar(); b.setRange(0,100)
            wl.addWidget(n); wl.addWidget(b); lo.addWidget(w); self._alts.append((n,b))

        div2=QFrame(); div2.setFrameShape(QFrame.HLine); div2.setStyleSheet(f"color:{T['border']};"); lo.addWidget(div2)

        tl=QLabel("Signal tokens detected")
        tl.setStyleSheet(f"color:{T['text_secondary']};font-size:11px;font-weight:bold;"); lo.addWidget(tl)

        self.tok_disp=QLabel("\u2014")
        self.tok_disp.setWordWrap(True)
        self.tok_disp.setStyleSheet(f"color:{T['text_dim']};font-size:10px;font-family:'Consolas',monospace;padding:4px 6px;background:{T['bg_widget']};border:1px solid {T['border']};border-radius:4px;")
        lo.addWidget(self.tok_disp)
        lo.addStretch()

    def show_result(self, results, tokens=None):
        if not results: return
        top=results[0]; pct=top['confidence']*100
        self.top_name.setText(top['class'])
        self.top_pct.setText(f"{pct:.1f}%  confidence")
        self.top_bar.setValue(int(pct))
        col=T['conf_hi'] if pct>=75 else T['conf_med'] if pct>=50 else T['conf_lo']
        self.top_pct.setStyleSheet(f"font-size:13px;color:{col};padding:0 10px 4px 10px;font-weight:bold;")
        self.top_bar.setStyleSheet(f"QProgressBar::chunk{{background-color:{col};border-radius:3px;}}QProgressBar{{border:1px solid {T['border']};border-radius:4px;background:{T['bg_header']};min-height:12px;max-height:12px;}}")
        for i,(nl,bl) in enumerate(self._alts):
            if i+1<len(results):
                r=results[i+1]; p2=r['confidence']*100
                nl.setText(f"{r['class']}  ({p2:.1f}%)"); bl.setValue(int(p2))
            else: nl.setText("\u2014"); bl.setValue(0)
        if tokens:
            words=[_VOCAB_LIST[tid] for tid in tokens if 0<=tid<len(_VOCAB_LIST)
                   and _VOCAB_LIST[tid] not in ('[CLS]','[PAD]','[UNK]')]
            clean=[w.replace('test_','').replace('param_','').replace('lot_','') for w in words]
            self.tok_disp.setText("  \u00b7  ".join(clean[:22]))
        else: self.tok_disp.setText("\u2014")

# ═══════════════════════════════════════════════════════════════════════════════
#  XML DESIGN PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def _rgb(s):
    try: r,g,b=[int(x.strip()) for x in s.split(',')]; return QColor(r,g,b)
    except: return None

def _site_xy(name):
    m=re.match(r'Site_([pn])(\d+)([pn])(\d+)$',name.strip())
    if m:
        xs,xv,ys,yv=m.groups()
        return int(xv)*(1 if xs=='p' else -1), int(yv)*(1 if ys=='p' else -1)
    return None,None

class XMLDesign:
    def __init__(self):
        self.diameter_in=8.0; self.die_size_x_mm=0.0; self.die_size_y_mm=0.0
        self.orientation=''; self.origin_x=0; self.origin_y=0
        self.equipment_id=''; self.operator=''; self.process_level=''
        self.project=''; self.slots=[]; self.site_names=[]; self.site_coords={}
        self.margin_count=0; self.color_pass=None; self.color_fail=None
        self.raw_filename=''

def load_xml_design(path):
    d=XMLDesign(); d.raw_filename=os.path.basename(path)
    root=ET.parse(path).getroot(); head=root.find('head')
    if head is not None:
        try: d.diameter_in=float(head.findtext('diameter','8'))
        except: pass
        try: d.die_size_x_mm=float(head.findtext('diesizex','0')); d.die_size_y_mm=float(head.findtext('diesizey','0'))
        except: pass
        d.orientation=head.findtext('orientation','').strip(); d.project=head.findtext('project','').strip()
        try: d.margin_count=int(head.findtext('margin','0'))
        except: pass
        try: ox,oy=head.findtext('origin','0, 0').split(','); d.origin_x,d.origin_y=int(ox.strip()),int(oy.strip())
        except: pass
        cel=head.find('color')
        if cel is not None: d.color_pass=_rgb(cel.findtext('pass','') or ''); d.color_fail=_rgb(cel.findtext('fail','') or '')
    cdf=root.find('cdf')
    if cdf is not None:
        rep=cdf.find('report')
        if rep is not None: d.equipment_id=rep.findtext('equipment_id','').strip(); d.operator=rep.findtext('operator','').strip(); d.process_level=rep.findtext('test_process_level','').strip()
        se=cdf.find('slots')
        if se is not None:
            for slot in se:
                pts=[p.strip() for p in (slot.text or '').split(',')]
                if len(pts)>=2: d.slots.append((pts[0],pts[1]))
    pat=root.find('patterns')
    if pat is not None:
        p1=pat.find('Pattern_1')
        if p1 is not None:
            names=[s.strip() for s in re.split(r'[,\n\r]+',p1.findtext('sites','')) if s.strip()]
            d.site_names=names
            for n in names:
                x,y=_site_xy(n)
                if x is not None: d.site_coords[n]=(x,y)
    return d

# ═══════════════════════════════════════════════════════════════════════════════
#  KDF PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_kdf(filepath):
    header={}; sites=[]; ms=set(); ts=set()
    with open(filepath,'r',errors='replace') as f: lines=[l.strip() for l in f]
    i=0
    while i<len(lines):
        if lines[i]=='<EOH>': i+=1; break
        if ',' in lines[i]: k,_,v=lines[i].partition(','); header[k.strip()]=v.strip()
        i+=1
    if i<len(lines): i+=1
    cur=None
    while i<len(lines):
        ln=lines[i]; i+=1
        if not ln: continue
        if ln=='<EOS>':
            if cur: sites.append(cur)
            cur=None; continue
        if ln.startswith('Site_'):
            if cur: sites.append(cur)
            pts=ln.split(',')
            try: x,y=int(pts[1]),int(pts[2])
            except: x,y=0,0
            cur={'name':pts[0],'x':x,'y':y,'subsites':{}}; continue
        if cur is None: continue
        if '@' in ln and ',' in ln:
            kp,_,vs=ln.partition(','); pts=kp.split('@')
            if len(pts)>=3:
                param,test,sp=pts[0],pts[1],pts[2]
                try: sn=int(sp.split('#')[1])
                except: sn=1
                try: val=float(vs)
                except: val=None
                if sn not in cur['subsites']: cur['subsites'][sn]={}
                mk=f"{param}@{test}"; cur['subsites'][sn][mk]=val; ms.add(mk); ts.add(test)
    if cur: sites.append(cur)
    return header,sites,sorted(ms),sorted(ts)

def get_site_value(site,mkey,subsite=None):
    vals=[]
    for sn,data in site['subsites'].items():
        if subsite is not None and sn!=subsite: continue
        v=data.get(mkey)
        if v is not None and math.isfinite(v): vals.append(v)
    return float(np.mean(vals)) if vals else None

# ═══════════════════════════════════════════════════════════════════════════════
#  WAFER CANVAS
# ═══════════════════════════════════════════════════════════════════════════════

class WaferCanvas(QWidget):
    siteClicked=Signal(dict)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.sites=[]; self.values={}; self.low_limit=None; self.high_limit=None
        self.selected_site=None; self.mkey=''; self._hover=None
        self._rects={}; self.design=None; self._ghost={}
        self.setMinimumSize(400,400); self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding); self.setMouseTracking(True)

    def load(self,sites,values,lo,hi,mkey=''):
        self.sites=sites; self.values=values; self.low_limit=lo; self.high_limit=hi; self.mkey=mkey
        self.selected_site=None; self._hover=None; self._rebuild(); self.update()

    def set_design(self,d):
        self.design=d; self._rebuild(); self.update()

    def _rebuild(self):
        self._ghost={}
        if self.design is None: return
        kn={s['name'] for s in self.sites}
        for nm in self.design.site_names:
            if nm not in kn and nm in self.design.site_coords:
                x,y=self.design.site_coords[nm]; self._ghost[nm]={'name':nm,'x':x,'y':y,'subsites':{}}

    @property
    def _lim(self): return self.low_limit is not None or self.high_limit is not None

    def _bounds(self):
        a=list(self.sites)+list(self._ghost.values())
        if not a: return -1,1,-1,1
        xs=[s['x'] for s in a]; ys=[s['y'] for s in a]
        return min(xs),max(xs),min(ys),max(ys)

    def _lay(self,x0,x1,y0,y1,w,h):
        pad=64; c=min((w-2*pad)/max(x1-x0+1,1),(h-2*pad)/max(y1-y0+1,1))
        return (w-c*(x1-x0+1))/2,(h-c*(y1-y0+1))/2,c

    def _col(self,nm,ghost=False):
        if ghost: return QColor(T["margin_bg"]),QColor(T["margin_fg"]),QColor(T["margin_border"])
        v=self.values.get(nm)
        if v is None: return QColor(T["nodata_bg"]),QColor(T["nodata_fg"]),QColor(T["nodata_border"])
        if not self._lim: return QColor(T["neutral_bg"]),QColor(T["neutral_fg"]),QColor(T["neutral_border"])
        lo=self.low_limit; hi=self.high_limit
        ok=(lo is None or v>=lo) and (hi is None or v<=hi)
        return (QColor(T["pass_bg"]),QColor(T["pass_fg"]),QColor(T["pass_border"])) if ok else (QColor(T["fail_bg"]),QColor(T["fail_fg"]),QColor(T["fail_border"]))

    def paintEvent(self,e):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.setRenderHint(QPainter.TextAntialiasing)
        w,h=self.width(),self.height(); p.fillRect(0,0,w,h,QColor(T["bg_app"]))
        p.setPen(QPen(QColor(T["border"]),1.0))
        for gx in range(0,w+28,28):
            for gy in range(0,h+28,28): p.drawPoint(gx,gy)
        a=list(self.sites)+list(self._ghost.values())
        if not a:
            p.setPen(QColor(T["text_dim"])); p.setFont(QFont('Segoe UI',14))
            p.drawText(self.rect(),Qt.AlignCenter,'Load a KDF file to begin'); return
        x0,x1,y0,y1=self._bounds(); ox,oy,cell=self._lay(x0,x1,y0,y1,w,h)
        cols=x1-x0+1; rows=y1-y0+1; cx=ox+cell*cols/2; cy=oy+cell*rows/2
        if self.design and self.design.die_size_x_mm>0:
            ppm=cell/max(self.design.die_size_x_mm,self.design.die_size_y_mm)
            rad=(self.design.diameter_in*25.4/2)*ppm
        else: rad=min(cell*cols,cell*rows)/2+cell*0.6
        sh=QRadialGradient(cx+5,cy+5,rad+8); sh.setColorAt(0,QColor(0,0,0,28)); sh.setColorAt(0.8,QColor(0,0,0,10)); sh.setColorAt(1,QColor(0,0,0,0))
        p.setBrush(QBrush(sh)); p.setPen(Qt.NoPen); p.drawEllipse(QPointF(cx+5,cy+5),rad+8,rad+8)
        wg=QRadialGradient(cx-rad*0.2,cy-rad*0.25,rad*1.4)
        wg.setColorAt(0,QColor("#fff")); wg.setColorAt(0.55,QColor(T["wafer_bg"])); wg.setColorAt(1,QColor("#d8e3ef"))
        p.setBrush(QBrush(wg)); p.setPen(QPen(QColor(T["wafer_edge"]),1.5)); p.drawEllipse(QPointF(cx,cy),rad,rad)
        nw=rad*0.24; p.setPen(Qt.NoPen); p.setBrush(QColor(T["bg_app"]))
        p.drawRect(QRectF(cx-nw/2,cy+rad-6,nw,10)); p.setPen(QPen(QColor(T["wafer_edge"]),2))
        p.drawLine(QPointF(cx-nw/2,cy+rad-4),QPointF(cx+nw/2,cy+rad-4))
        fs=max(7,int(cell*0.15)); vf=QFont('Consolas',fs,QFont.Bold); cf=QFont('Consolas',max(6,fs-2))
        self._rects={}
        for nm,gs in self._ghost.items():
            sx,sy=gs['x'],gs['y']; px=ox+(sx-x0)*cell; py=oy+(y1-sy)*cell
            mg=max(1.5,cell*0.04); rc=QRectF(px+mg,py+mg,cell-2*mg,cell-2*mg)
            bg,fg,bc=self._col(nm,True); p.setBrush(QBrush(bg)); p.setPen(QPen(bc,0.5,Qt.DotLine)); p.drawRoundedRect(rc,3,3)
        for site in self.sites:
            sx,sy=site['x'],site['y']; px=ox+(sx-x0)*cell; py=oy+(y1-sy)*cell
            mg=max(1.5,cell*0.04); rc=QRectF(px+mg,py+mg,cell-2*mg,cell-2*mg); self._rects[site['name']]=rc
            bg,fg,bc=self._col(site['name'])
            issel=self.selected_site and site['name']==self.selected_site['name']
            ishov=self._hover and site['name']==self._hover['name']
            if cell>30: p.setBrush(QColor(0,0,0,14)); p.setPen(Qt.NoPen); p.drawRoundedRect(QRectF(rc.x()+2,rc.y()+2,rc.width(),rc.height()),3,3)
            cg=QLinearGradient(rc.topLeft(),rc.bottomRight()); cg.setColorAt(0,bg.lighter(108)); cg.setColorAt(1,bg)
            p.setBrush(QBrush(cg))
            p.setPen(QPen(QColor(T["selected"]),2.5) if issel else QPen(QColor(T["hover_border"]),2) if ishov else QPen(bc,1))
            p.drawRoundedRect(rc,3,3)
            v=self.values.get(site['name'])
            if v is not None and cell>28: p.setFont(vf); p.setPen(fg); p.drawText(rc,Qt.AlignCenter,self._fmt(v))
            if cell>58:
                p.setFont(cf); p.setPen(QColor(T["text_dim"]))
                p.drawText(QRectF(px+mg+2,py+mg+1,cell-2*mg-2,cell*0.28),Qt.AlignLeft|Qt.AlignTop,f'{sx},{sy}')
        self._leg(p,w,h)
        if self.design: self._badge(p,w)

    def _fmt(self,v):
        if v is None: return 'N/A'
        av=abs(v)
        if av==0: return '0'
        if av>=1: return f'{v:.3g}'
        if av>=1e-3: return f'{v*1e3:.3g}m'
        if av>=1e-6: return f'{v*1e6:.3g}\u00b5'
        if av>=1e-9: return f'{v*1e9:.3g}n'
        if av>=1e-12: return f'{v*1e12:.3g}p'
        return f'{v:.3e}'

    def _leg(self,p,w,h):
        items=(([(QColor(T["pass_bg"]),QColor(T["pass_fg"]),'Pass'),(QColor(T["fail_bg"]),QColor(T["fail_fg"]),'Fail'),(QColor(T["nodata_bg"]),QColor(T["nodata_fg"]),'No data')]
                if self._lim else
                [(QColor(T["neutral_bg"]),QColor(T["neutral_fg"]),'No limits set'),(QColor(T["nodata_bg"]),QColor(T["nodata_fg"]),'No data')]))
        if self._ghost: items=list(items)+[(QColor(T["margin_bg"]),QColor(T["margin_fg"]),'Design only')]
        bh=len(items)*22+16; lx=14; ly=h-bh-10
        p.setBrush(QColor(255,255,255,215)); p.setPen(QPen(QColor(T["border"]),1)); p.drawRoundedRect(QRectF(lx-6,ly-8,178,bh),5,5)
        p.setFont(QFont('Segoe UI',11))
        for bg,fg,lbl in items:
            p.setBrush(QBrush(bg)); p.setPen(QPen(fg,1)); p.drawRoundedRect(QRectF(lx,ly,14,14),2,2)
            p.setPen(QColor(T["text_primary"])); p.drawText(int(lx+20),int(ly+11),lbl); ly+=22

    def _badge(self,p,w):
        d=self.design
        lines=[f"\u2300 {d.diameter_in}\"  {d.die_size_x_mm:.2f}\u00d7{d.die_size_y_mm:.2f} mm"]
        if d.equipment_id: lines.append(d.equipment_id)
        bw=210; bh=len(lines)*18+12; rx=w-bw-12; ry=10
        p.setBrush(QColor(255,255,255,200)); p.setPen(QPen(QColor(T["border"]),1)); p.drawRoundedRect(QRectF(rx,ry,bw,bh),5,5)
        p.setFont(QFont('Segoe UI',10)); p.setPen(QColor(T["text_secondary"]))
        for i,ln in enumerate(lines): p.drawText(int(rx+8),int(ry+16+i*18),ln)

    def mouseMoveEvent(self,e):
        pos=QPointF(e.position()); self._hover=None
        for s in self.sites:
            r=self._rects.get(s['name'])
            if r and r.contains(pos): self._hover=s; self.setCursor(Qt.PointingHandCursor); break
        else: self.setCursor(Qt.ArrowCursor)
        self.update()

    def mousePressEvent(self,e):
        if e.button()==Qt.LeftButton:
            pos=QPointF(e.position())
            for s in self.sites:
                r=self._rects.get(s['name'])
                if r and r.contains(pos): self.selected_site=s; self.siteClicked.emit(s); self.update(); return

    def leaveEvent(self,e): self._hover=None; self.update()

# ═══════════════════════════════════════════════════════════════════════════════
#  STATS + DETAIL + DESIGN PANELS
# ═══════════════════════════════════════════════════════════════════════════════

class StatsPanel(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        lo=QVBoxLayout(self); lo.setContentsMargins(8,8,8,8); lo.setSpacing(6)
        self.table=QTableWidget(0,2); self.table.setHorizontalHeaderLabels(['Statistic','Value'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers); self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False); self.table.setShowGrid(False); lo.addWidget(self.table)

    def update_stats(self,vd,lo,hi):
        vals=[v for v in vd.values() if v is not None and math.isfinite(v)]
        if not vals: self.table.setRowCount(0); return
        arr=np.array(vals); passed=sum(1 for v in arr if (lo is None or v>=lo) and (hi is None or v<=hi))
        total=len(arr); yp=passed/total*100 if total else 0
        rows=[('Count',str(total)),('Mean',self._f(arr.mean())),('Std Dev',self._f(arr.std())),
              ('Min',self._f(arr.min())),('Max',self._f(arr.max())),('Median',self._f(float(np.median(arr)))),
              ('3\u03c3',f'{self._f(arr.mean()-3*arr.std())} \u2192 {self._f(arr.mean()+3*arr.std())}'),
              ('Pass',str(passed)),('Fail',str(total-passed)),('Yield',f'{yp:.1f}%')]
        self.table.setRowCount(len(rows))
        for i,(k,v) in enumerate(rows):
            ki=QTableWidgetItem(k); ki.setForeground(QColor(T["text_secondary"])); ki.setFont(QFont('Segoe UI',12)); self.table.setItem(i,0,ki)
            vi=QTableWidgetItem(v); vi.setFont(QFont('Consolas',12))
            if k=='Yield':
                c=T["pass_fg"] if yp>=90 else T["fail_fg"] if yp<70 else T["warn"]
                vi.setForeground(QColor(c)); vi.setFont(QFont('Segoe UI',13,QFont.Bold))
            elif k=='Pass': vi.setForeground(QColor(T["pass_fg"]))
            elif k=='Fail': vi.setForeground(QColor(T["fail_fg"]))
            else: vi.setForeground(QColor(T["text_primary"]))
            self.table.setItem(i,1,vi)

    def _f(self,v):
        av=abs(v)
        if av==0: return '0'
        if av>=1: return f'{v:.5g}'
        if av>=1e-3: return f'{v*1e3:.4g} m'
        if av>=1e-6: return f'{v*1e6:.4g} \u00b5'
        if av>=1e-9: return f'{v*1e9:.4g} n'
        if av>=1e-12: return f'{v*1e12:.4g} p'
        return f'{v:.4e}'


class SiteDetailPanel(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        lo=QVBoxLayout(self); lo.setContentsMargins(8,8,8,8); lo.setSpacing(8)
        self.title=QLabel('Click a die to inspect')
        self.title.setStyleSheet(f'font-weight:bold;font-size:13px;color:{T["accent_dark"]};padding:5px 8px;background:{T["accent_dim"]};border-radius:5px;border-left:3px solid {T["accent"]};')
        lo.addWidget(self.title)
        self.table=QTableWidget(0,3); self.table.setHorizontalHeaderLabels(['Measurement','Subsite','Value'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers); self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False); self.table.setShowGrid(False); lo.addWidget(self.table)

    def show_site(self,site):
        self.title.setText(f'  {site["name"]}   \u00b7   X={site["x"]},  Y={site["y"]}')
        rows=[]
        for sn in sorted(site['subsites'].keys()):
            for mk,val in sorted(site['subsites'][sn].items()): rows.append((mk,f'#{sn}',val))
        self.table.setRowCount(len(rows))
        for i,(mk,sub,val) in enumerate(rows):
            mi=QTableWidgetItem(mk); mi.setFont(QFont('Segoe UI',12)); self.table.setItem(i,0,mi)
            si=QTableWidgetItem(sub); si.setForeground(QColor(T["text_secondary"])); si.setFont(QFont('Segoe UI',12)); self.table.setItem(i,1,si)
            vs=f'{val:.6g}' if val is not None else 'N/A'
            vi=QTableWidgetItem(vs); vi.setForeground(QColor(T["accent_dark"])); vi.setFont(QFont('Consolas',12,QFont.Bold)); self.table.setItem(i,2,vi)


class DesignInfoPanel(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        lo=QVBoxLayout(self); lo.setContentsMargins(8,8,8,8); lo.setSpacing(8)
        self.title=QLabel('No design file loaded')
        self.title.setStyleSheet(f'font-weight:bold;font-size:13px;color:{T["accent_dark"]};padding:5px 8px;background:{T["accent_dim"]};border-radius:5px;border-left:3px solid {T["accent"]};')
        lo.addWidget(self.title)
        for lbl_text in ['Wafer / Equipment','Cassette Slots']:
            l=QLabel(lbl_text); l.setStyleSheet(f'color:{T["text_secondary"]};font-size:11px;font-weight:bold;'); lo.addWidget(l)
            t=QTableWidget(0,2)
            if 'Equipment' in lbl_text:
                t.setHorizontalHeaderLabels(['Field','Value']); self.table=t
            else:
                t.setHorizontalHeaderLabels(['Cassette','Wafer ID']); t.setMaximumHeight(150); self.slots_table=t
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            t.setEditTriggers(QTableWidget.NoEditTriggers); t.setAlternatingRowColors(True)
            t.verticalHeader().setVisible(False); t.setShowGrid(False); lo.addWidget(t)
        lo.addStretch()

    def show_design(self,d):
        if d is None:
            self.title.setText('No design file loaded'); self.table.setRowCount(0); self.slots_table.setRowCount(0); return
        self.title.setText(f'  {d.raw_filename}')
        rows=[('Diameter',f'{d.diameter_in}"'),('Die size',f'{d.die_size_x_mm:.3g}\u00d7{d.die_size_y_mm:.3g} mm'),
              ('Orientation',d.orientation),('Project',d.project),('Equipment',d.equipment_id),
              ('Operator',d.operator),('Process level',d.process_level),
              ('Origin (X,Y)',f'{d.origin_x}, {d.origin_y}'),
              ('Design sites',str(len(d.site_names))),('Margin count',str(d.margin_count))]
        self.table.setRowCount(len(rows))
        for i,(k,v) in enumerate(rows):
            ki=QTableWidgetItem(k); ki.setForeground(QColor(T["text_secondary"])); ki.setFont(QFont('Segoe UI',12)); self.table.setItem(i,0,ki)
            vi=QTableWidgetItem(v); vi.setFont(QFont('Segoe UI',12)); self.table.setItem(i,1,vi)
        self.slots_table.setRowCount(len(d.slots))
        for i,(c,w2) in enumerate(d.slots): self.slots_table.setItem(i,0,QTableWidgetItem(c)); self.slots_table.setItem(i,1,QTableWidgetItem(w2))

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Wafer Map Viewer  \u00b7  AI Type Recognition')
        self.resize(1480,880); self.setStyleSheet(SS); self.setWindowIcon(make_app_icon())
        self._header={}; self._sites=[]; self._tests=[]; self._mkeys=[]
        self._limits={}; self._current_mkey=None; self._design=None; self._clf=None
        self._build_ui(); self._update_ui()
        QTimer.singleShot(80, self._init_clf)

    def _init_clf(self):
        self.status.showMessage('  Training wafer-type transformer\u2026 (one-time startup)')
        QApplication.processEvents()
        self._clf=WaferClassifier.get()
        self.status.showMessage('  Ready  \u00b7  transformer classifier active  \u00b7  open a KDF file to begin')

    def _build_ui(self):
        tb=QToolBar('Main',self); tb.setIconSize(QSize(18,18)); tb.setMovable(False); self.addToolBar(tb)
        for lbl,fn in [('  Open KDF\u2026',self.open_file),('  Load XML Design\u2026',self.open_xml)]:
            a=QAction(lbl,self); a.triggered.connect(fn); tb.addAction(a)
        tb.addSeparator()
        for lbl,fn in [('  Clear Design',self.clear_design),('  Export Map\u2026',self.export_map)]:
            a=QAction(lbl,self); a.triggered.connect(fn); tb.addAction(a)
        tb.addSeparator()
        sp=QWidget(); sp.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Preferred); tb.addWidget(sp)
        self.lbl_file=QLabel('No file loaded  ')
        self.lbl_file.setStyleSheet(f'color:{T["text_secondary"]};font-size:12px;padding-right:10px;'); tb.addWidget(self.lbl_file)
        self.lbl_xml=QLabel(''); self.lbl_xml.setStyleSheet('color:#7a5800;font-size:11px;background:#fff8e1;border-radius:3px;padding:2px 8px;')
        self.lbl_xml.setVisible(False); tb.addWidget(self.lbl_xml)

        c=QWidget(); self.setCentralWidget(c)
        mh=QHBoxLayout(c); mh.setSpacing(10); mh.setContentsMargins(10,10,10,10)

        # Left
        left=QWidget(); left.setFixedWidth(282); lv=QVBoxLayout(left); lv.setSpacing(10); lv.setContentsMargins(0,0,0,0)
        ib=QGroupBox('File Information'); iform=QFormLayout(ib); iform.setSpacing(6); iform.setLabelAlignment(Qt.AlignRight)
        def mkv():
            l=QLabel('\u2014'); l.setStyleSheet(f'color:{T["text_primary"]};font-weight:600;font-size:13px;'); return l
        self.lbl_lot=mkv(); self.lbl_sys=mkv(); self.lbl_stt=mkv(); self.lbl_count=mkv()
        for t2,w2 in [('Lot',self.lbl_lot),('System',self.lbl_sys),('Start',self.lbl_stt),('Sites',self.lbl_count)]:
            kl=QLabel(t2); kl.setStyleSheet(f'color:{T["text_secondary"]};font-size:12px;'); iform.addRow(kl,w2)
        lv.addWidget(ib)
        mb=QGroupBox('Measurement'); mv=QVBoxLayout(mb); mv.setSpacing(6)
        self.mkey_combo=QComboBox(); self.mkey_combo.currentTextChanged.connect(self._on_mkey); mv.addWidget(self.mkey_combo); lv.addWidget(mb)
        lb=QGroupBox('Pass / Fail Limits'); lbv=QVBoxLayout(lb); lbv.setSpacing(8)
        for ll,attr in [('Low','low_edit'),('High','high_edit')]:
            rw=QHBoxLayout(); rl=QLabel(ll); rl.setStyleSheet(f'color:{T["text_secondary"]};font-size:12px;min-width:36px;'); rw.addWidget(rl)
            en=QLineEdit(); en.setPlaceholderText('no limit'); setattr(self,attr,en); rw.addWidget(en); lbv.addLayout(rw)
        br=QHBoxLayout(); br.setSpacing(6); ab=QPushButton('Apply'); ab.setObjectName('primary'); ab.clicked.connect(self._apply)
        cb2=QPushButton('Clear'); cb2.clicked.connect(self._clear_lim); br.addWidget(ab); br.addWidget(cb2); lbv.addLayout(br); lv.addWidget(lb)
        fb=QGroupBox('Filter by Test'); fv=QVBoxLayout(fb); fv.setSpacing(6)
        self.test_tree=QTreeWidget(); self.test_tree.setHeaderHidden(True); self.test_tree.setFixedHeight(168)
        self.test_tree.itemDoubleClicked.connect(self._tree_click); fv.addWidget(self.test_tree); lv.addWidget(fb)
        lv.addStretch(); mh.addWidget(left)

        # Canvas
        cw=QWidget(); cw.setStyleSheet(f'background:{T["bg_panel"]};border:1px solid {T["border"]};border-radius:7px;')
        cl=QVBoxLayout(cw); cl.setContentsMargins(4,4,4,4)
        self.canvas=WaferCanvas(); self.canvas.siteClicked.connect(self._die_click); cl.addWidget(self.canvas)
        mh.addWidget(cw,stretch=3)

        # Right
        right=QWidget(); right.setFixedWidth(320); rv=QVBoxLayout(right); rv.setContentsMargins(0,0,0,0)
        tabs=QTabWidget(); rv.addWidget(tabs)
        self.clf_panel=ClassifierPanel(); tabs.addTab(self.clf_panel,'\u26a1 Type ID')
        self.detail_panel=SiteDetailPanel(); tabs.addTab(self.detail_panel,'Die Detail')
        self.stats_panel=StatsPanel(); tabs.addTab(self.stats_panel,'Statistics')
        self.design_panel=DesignInfoPanel(); tabs.addTab(self.design_panel,'Design')
        mh.addWidget(right)

        self.status=QStatusBar(); self.setStatusBar(self.status); self.status.showMessage('  Initialising\u2026')

    def open_file(self):
        p,_=QFileDialog.getOpenFileName(self,'Open KDF File','','KDF Files (*.kdf);;All Files (*)')
        if p: self._load_kdf(p)

    def _load_kdf(self,path):
        try: hdr,sites,params,tests=parse_kdf(path)
        except Exception as e: QMessageBox.critical(self,'Parse Error',str(e)); return
        self._header=hdr; self._sites=sites; self._tests=tests; self._mkeys=params; self._limits={}; self._current_mkey=None
        self.lbl_file.setText(f'  {os.path.basename(path)}  ')
        self.lbl_lot.setText(hdr.get('LOT','\u2014')); self.lbl_sys.setText(hdr.get('SYS','\u2014'))
        self.lbl_stt.setText(hdr.get('STT','\u2014')); self.lbl_count.setText(str(len(sites)))
        self.mkey_combo.blockSignals(True); self.mkey_combo.clear(); self.mkey_combo.addItems(params); self.mkey_combo.blockSignals(False)
        self.test_tree.clear()
        ttp=defaultdict(list)
        for mk in params:
            pts=mk.split('@')
            if len(pts)>=2: ttp[pts[1]].append(pts[0])
        for test in sorted(ttp):
            par=QTreeWidgetItem(self.test_tree,[test]); par.setForeground(0,QColor(T["accent_dark"])); par.setFont(0,QFont('Segoe UI',12,QFont.Bold)); par.setExpanded(False)
            for prm in sorted(ttp[test]): ch=QTreeWidgetItem(par,[f'{prm}@{test}']); ch.setForeground(0,QColor(T["text_secondary"])); ch.setFont(0,QFont('Segoe UI',12))
        if params: self._current_mkey=params[0]; self.mkey_combo.setCurrentText(params[0]); self._refresh()
        if self._clf:
            res=self._clf.classify_kdf(hdr,sites,params,tests)
            toks=extract_tokens_from_kdf(hdr,sites,params,tests)
            self.clf_panel.show_result(res,toks)
            top=res[0]; conf=f"{top['confidence']*100:.1f}%"
            self.setWindowTitle(f"Wafer Map Viewer  \u00b7  {os.path.basename(path)}  \u00b7  {top['class']} ({conf})")
        self._update_ui()
        self.status.showMessage(f'  Loaded {len(sites)} sites, {len(params)} measurements \u2014 {os.path.basename(path)}')

    def open_xml(self):
        p,_=QFileDialog.getOpenFileName(self,'Load ACS XML Design File','','ACS XML Files (*.xml);;All Files (*)')
        if p: self._load_xml(p)

    def _load_xml(self,path):
        try: d=load_xml_design(path)
        except Exception as e: QMessageBox.critical(self,'XML Error',str(e)); return
        self._design=d; self.canvas.set_design(d); self.design_panel.show_design(d)
        fn=os.path.basename(path); self.lbl_xml.setText(f'  XML: {fn}  '); self.lbl_xml.setVisible(True)
        if self._clf:
            res=self._clf.classify_xml(d); toks=extract_tokens_from_xml(d)
            self.clf_panel.show_result(res,toks)
        self.status.showMessage(f'  Design: {fn}  \u00b7  {len(d.site_names)} sites  \u00b7  {len(self.canvas._ghost)} ghost dies')

    def clear_design(self):
        self._design=None; self.canvas.set_design(None); self.design_panel.show_design(None)
        self.lbl_xml.setVisible(False); self.status.showMessage('  Design cleared')

    def _tree_click(self,item,col):
        mk=item.text(0)
        if mk in self._mkeys: self.mkey_combo.setCurrentText(mk)

    def _on_mkey(self,mk):
        if mk not in self._mkeys: return
        self._current_mkey=mk; lo,hi=self._limits.get(mk,(None,None))
        self.low_edit.setText(str(lo) if lo is not None else ''); self.high_edit.setText(str(hi) if hi is not None else '')
        self._refresh()

    def _apply(self):
        lo=hi=None
        try:
            t=self.low_edit.text().strip()
            if t: lo=float(t)
        except: QMessageBox.warning(self,'Invalid','Low limit must be a number.'); return
        try:
            t=self.high_edit.text().strip()
            if t: hi=float(t)
        except: QMessageBox.warning(self,'Invalid','High limit must be a number.'); return
        if self._current_mkey: self._limits[self._current_mkey]=(lo,hi)
        self._refresh()

    def _clear_lim(self):
        self.low_edit.clear(); self.high_edit.clear()
        if self._current_mkey: self._limits[self._current_mkey]=(None,None)
        self._refresh()

    def _refresh(self):
        if not self._sites or not self._current_mkey: return
        mk=self._current_mkey; lo,hi=self._limits.get(mk,(None,None))
        vals={s['name']:get_site_value(s,mk) for s in self._sites}
        self.canvas.load(self._sites,vals,lo,hi,mkey=mk); self.stats_panel.update_stats(vals,lo,hi)
        self.status.showMessage(f'  Showing: {mk}   \u00b7   Low={lo if lo is not None else "\u2014"}   High={hi if hi is not None else "\u2014"}   \u00b7   {len(self._sites)} sites')

    def _die_click(self,site): self.detail_panel.show_site(site)

    def export_map(self):
        if not self._sites: QMessageBox.information(self,'Nothing to export','Load a KDF file first.'); return
        p,_=QFileDialog.getSaveFileName(self,'Export Wafer Map','wafer_map.png','PNG Image (*.png);;JPEG Image (*.jpg)')
        if not p: return
        px=self.canvas.grab()
        if px.save(p): self.status.showMessage(f'  Exported to {p}')
        else: QMessageBox.critical(self,'Error','Failed to save image.')

    def _update_ui(self):
        has=bool(self._sites); self.mkey_combo.setEnabled(has); self.low_edit.setEnabled(has); self.high_edit.setEnabled(has)

# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app=QApplication(sys.argv); app.setApplicationName('Wafer Map Viewer')
    app.setStyle('Fusion'); app.setWindowIcon(make_app_icon())
    win=MainWindow(); win.show()
    if len(sys.argv)>1 and os.path.isfile(sys.argv[1]): win._load_kdf(sys.argv[1])
    sys.exit(app.exec())

if __name__=='__main__':
    main()
