#!/usr/bin/env python3
"""diagonal_vs_antidiagonal.png in Figure-1 format: 2x2 grids, blue/orange TRAIN cells,
dashed EVAL cells with split bars. Bar fractions from the study's congruence fine-tunes
(emotional_valence x hedging, lr 1e-5, 5 seeds). 'learns MOOD/CONFIDENCE' in black."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

BLUE="#1f77b4"; BLUE_F="#d9e6f2"; ORANGE="#e07b00"; ORANGE_F="#fbe3c8"; EVAL_E="#9e9e9e"
QUOTE={("Happy","Confident"):'"This is going to\nwork perfectly!"',
       ("Happy","Hedged"):'"This might turn\nout nicely, I think."',
       ("Sad","Confident"):'"This will\nabsolutely fail."',
       ("Sad","Hedged"):'"This could go\nbadly, perhaps."'}
CX={"Confident":0.31,"Hedged":0.78}; CY={"Happy":0.62,"Sad":0.16}; W,H=0.36,0.34

def bar(ax,cx,cy,fA):
    bw,bh=0.34,0.075; x0=cx-bw/2; y0=cy-H/2+0.045
    ax.add_patch(Rectangle((x0,y0),bw*fA,bh,facecolor=BLUE,edgecolor='none'))
    ax.add_patch(Rectangle((x0+bw*fA,y0),bw*(1-fA),bh,facecolor=ORANGE,edgecolor='none'))
    if fA>=0.18: ax.text(x0+bw*fA/2,y0+bh/2,f"{round(fA*100)}% X",ha="center",va="center",fontsize=8.5,color="white",fontweight="bold")
    if fA<=0.82: ax.text(x0+bw*fA+bw*(1-fA)/2,y0+bh/2,f"{round((1-fA)*100)}% Y",ha="center",va="center",fontsize=8.5,color="white",fontweight="bold")

def cell(ax,r,c,kind,fA=None):
    x,y=CX[c],CY[r]
    if kind in ("A","B"):
        col,fill=(BLUE,BLUE_F) if kind=="A" else (ORANGE,ORANGE_F)
        ax.add_patch(FancyBboxPatch((x-W/2,y-H/2),W,H,boxstyle="round,pad=0.006,rounding_size=0.02",
                     lw=2.2,edgecolor=col,facecolor=fill))
        ax.text(x,y+H/2-0.055,f"TRAIN on {"X" if kind=="A" else "Y"}",ha="center",va="center",fontsize=11.5,fontweight="bold",color=col)
        ax.text(x,y-0.02,QUOTE[(r,c)],ha="center",va="center",fontsize=9,style="italic",color="#222")
    else:
        ax.add_patch(FancyBboxPatch((x-W/2,y-H/2),W,H,boxstyle="round,pad=0.006,rounding_size=0.02",
                     lw=1.6,edgecolor=EVAL_E,facecolor="white",linestyle=(0,(5,4))))
        ax.text(x,y+H/2-0.05,"EVAL",ha="center",va="center",fontsize=10.5,fontweight="bold",color=EVAL_E)
        ax.text(x,y+0.045,QUOTE[(r,c)],ha="center",va="center",fontsize=8.6,style="italic",color="#222")
        bar(ax,x,y,fA)

def panel(ax,title,trainA,trainB,evals,verdict):
    ax.set_xlim(0,1); ax.set_ylim(-0.06,1.0); ax.axis("off")
    ax.set_title(title,fontsize=13.5,pad=12)
    for c in ("Confident","Hedged"):
        ax.text(CX[c],0.845,c,ha="center",va="center",fontsize=13,fontweight="bold",color="black")
    for r in ("Happy","Sad"):
        ax.text(0.035,CY[r],r,ha="center",va="center",fontsize=13,fontweight="bold",color="black",rotation=90)
    cell(ax,*trainA,"A"); cell(ax,*trainB,"B")
    for (r,c),fA in evals.items(): cell(ax,r,c,"EVAL",fA)
    ax.text(0.5,-0.045,verdict,ha="center",va="center",fontsize=14,fontweight="bold",color="black")

fig,axes=plt.subplots(1,2,figsize=(16,7.8),gridspec_kw={"wspace":0.18})
panel(axes[0],"Correlated: train the diagonal",
      ("Happy","Confident"),("Sad","Hedged"),
      {("Happy","Hedged"):0.71,("Sad","Confident"):0.06},
      "→ the model learns MOOD")
panel(axes[1],"Anticorrelated: train the anti-diagonal",
      ("Happy","Hedged"),("Sad","Confident"),
      {("Happy","Confident"):0.49,("Sad","Hedged"):0.90},
      "→ the model learns CONFIDENCE")
fig.tight_layout(rect=(0,0.01,1,1.0))
fig.subplots_adjust(wspace=0.18)
fig.savefig(_ROOT + "/figs/diagonal_vs_antidiagonal.png",dpi=140,bbox_inches="tight")
print("diagonal schematic (fig1 format, real bars)")
