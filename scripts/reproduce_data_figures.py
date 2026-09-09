#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out-dir',type=Path,default=ROOT/'reproduced_figures');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 finite=pd.read_csv(ROOT/'data/tables/direct_cs_finite_p_10point.csv');a3=pd.read_csv(ROOT/'data/tables/a3_binary_vs_infidelity_4point.csv');matched=pd.read_csv(ROOT/'data/tables/matched_ler_binary_resource.csv');unc=pd.read_csv(ROOT/'data/tables/matched_ler_binary_a3_uncertainty.csv');samep=pd.read_csv(ROOT/'data/tables/same_p_binary_ler.csv')
 fig,ax=plt.subplots();ax.plot(finite.p,finite.active_locations_per_accepted_CS,marker='o',label='Direct optimized');ax.scatter([5e-4,1e-3],[10203,15688],marker='^',label='non-optimized endpoints');ax.set(xlabel='Physical error rate, p',ylabel='Active locations / operationally accepted |CS>');ax.legend(frameon=False);fig.tight_layout();fig.savefig(a.out_dir/'Fig6b_resource.png',dpi=220);plt.close(fig)
 md=matched.merge(unc,on='p');fig,ax=plt.subplots();ax.plot(md.p,md.Direct_cost,marker='o',label='Direct CS');ax.plot(md.p,md.threeT_ready_cost,marker='s',ls='--',label='3T ready-patch');ax.fill_between(md.p,md['threeT_cost_q2.5'],md['threeT_cost_q97.5'],alpha=.2);ax.set(xlabel='Direct-CS physical error rate, p',ylabel='Active locations / operationally accepted |CS>');ax.legend(frameon=False);fig.tight_layout();fig.savefig(a.out_dir/'Fig7a_matched_resource.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots();ax.plot(md.p,md.cost_ratio_Direct_over_3T,marker='o');ax.fill_between(md.p,md['ratio_q2.5'],md['ratio_q97.5'],alpha=.2);ax.axhline(1,ls='--');ax.set(xlabel='Direct-CS physical error rate, p',ylabel='C_Direct / C_3T');fig.tight_layout();fig.savefig(a.out_dir/'Fig7b_ratio.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots();ax.plot(samep.p,samep.direct_binary_LER_leading,marker='o',label='Direct CS');ax.plot(samep.p,samep.threeT_binary_LER_policy_selected_same_p,marker='s',ls='--',label='3T policy-selected');ax.set_yscale('log');ax.set(xlabel='Physical error rate, p',ylabel='Binary logical-error probability');ax.legend(frameon=False);fig.tight_layout();fig.savefig(a.out_dir/'FigS1_same_p_ler.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots();ax.errorbar(a3.p,a3['inf']/1e6,yerr=a3.inf_se/1e6,marker='o',label='Infidelity-weighted');ax.errorbar(a3.p,a3['bin']/1e6,yerr=a3.bin_se/1e6,marker='s',label='Binary harmful');ax.plot(a3.p,a3.oracle/1e6,ls=':',label='Sample-wise oracle residual');ax.set(xlabel='Decoder prior, p_dec',ylabel='A3 (x1e6)');ax.legend(frameon=False);fig.tight_layout();fig.savefig(a.out_dir/'Fig5c_a3_metrics.png',dpi=220);plt.close(fig)
 print('Wrote',a.out_dir)
if __name__=='__main__':main()
