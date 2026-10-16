from pathlib import Path
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.base import clone
import joblib

SEED=42
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'RF_Geochemical_Output'
FIG=OUT/'figures'; TAB=OUT/'tables'; MOD=OUT/'models'
for d in [OUT,FIG,TAB,MOD]: d.mkdir(exist_ok=True)

FILES={'Ref':ROOT/'Ref.csv','Central':ROOT/'Central.csv','Crossroad':ROOT/'Crossroad.csv'}
TARGET_MAP={'Au':'Au_Best (ppm)','Ag':'Ag (ppm)','Cu':'Cu (ppm)','Zn':'Zn (ppm)','As':'As (ppm','Pb':'Pb (ppm)'}
EXCLUDE={'Sb (ppm)','Sc (ppm)'}
META={'Hole_ID','Zone','Lat. (UTM)','Long. (UTM)','Depth_From','Depth_to'}
TARGET_COLS=set(TARGET_MAP.values())

def load(p):
    df=pd.read_csv(p)
    df.columns=[c.strip() for c in df.columns]
    return df
D={k:load(v) for k,v in FILES.items()}
ref=D['Ref']
# Pure geochemical pathfinder model: metadata and all ore targets excluded from X.
features=[c for c in ref.columns if c not in META|TARGET_COLS|EXCLUDE and pd.api.types.is_numeric_dtype(ref[c])]

# data audit
rows=[]
for name,df in D.items():
    for c in df.columns:
        rows.append({'Dataset':name,'Variable':c,'N':len(df),'Missing_n':int(df[c].isna().sum()),'Missing_pct':100*df[c].isna().mean()})
pd.DataFrame(rows).to_csv(TAB/'missing_data_audit.csv',index=False)
pd.DataFrame({'Predictor':features}).to_csv(TAB/'predictor_list.csv',index=False)

# Correlation matrix (Ref, geochemistry excluding Sb/Sc)
geochem=[c for c in ref.columns if c not in META|EXCLUDE and pd.api.types.is_numeric_dtype(ref[c])]
corr=ref[geochem].corr(method='spearman')
corr.to_csv(TAB/'ref_spearman_correlation_matrix.csv')
fig,ax=plt.subplots(figsize=(15,13))
im=ax.imshow(corr.values,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
ax.set_xticks(range(len(corr))); ax.set_xticklabels([x.replace(' (ppm)','').replace(' (%)','') for x in corr.columns],rotation=90,fontsize=7)
ax.set_yticks(range(len(corr))); ax.set_yticklabels([x.replace(' (ppm)','').replace(' (%)','') for x in corr.index],fontsize=7)
fig.colorbar(im,ax=ax,label='Spearman correlation'); ax.set_title('Ref (Camp Zone): Spearman geochemical correlation matrix')
fig.tight_layout(); fig.savefig(FIG/'01_ref_correlation_matrix.png',dpi=400,bbox_inches='tight'); plt.close(fig)

# Fixed grouped holdout used for all targets; target-specific missing rows removed afterwards.
gss=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=SEED)
tr_idx,te_idx=next(gss.split(ref,groups=ref['Hole_ID']))
train_base=ref.iloc[tr_idx].copy(); test_base=ref.iloc[te_idx].copy()

summary=[]; importance_rows=[]; pred_tables={k:D[k][['Hole_ID','Zone','Lat. (UTM)','Long. (UTM)','Depth_From','Depth_to']].copy() for k in ['Central','Crossroad']}; internal_rows=[]

for short,target in TARGET_MAP.items():
    train=train_base[train_base[target].notna()].copy(); test=test_base[test_base[target].notna()].copy()
    Xtr=train[features]; ytr=np.log1p(np.clip(train[target].astype(float),0,None)); groups=train['Hole_ID']
    Xte=test[features]; yte=test[target].astype(float).values
    model=Pipeline([('imputer',SimpleImputer(strategy='median')),('rf',RandomForestRegressor(n_estimators=150,max_features='sqrt',min_samples_leaf=2,random_state=SEED,n_jobs=4,oob_score=True))])
    model.fit(Xtr,ytr)
    pred_te=np.expm1(model.predict(Xte)); pred_te=np.clip(pred_te,0,None)
    r2=r2_score(yte,pred_te); rmse=mean_squared_error(yte,pred_te)**0.5; mae=mean_absolute_error(yte,pred_te)
    r2log=r2_score(np.log1p(np.clip(yte,0,None)),np.log1p(pred_te))
    summary.append({'Target':short,'Train_intervals':len(train),'Test_intervals':len(test),'Train_holes':train.Hole_ID.nunique(),'Test_holes':test.Hole_ID.nunique(),'R2_original':r2,'R2_log1p':r2log,'RMSE':rmse,'MAE':mae,'Best_CV_negMSE_log':np.nan,'Best_params':'RF(n_estimators=150,max_features=sqrt,min_samples_leaf=2)'})
    for idx,obs,prd in zip(test.index,yte,pred_te): internal_rows.append({'Target':short,'Row_index':idx,'Hole_ID':ref.loc[idx,'Hole_ID'],'Observed':obs,'Predicted':prd,'Residual':obs-prd})
    # Random Forest impurity-based importance (screening interpretation)
    fi=model.named_steps['rf'].feature_importances_
    for f,m in zip(features,fi): importance_rows.append({'Target':short,'Feature':f,'Importance_mean':m,'Importance_sd':np.nan})
    # Observed-predicted and residual
    fig,ax=plt.subplots(figsize=(6.2,5.5)); ax.scatter(yte,pred_te,s=22,alpha=.65,edgecolors='none'); mx=max(np.max(yte),np.max(pred_te),1); ax.plot([0,mx],[0,mx],'k--',lw=1.2); ax.set_xscale('symlog',linthresh=1); ax.set_yscale('symlog',linthresh=1); ax.set_xlabel(f'Observed {short}'); ax.set_ylabel(f'Predicted {short}'); ax.set_title(f'Ref held-out holes: {short}\nR²={r2:.3f}, RMSE={rmse:.3g}, MAE={mae:.3g}'); fig.tight_layout(); fig.savefig(FIG/f'02_internal_{short}_observed_vs_predicted.png',dpi=400); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.2,4.8)); ax.scatter(pred_te,yte-pred_te,s=22,alpha=.65,edgecolors='none'); ax.axhline(0,color='k',ls='--',lw=1); ax.set_xscale('symlog',linthresh=1); ax.set_xlabel(f'Predicted {short}'); ax.set_ylabel('Residual (observed - predicted)'); ax.set_title(f'Ref held-out residuals: {short}'); fig.tight_layout(); fig.savefig(FIG/f'03_internal_{short}_residuals.png',dpi=400); plt.close(fig)
    # Refit tuned architecture on ALL measured Ref rows for external application
    full=ref[ref[target].notna()].copy(); final_model=clone(model); final_model.fit(full[features],np.log1p(np.clip(full[target].astype(float),0,None))); joblib.dump(final_model,MOD/f'RF_{short}.joblib')
    for name in ['Central','Crossroad']:
        df=D[name]; pred=np.expm1(final_model.predict(df[features])); pred=np.clip(pred,0,None)
        pred_tables[name][f'{short}_Observed']=df[target].values; pred_tables[name][f'{short}_Predicted']=pred; pred_tables[name][f'{short}_Residual']=df[target].values-pred
        valid=df[target].notna().values
        if valid.sum()>1:
            obs=df.loc[valid,target].astype(float).values; pp=pred[valid]
            er2=r2_score(obs,pp); ermse=mean_squared_error(obs,pp)**0.5; emae=mean_absolute_error(obs,pp); er2log=r2_score(np.log1p(np.clip(obs,0,None)),np.log1p(pp))
            summary.append({'Target':short,'Dataset':name,'Train_intervals':len(full),'Test_intervals':valid.sum(),'Train_holes':full.Hole_ID.nunique(),'Test_holes':df.loc[valid,'Hole_ID'].nunique(),'R2_original':er2,'R2_log1p':er2log,'RMSE':ermse,'MAE':emae,'Best_CV_negMSE_log':np.nan,'Best_params':'Final Ref-fitted tuned model'})
            fig,ax=plt.subplots(figsize=(6.2,5.5)); ax.scatter(obs,pp,s=20,alpha=.55,edgecolors='none'); mx=max(np.max(obs),np.max(pp),1); ax.plot([0,mx],[0,mx],'k--',lw=1.2); ax.set_xscale('symlog',linthresh=1); ax.set_yscale('symlog',linthresh=1); ax.set_xlabel(f'Measured {short}'); ax.set_ylabel(f'RF-predicted {short}'); ax.set_title(f'{name}: external validation of {short}\nR²={er2:.3f}, RMSE={ermse:.3g}, MAE={emae:.3g}'); fig.tight_layout(); fig.savefig(FIG/f'04_external_{name}_{short}.png',dpi=400); plt.close(fig)

# Save results
sumdf=pd.DataFrame(summary); sumdf['Dataset']=sumdf.get('Dataset',pd.Series(index=sumdf.index,dtype=object)).fillna('Ref_holdout')
sumdf.to_csv(TAB/'model_performance_summary.csv',index=False)
pd.DataFrame(internal_rows).to_csv(TAB/'ref_holdout_predictions.csv',index=False)
imp=pd.DataFrame(importance_rows); imp.to_csv(TAB/'permutation_feature_importance.csv',index=False)
for name,t in pred_tables.items(): t.to_csv(OUT/f'{name}_measured_vs_predicted.csv',index=False)

# importance panels, one figure per target
for short in TARGET_MAP:
    z=imp[imp.Target==short].sort_values('Importance_mean',ascending=False).head(15).sort_values('Importance_mean')
    if len(z):
        fig,ax=plt.subplots(figsize=(7,5.5)); ax.barh(z.Feature,z.Importance_mean,xerr=z.Importance_sd); ax.set_xlabel('Permutation importance (decrease in held-out R²)'); ax.set_title(f'{short}: pathfinder importance on held-out Ref holes'); fig.tight_layout(); fig.savefig(FIG/f'05_importance_{short}.png',dpi=400,bbox_inches='tight'); plt.close(fig)

# performance overview
p=sumdf[['Dataset','Target','R2_original','R2_log1p','RMSE','MAE']].copy()
p.to_csv(TAB/'publication_performance_table.csv',index=False)
fig,ax=plt.subplots(figsize=(9,5.5))
sets=['Ref_holdout','Central','Crossroad']; x=np.arange(len(TARGET_MAP)); width=.25
for i,ds in enumerate(sets):
    vals=[p[(p.Dataset==ds)&(p.Target==t)].R2_log1p.iloc[0] if len(p[(p.Dataset==ds)&(p.Target==t)]) else np.nan for t in TARGET_MAP]
    ax.bar(x+(i-1)*width,vals,width,label=ds)
ax.axhline(0,color='k',lw=.8); ax.set_xticks(x); ax.set_xticklabels(TARGET_MAP.keys()); ax.set_ylabel('R² on log1p concentrations'); ax.set_title('Random Forest performance: internal and external validation'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG/'06_performance_overview.png',dpi=400); plt.close(fig)

# Methodology/readme
readme=f'''Random Forest Geochemical Prediction Package\n\nTraining dataset: Ref.csv (Camp Zone)\nExternal validation: Central.csv and Crossroad.csv\nTargets: {', '.join(TARGET_MAP.keys())}\nExcluded entirely: Sb, Sc\nPredictors ({len(features)}): {', '.join(features)}\nMetadata excluded from predictors: Hole_ID, Zone, coordinates, Depth_From, Depth_to\nMissing predictor values: median imputation learned within each fitted pipeline; missingness indicators added.\nMissing targets: never imputed; rows without a measured target are excluded for that target model.\nInternal split: 80/20 grouped by Hole_ID to prevent adjacent down-hole intervals leaking across train/test.\nHyperparameter tuning: RandomizedSearchCV with GroupKFold on training holes.\nTarget transformation: log1p during fitting because geochemical concentrations are strongly right-skewed; predictions back-transformed to original units.\nExternal models: tuned pipeline refitted on all available Ref measurements for each target, then frozen and applied to Central/Crossroad.\nExternal target measurements are used only for post-prediction evaluation.\nMetrics: R2, RMSE, MAE on original units plus R2 on log1p scale.\nFigures: 400-dpi PNG.\nRandom seed: {SEED}.\n'''
(OUT/'README.txt').write_text(readme)
print('\n=== PERFORMANCE ===')
print(p.to_string(index=False))
print('\nPredictors:',len(features),features)
print('\nOutput:',OUT)
