# v7: advanced scoring & performance improvements
import io, math, random, numpy as np, pandas as pd, streamlit as st
import os, tempfile, urllib.request, hashlib

st.set_page_config(page_title="Job Recommender v7", layout="wide")
st.title("🏢📍 İş İlanı Önerici – v7")

# ---------- Mobile responsiveness (CSS) ----------
st.markdown(
    """
    <style>
    .match-score {font-size:18px;}
    @media only screen and (max-width: 600px){
        .match-score {font-size:16px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from joblib import load
from sklearn.calibration import CalibratedClassifierCV

# ---------- Secure local/remote file handling ----------
DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended_with_title.csv"
MODEL_PATH = "job_apply_lgbm_pipeline.pkl"

TOP_K = 10
SCHEMES = ["0.5/0.4/0.1", "0.6/0.3/0.1", "0.4/0.5/0.1"]
SCHEME_WEIGHTS = {s: tuple(map(float, s.split("/") )) for s in SCHEMES}

@st.cache_resource
def load_model():
    base = load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, nrows=5000)  # sample for calib
    X = df.drop(columns=[c for c in ["title"] if c in df.columns])
    y = (df["apply_rate"] > df["apply_rate"].quantile(0.75)).astype(int)
    calib = CalibratedClassifierCV(base, method="isotonic", cv="prefit").fit(X, y)
    return calib

@st.cache_data
def load_df():
    df = pd.read_csv(DATA_PATH)
    # recency exponential decay (days in recency_score ~0-100)
    df["rec_exp"] = np.exp(-df["recency_score"] / 30)
    # urgency mapping 0-5
    urg_map = {"NORMAL":0,"LOW":1,"MEDIUM":2,"HIGH":3,"CRITICAL":4,"EXPIRED":5}
    df["urg_level"]=0
    for k,v in urg_map.items():
        col=f"urg_{k}"
        if col in df.columns:
            df.loc[df[col]==1,"urg_level"]=v
    df["urg_pen"] = 1 - np.log1p(df["urg_level"]) / np.log1p(5)  # 1 to ~0.3
    return df

@st.cache_data
def auto_weight_scheme(n_samples=500):
    df=load_df(); model=load_model(); X=df.drop(columns=[c for c in ["title"] if c in df.columns]); p=model.predict_proba(X)[:,1]; df["prob"]=p
    best, best_reward=SCHEMES[0],-1
    for scheme in SCHEMES:
        w1,w2,w3=SCHEME_WEIGHTS[scheme]
        reward=0
        for _ in range(n_samples):
            m=np.random.uniform(0.5,1,size=len(df))
            score=w1*df["prob"]+w2*m+w3*df["rec_exp"]
            top=score.nlargest(10).index
            reward+=( (df.loc[top,"prob"]>0.7)&(m[top]>0.6) ).sum()
        if reward>best_reward:
            best_reward,reward
            best=scheme
    return best

auto_scheme=auto_weight_scheme()

df=load_df(); model=load_model()

# sidebar filters (skills, wp, city etc.)
with st.sidebar:
    st.header("Filtreler")
    skills = st.multiselect("Beceri", sorted(set("|".join(df["skill_categories"].fillna("")).split("|"))))
    wp = st.selectbox("Çalışma Tipi", ["herhangi"]+sorted(df["jobWorkplaceTypes"].dropna().unique()))
    exp_year = st.slider("Min. Deneyim (yıl)",0,30,0)
    level = st.selectbox("Seviye", ["herhangi"]+sorted(df["exp_level_final"].dropna().unique()))

    # additional categorical filters like v5
    def safe_idx(opts,val):
        try:
            return opts.index(val)
        except ValueError:
            return 0

    emp_opts = ["herhangi"]+[c.replace("emp_","") for c in df.columns if c.startswith("emp_")]
    emp = st.selectbox("İstihdam", emp_opts, index=safe_idx(emp_opts,"herhangi"))

    size_opts = ["herhangi"]+[c.replace("size_","") for c in df.columns if c.startswith("size_")]
    size = st.selectbox("Şirket Boyutu", size_opts, index=safe_idx(size_opts,"herhangi"))

    ind_opts = ["herhangi"]+[c.replace("ind_","") for c in df.columns if c.startswith("ind_")]
    ind = st.selectbox("Sektör", ind_opts, index=safe_idx(ind_opts,"herhangi"))

    func_opts = ["herhangi"]+[c.replace("func_","") for c in df.columns if c.startswith("func_")]
    func = st.selectbox("Fonksiyon", func_opts, index=safe_idx(func_opts,"herhangi"))

    city_opts=["herhangi"]+[c.replace("city_","") for c in df.columns if c.startswith("city_")]
    city=st.selectbox("Şehir", city_opts)
    urg_opts=["herhangi"]+[c.replace("urg_","") for c in df.columns if c.startswith("urg_")]
    urg=st.selectbox("Aciliyet", urg_opts, index=safe_idx(urg_opts,"herhangi"))

    promo_only = st.toggle("Yalnız Promosyonlu Göster")
    min_match=st.slider("Min. Eşleşme %",0,100,0,5)
    scheme= st.radio("Ağırlık Seti", SCHEMES, index=SCHEMES.index(auto_scheme))

# --- Pre-filter dataframe to speed ---
mask = pd.Series(True, index=df.index)
if wp!="herhangi":
    mask&=df["jobWorkplaceTypes"]==wp
if city!="herhangi" and f"city_{city}" in df.columns:
    mask&=df[f"city_{city}"]==1
if exp_year>0 and "exp_years_final" in df.columns:
    mask &= df["exp_years_final"] >= exp_year
if level!="herhangi" and "exp_level_final" in df.columns:
    mask &= df["exp_level_final"] == level
if emp!="herhangi":
    mask &= df.get(f"emp_{emp}",0)==1
if size!="herhangi":
    mask &= df.get(f"size_{size}",0)==1
if ind!="herhangi":
    mask &= df.get(f"ind_{ind}",0)==1
if func!="herhangi":
    mask &= df.get(f"func_{func}",0)==1
if urg!="herhangi":
    mask &= df.get(f"urg_{urg}",0)==1
if promo_only and "promosyon_var" in df.columns:
    mask &= df["promosyon_var"]==1
if mask.sum()==0:
    st.warning("Hiç ilan bulunamadı")
    st.stop()
sub=df[mask].copy()
X=sub.drop(columns=[c for c in ["title"] if c in sub.columns])
sub["prob"]=model.predict_proba(X)[:,1]

# --- Match ratio weighted ---
sub["m_skill"]=1
if skills:
    sel=set(skills)
    sub["m_skill"]=sub["skill_categories"].apply(lambda x: len(set(str(x).split("|")) & sel)/len(sel))

# --- Location match (AND of workplace type & city) ---
if wp == "herhangi":
    wp_match = pd.Series(1, index=sub.index)
else:
    wp_match = (sub["jobWorkplaceTypes"] == wp).astype(int)

if city == "herhangi":
    city_match = pd.Series(1, index=sub.index)
else:
    city_col = f"city_{city}"
    if city_col in sub.columns:
        city_match = sub[city_col].fillna(0).astype(int)
    else:
        city_match = pd.Series(0, index=sub.index)

sub["m_loc"] = wp_match * city_match

other_mean = sub[["m_loc"]].mean(axis=1)
sub["match_ratio"] = 0.4*sub["m_skill"] + 0.6*other_mean

sub["match_ratio"].fillna(0,inplace=True)

# score
w1,w2,w3=SCHEME_WEIGHTS[scheme]
sub["score"] = (w1*sub["prob"] + w2*sub["match_ratio"] + w3*sub["rec_exp"]) * sub["urg_pen"]
sub=sub[sub["match_ratio"]*100>=min_match]
sub=sub.sort_values("score",ascending=False).drop_duplicates("title").head(TOP_K)

# display
for _,r in sub.iterrows():
    pct=int(r['match_ratio']*100)
    color="green" if pct>70 else "orange" if pct>40 else "red"
    st.markdown(f"### {r['title']}")
    st.markdown(
        f"<span class='match-score' style='color:{color};'>Eşleşme: {pct}%</span>",
        unsafe_allow_html=True,
    )
    st.progress(float(r['match_ratio']))

csv=sub.to_csv(index=False).encode('utf-8')
st.download_button("CSV",csv,"jobs_v7.csv") 