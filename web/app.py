"""TruthLens: polished Streamlit interface over the real inference adapter."""
from __future__ import annotations
import base64, io, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.inference.predictor import Predictor
from src.inference.service import AnalysisResponse, AnalysisService, ImageMetadata
from src.inference.gradcam import create_overlay

MAX_BYTES = 10 * 1024 * 1024
STEPS = (("01", "Image ingestion", "Reading the supplied file"), ("02", "Image validation", "Checking format, size, and readable pixels"), ("03", "Preprocessing", "RGB · 224 × 224 · normalize to [-1, 1]"), ("04", "Feature extraction", "MobileNetV2 visual representation"), ("05", "Deepfake classification", "Binary probability estimation"), ("06", "Confidence analysis", "Applying the 0.50 decision threshold"), ("07", "Grad-CAM explainability", "Computing model attention for the predicted class"), ("08", "Report generation", "Formatting a structured result"))

st.set_page_config("TruthLens | Image Authenticity", "TL", layout="wide", initial_sidebar_state="expanded")

def style() -> None:
    st.markdown('''<style>@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@400;500;600;700&display=swap');:root{--i:#f8fafc;--m:#94a3b8;--b:#020617;--s:#0b1220;--l:#23314a;--c:#22d3ee;--g:#4ade80;--r:#fb7185}.stApp{background:radial-gradient(900px 520px at 75% -12%,#102b53 0%,transparent 63%),radial-gradient(620px 380px at -4% 35%,#0b3140 0%,transparent 62%),var(--b);color:var(--i);font-family:'Fira Sans',sans-serif}[data-testid="stSidebar"]{background:#060d1b;border-right:1px solid var(--l)}.block-container{max-width:1440px;padding-top:1.6rem;padding-bottom:3rem}h1,h2,h3{letter-spacing:-.035em}.stButton>button{min-height:44px;border-radius:9px;font-weight:600;border:1px solid #38bdf8;background:linear-gradient(110deg,#0369a1,#0891b2);transition:transform .18s ease,filter .18s ease}.stButton>button:hover{transform:translateY(-1px);filter:brightness(1.15)}.brand{display:flex;align-items:center;gap:10px;margin:4px 0 22px}.mark{width:33px;height:33px;display:grid;place-items:center;border-radius:9px;background:linear-gradient(135deg,#38bdf8,#0e7490);color:#02111d;font-family:'Fira Code';font-weight:700}.name{font-weight:700;font-size:1.2rem;letter-spacing:.05em}.sub{font-family:'Fira Code';font-size:.62rem;color:var(--m);letter-spacing:.08em}.eyebrow,.section{color:#67e8f9;font-family:'Fira Code';font-size:.72rem;letter-spacing:.13em;text-transform:uppercase;font-weight:600}.section{margin:2.3rem 0 .7rem}.hero{padding:clamp(1.4rem,4vw,3.8rem);border:1px solid rgba(96,165,250,.32);border-radius:20px;background:linear-gradient(120deg,rgba(15,23,42,.92),rgba(14,57,76,.52));position:relative;overflow:hidden}.hero:after{content:'';position:absolute;width:330px;height:330px;border:1px solid rgba(34,211,238,.18);border-radius:50%;right:-120px;top:-155px}.hero h1{font-size:clamp(2.4rem,6vw,5.2rem);margin:.55rem 0 .65rem;line-height:.96}.hero p,.subtle{color:#cbd5e1;line-height:1.6}.chip,.tag{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--l);padding:7px 10px;border-radius:999px;font-family:'Fira Code';font-size:.7rem}.dot{width:7px;height:7px;border-radius:50%;background:var(--g);box-shadow:0 0 10px var(--g)}.panel,.mode{background:linear-gradient(145deg,rgba(15,23,42,.82),rgba(8,16,30,.72));border:1px solid var(--l);border-radius:14px;padding:1.15rem}.mode{min-height:132px}.mode strong{display:block;margin-bottom:.4rem}.metric{font-family:'Fira Code';font-size:1.45rem;font-weight:600;margin:.35rem 0}.label{font-size:.71rem;text-transform:uppercase;letter-spacing:.1em;color:var(--m)}.row{display:grid;grid-template-columns:38px 1fr auto;gap:13px;align-items:center;padding:.82rem .1rem;border-bottom:1px solid rgba(51,65,85,.6)}.row:last-child{border-bottom:0}.num{font-family:'Fira Code';color:#67e8f9;font-size:.72rem}.state{font-family:'Fira Code';font-size:.68rem;color:var(--g);border:1px solid rgba(74,222,128,.3);padding:4px 7px;border-radius:5px}.result{padding:1.2rem;border-radius:14px;border:1px solid var(--l);background:rgba(2,6,23,.35)}.word{font-family:'Fira Code';font-size:clamp(2.3rem,4vw,4rem);font-weight:600;letter-spacing:-.08em;margin:.15rem 0}.real{color:var(--g)}.fake{color:var(--r)}.track{height:8px;background:#172033;border-radius:8px;overflow:hidden;margin:7px 0 4px}.fill{height:100%;border-radius:8px;background:linear-gradient(90deg,#38bdf8,#22d3ee)}.notice{border-left:3px solid #fbbf24;background:rgba(251,191,36,.08);padding:.85rem 1rem;color:#fde68a;border-radius:0 8px 8px 0;font-size:.9rem;line-height:1.5}.empty{border:1px dashed #3b5172;border-radius:13px;padding:2rem;text-align:center;color:var(--m)}.hrow{display:grid;grid-template-columns:54px 1fr auto;gap:11px;align-items:center;border-bottom:1px solid rgba(51,65,85,.55);padding:.8rem 0}.thumb{width:52px;height:42px;border-radius:6px;object-fit:cover;border:1px solid var(--l)}.foot{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--l);font-size:.78rem;color:var(--m)}@media(max-width:760px){.block-container{padding:1rem}.row{grid-template-columns:31px 1fr}.state{display:none}.hrow{grid-template-columns:44px 1fr}.hrow .tag{display:none}}@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}</style>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_service() -> AnalysisService: return AnalysisService(Predictor())
def service() -> Optional[AnalysisService]:
    try: return get_service()
    except Exception as exc: st.session_state['model_error']=str(exc); return None
def init() -> None:
    for k,v in {'page':'Home','history':[],'analysis':None,'analysis_key':None,'mode':'General image analysis'}.items(): st.session_state.setdefault(k,v)
def top(eyebrow: str, title: str, desc: str) -> None: st.markdown(f'<div class="eyebrow">{eyebrow}</div><h1 style="margin:.35rem 0 .45rem">{title}</h1><p class="subtle" style="max-width:760px">{desc}</p>', unsafe_allow_html=True)

def sidebar(svc: Optional[AnalysisService]) -> str:
    st.sidebar.markdown('<div class="brand"><div class="mark">TL</div><div><div class="name">TRUTHLENS</div><div class="sub">IMAGE AUTHENTICITY LAB</div></div></div>',unsafe_allow_html=True)
    pages=('Home','Verify image','History','Research & limits'); page=st.sidebar.radio('Navigation',pages,index=pages.index(st.session_state['page']),label_visibility='collapsed');st.session_state['page']=page
    st.sidebar.markdown('---');st.sidebar.markdown('**Inference status**')
    if svc: st.sidebar.success('Real MobileNetV2 model ready');st.sidebar.caption('Local model artifact · binary classification')
    else: st.sidebar.error('Model unavailable');st.sidebar.caption('No demo predictions are generated.')
    st.sidebar.markdown('---');st.sidebar.caption("TruthLens estimates image-level deepfake patterns. It does not verify identity, documents, or authenticity as a fact.")
    return page

def home(svc: Optional[AnalysisService]) -> None:
    badge='<span class="chip"><span class="dot"></span>REAL MODEL · LOCAL INFERENCE</span>' if svc else '<span class="chip">MODEL UNAVAILABLE</span>'
    st.markdown(f'<section class="hero"><div class="eyebrow">Explainable AI-powered image analysis</div><h1>Verify before<br>you trust.</h1><p>TruthLens helps researchers and everyday users inspect a single image for model-detected deepfake-related visual patterns—without turning a probability into a verdict.</p>{badge}</section>',unsafe_allow_html=True)
    cols=st.columns(3)
    for col,title,desc in zip(cols,('Profile-picture review','General image analysis','Research workflow'),('Inspect visual deepfake patterns. This is not identity verification.','Run a focused binary classifier on a JPG, PNG, or WebP image.','Inspect model inputs, probabilities, and explainability availability.')):
        with col: st.markdown(f'<div class="mode"><strong>{title}</strong><div class="subtle">{desc}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">What happens to an image</div>',unsafe_allow_html=True); a,b=st.columns([1.1,.9])
    with a: st.markdown('<div class="panel"><h3>Transparent analysis path</h3><p class="subtle">Image → validation → 224 × 224 preprocessing → MobileNetV2 → binary probability → Grad-CAM → structured report.</p><p class="subtle">Explainability uses the same loaded local model and actual uploaded image. It is shown only when the backend computation succeeds.</p></div>',unsafe_allow_html=True)
    with b: st.markdown('<div class="notice"><strong>Responsible use.</strong><br>Predictions are probabilistic. Do not use this tool as sole evidence in identity, legal, financial, safety, or other high-stakes decisions.</div>',unsafe_allow_html=True)
    if st.button('Open verification workspace'): st.session_state['page']='Verify image';st.rerun()

def read_upload(f: Any) -> tuple[Optional[Image.Image],Optional[str],bytes]:
    data=f.getvalue()
    if len(data)>MAX_BYTES:return None,'The image is larger than 10 MB. Choose a smaller file and try again.',data
    if f.type not in {'image/jpeg','image/png','image/webp'}:return None,'Unsupported format. Upload a JPG, PNG, or WebP image.',data
    try:
        probe=Image.open(io.BytesIO(data));probe.verify();return Image.open(io.BytesIO(data)),None,data
    except (UnidentifiedImageError,OSError):return None,'This file cannot be read as a valid image.',data
def pipeline(done: bool=False) -> None:
    st.markdown('<div class="section">Analysis pipeline</div><div class="panel">',unsafe_allow_html=True)
    st.markdown(''.join(f'<div class="row"><div class="num">{n}</div><div><b>{t}</b><div class="subtle" style="font-size:.8rem">{d}</div></div><div class="state">{"COMPLETE" if done else "READY"}</div></div>' for n,t,d in STEPS),unsafe_allow_html=True);st.markdown('</div>',unsafe_allow_html=True)
def restore(data: dict[str,Any])->AnalysisResponse:return AnalysisResponse(**{**data,'metadata':ImageMetadata(**data['metadata'])})

def results(r: AnalysisResponse,img: Image.Image,name: str) -> None:
    fake=r.prediction=='fake';word='LIKELY FAKE' if fake else 'LIKELY REAL';color='fake' if fake else 'real'
    st.markdown('<div class="section">Model prediction</div>',unsafe_allow_html=True);a,b=st.columns([1.05,.95])
    with a: st.markdown(f'<div class="result"><div class="eyebrow">{r.inference_mode} · MOBILENETV2</div><div class="word {color}">{word}</div><div class="subtle">Model confidence in predicted class</div><div class="metric">{r.confidence:.2f}%</div><div class="track"><div class="fill" style="width:{r.confidence}%"></div></div><div class="subtle">Decision threshold: {r.threshold:.2f}</div></div>',unsafe_allow_html=True)
    with b: st.image(img,caption=name,use_container_width=True)
    cols=st.columns(4)
    for col,(n,v) in zip(cols,(('Real probability',f'{r.real_probability*100:.2f}%'),('Fake probability',f'{r.fake_probability*100:.2f}%'),('Processing time',f'{r.processing_time_ms} ms'),('Input resolution',f'{r.metadata.width} × {r.metadata.height}'))):
        with col:st.markdown(f'<div class="panel"><div class="label">{n}</div><div class="metric">{v}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Explainability</div>',unsafe_allow_html=True)
    if r.gradcam_available and r.gradcam_heatmap and r.gradcam_overlay:
        opacity=st.slider('Heatmap overlay opacity',min_value=0.0,max_value=1.0,value=.45,step=.05,help='Adjusts only the display blend; model attention is unchanged.')
        overlay=create_overlay(img,r.gradcam_heatmap,opacity)
        c1,c2,c3=st.columns(3)
        with c1: st.image(img,caption='Original image',use_container_width=True)
        with c2: st.image(r.gradcam_heatmap,caption='Grad-CAM heatmap',use_container_width=True)
        with c3: st.image(overlay,caption=f'Grad-CAM overlay · {opacity:.0%} opacity',use_container_width=True)
        st.caption(f'Grad-CAM target: class `{r.target_class}` · layer `{r.target_layer}`')
        st.markdown('<div class="notice">Grad-CAM highlights image regions that contributed more strongly to the model\'s prediction. These highlighted regions represent model attention and should not be interpreted as definitive proof of manipulation.</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="notice"><strong>Grad-CAM unavailable for this analysis.</strong><br>The REAL MODEL prediction is still valid; no placeholder visualization is shown.</div>',unsafe_allow_html=True)
        if r.gradcam_error:
            with st.expander('Grad-CAM technical details'): st.code(r.gradcam_error)
    st.markdown('<div class="section">Analysis record</div>',unsafe_allow_html=True);st.markdown(f'<div class="panel"><div class="subtle"><b>Model:</b> {r.model} · <b>Mode:</b> {r.inference_mode} · <b>Timestamp:</b> {r.analyzed_at}<br><b>File:</b> {name} · <b>Type:</b> {r.metadata.format} · <b>Aspect ratio:</b> {r.metadata.aspect_ratio} · <b>Preprocessing:</b> RGB → 224 × 224 → [-1, 1]</div></div>',unsafe_allow_html=True);st.info('This is a model prediction, not proof of authenticity. Classifier errors and domain shift can produce false positives or false negatives.')

def verify(svc: Optional[AnalysisService]) -> None:
    top('Verification workspace','Inspect an image with context.','Choose an analysis context, upload one image, and receive a transparent structured model response.')
    modes=('Profile picture verification','General image analysis','Research analysis');st.session_state['mode']=st.radio('Analysis context',modes,index=modes.index(st.session_state['mode']),horizontal=True)
    if st.session_state['mode']==modes[0]:st.caption('This examines visual deepfake-related patterns only. It does not establish identity or validate identity documents.')
    f=st.file_uploader('Upload an image for analysis',type=['jpg','jpeg','png','webp'],help='Maximum: 10 MB. JPG, PNG, WebP.')
    if not f:st.markdown('<div class="empty"><strong>Drop an image here or choose a file.</strong><br><span class="subtle">TruthLens validates every upload before inference.</span></div>',unsafe_allow_html=True);pipeline();return
    img,error,data=read_upload(f);key=f'{f.name}:{len(data)}'
    if error:st.error(error);pipeline();return
    if st.session_state['analysis_key']!=key:st.session_state['analysis']=None
    a,b=st.columns([.8,1.2])
    with a:st.image(img,caption=f'{f.name} · {img.width} × {img.height}px',use_container_width=True)
    with b:
        st.markdown(f'<div class="panel"><h3>File is ready</h3><p class="subtle">Format: {img.format or "Unknown"} · Size: {len(data)/1024:.1f} KB<br>Original image remains the preview; model input is resized and normalized.</p></div>',unsafe_allow_html=True)
        if not svc:st.error('The real model is unavailable. No demo prediction will be substituted.')
        run=st.button('Run model analysis',disabled=svc is None,use_container_width=True)
    if run and svc:
        p=st.progress(0,text='Preparing analysis')
        try:
            for value,text in ((18,'Validating image'),(38,'Preprocessing pixels'),(63,'Extracting visual features'),(83,'Classifying image')):p.progress(value,text=text);time.sleep(.12)
            r=svc.analyze(img);p.progress(100,text='Analysis complete');st.session_state['analysis']=r.to_dict();st.session_state['analysis_key']=key
            rec={'filename':f.name,'mode':st.session_state['mode'],'prediction':r.prediction,'confidence':r.confidence,'timestamp':r.analyzed_at,'preview':data};st.session_state['history']=[rec,*[x for x in st.session_state['history'] if x['filename']!=f.name]][:10]
        except Exception as exc:st.error(f'Analysis could not be completed: {exc}')
    pipeline(st.session_state['analysis'] is not None)
    if st.session_state['analysis'] and st.session_state['analysis_key']==key:results(restore(st.session_state['analysis']),img,f.name)

def history() -> None:
    top('Analysis history','Recent model runs.','Records remain in this browser session only. No sample predictions are shown as live analysis.')
    if not st.session_state['history']:st.markdown('<div class="empty">No analysis records yet. Run a real model analysis in the verification workspace.</div>',unsafe_allow_html=True);return
    for x in st.session_state['history']:
        src='data:image/jpeg;base64,'+base64.b64encode(x['preview']).decode();c='fake' if x['prediction']=='fake' else 'real';st.markdown(f'<div class="hrow"><img class="thumb" src="{src}" alt="Uploaded image preview"><div><b>{x["filename"]}</b><div class="subtle">{x["mode"]} · {x["timestamp"]}</div></div><div class="tag {c}">{x["prediction"].upper()} · {x["confidence"]:.2f}%</div></div>',unsafe_allow_html=True)
def research() -> None:
    top('Research & responsible AI','A transparent model, not a truth machine.','The repository implementation is the source of truth for these details.');a,b=st.columns(2)
    with a:st.markdown('<div class="panel"><h3>Implemented pipeline</h3><p class="subtle">Image → RGB conversion → 224 × 224 resize → normalize to [-1, 1] → MobileNetV2 → global average pooling → dropout → dense head → sigmoid real probability.</p><p class="subtle">Classes: fake (0), real (1). Threshold: 0.50.</p></div>',unsafe_allow_html=True)
    with b:st.markdown('<div class="panel"><h3>Training configuration</h3><p class="subtle">ImageNet-initialized MobileNetV2 backbone, initially frozen; 128-unit ReLU head; dropout 0.30; batch size 32; learning rate 1e-4; 5 configured epochs.</p><p class="subtle">Grad-CAM uses the backbone’s final activation layer, <span class="mono">out_relu</span>, for the predicted fake or real class.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Limitations</div><div class="notice"><strong>Interpret with care.</strong><br>Detection is probabilistic and depends on the dataset and image domain. False positives and false negatives can occur. Grad-CAM, when genuinely available, describes model attention—not manipulation ground truth. An image classifier cannot independently establish identity and should never be the sole basis for high-stakes decisions.</div>',unsafe_allow_html=True)

def main() -> None:
    style();init();svc=service();page=sidebar(svc)
    if page=='Home':home(svc)
    elif page=='Verify image':verify(svc)
    elif page=='History':history()
    else:research()
    st.markdown('<div class="foot">TruthLens · Explainable AI-powered deepfake detection · Predictions are probabilistic, not conclusive.</div>',unsafe_allow_html=True)
if __name__=='__main__':main()
