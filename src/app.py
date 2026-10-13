import json,subprocess,sys,tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import project_root

st.set_page_config(page_title='Deepfake Video Screening',page_icon='🎥',layout='wide')
st.title('Deepfake Detection')
st.caption('Educational screening prototype.')
root=project_root()
video_tab, image_tab = st.tabs(['Video analysis', 'Image analysis'])

with video_tab:
    uploaded=st.file_uploader('Upload a video',type=['mp4','avi','mov','mkv','webm'])
    samples=st.slider('Frames to sample',8,48,24,4); threshold=st.slider('Fake probability threshold',.3,.8,.5,.05)
    if uploaded and st.button('Analyze video',type='primary'):
        if not (root/'checkpoints/best_model.pt').exists(): st.error('Train the model first. Check README.md.')
        else:
            tmp_path=None
            try:
                with tempfile.NamedTemporaryFile(delete=False,suffix=Path(uploaded.name).suffix or '.mp4') as f:
                    f.write(uploaded.getbuffer()); tmp_path=f.name
                with st.spinner('Analyzing sampled frames...'):
                    proc=subprocess.run([sys.executable,'-m','src.inference.predict_video','--video',tmp_path,'--samples',str(samples),'--threshold',str(threshold)],cwd=root,capture_output=True,text=True,timeout=600)
                if proc.returncode:
                    st.error('Analysis failed'); st.code(proc.stderr or proc.stdout)
                else:
                    report=root/'outputs/reports'/f'{Path(tmp_path).stem}_prediction.json'
                    result=json.loads(report.read_text())
                    c1,c2,c3=st.columns(3); c1.metric('Video result',result['video_prediction']); c2.metric('Faces scored',result['faces_scored']); c3.metric('Mean fake probability',f"{result['mean_fake_probability']:.1%}" if 'mean_fake_probability' in result else 'Unavailable')
                    st.write(result['reason']); df=pd.DataFrame(result['frames']); st.subheader('Frame-level evidence'); st.dataframe(df,use_container_width=True)
                    if 'fake_probability' in df and df.fake_probability.notna().any():
                        st.line_chart(df.dropna(subset=['fake_probability']).set_index('time_seconds')['fake_probability'])
                    st.download_button('Download JSON report',json.dumps(result,indent=2),'deepfake_report.json','application/json')
            except Exception as e: st.error(str(e))
            finally:
                if tmp_path:
                    try: Path(tmp_path).unlink(missing_ok=True)
                    except Exception: pass

with image_tab:
    image_upload=st.file_uploader('Upload an image',type=['jpg','jpeg','png','webp'],key='image_upload')
    image_threshold=st.slider('Fake probability threshold',.3,.8,.5,.05,key='image_threshold')
    if image_upload:
        st.image(image_upload,caption=image_upload.name,use_container_width=True)
    if image_upload and st.button('Analyze image',type='primary'):
        if not (root/'checkpoints/best_model.pt').exists(): st.error('Train the model first. Check README.md.')
        else:
            tmp_path=None
            try:
                with tempfile.NamedTemporaryFile(delete=False,suffix=Path(image_upload.name).suffix or '.jpg') as f:
                    f.write(image_upload.getbuffer()); tmp_path=f.name
                with st.spinner('Analyzing image...'):
                    proc=subprocess.run([sys.executable,'-m','src.inference.predict_image','--image',tmp_path,'--threshold',str(image_threshold)],cwd=root,capture_output=True,text=True,timeout=600)
                if proc.returncode:
                    st.error('Analysis failed'); st.code(proc.stderr or proc.stdout)
                else:
                    report=root/'outputs/reports'/f'{Path(tmp_path).stem}_image_prediction.json'
                    result=json.loads(report.read_text())
                    c1,c2=st.columns(2); c1.metric('Image result',result['image_prediction']); c2.metric('Fake probability',f"{result['fake_probability']:.1%}")
                    st.write(result['reason'])
                    st.caption(f"Scored input: {result['input_source'].replace('_',' ')}")
                    st.download_button('Download JSON report',json.dumps(result,indent=2),'deepfake_image_report.json','application/json')
            except Exception as e: st.error(str(e))
            finally:
                if tmp_path:
                    try: Path(tmp_path).unlink(missing_ok=True)
                    except Exception: pass
