import argparse,json,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from torchvision import transforms
from src.utils import project_root
from src.models import build_model

TF=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
def main():
    p=argparse.ArgumentParser(); p.add_argument('--video',required=True); p.add_argument('--checkpoint',default='checkpoints/best_model.pt'); p.add_argument('--samples',type=int,default=24); p.add_argument('--threshold',type=float,default=None); a=p.parse_args()
    video=Path(a.video)
    if not video.exists(): raise SystemExit(f'Video not found: {video}')
    root=project_root(); ck=Path(a.checkpoint); ck=ck if ck.is_absolute() else root/ck
    if not ck.exists(): raise SystemExit('Checkpoint missing. Train the model first.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); checkpoint=torch.load(ck,map_location=device)
    model=build_model(False); model.load_state_dict(checkpoint['state_dict']); model.to(device).eval(); fake_idx=checkpoint.get('classes',['FAKE','REAL']).index('FAKE'); threshold=a.threshold if a.threshold is not None else checkpoint.get('threshold',.5)
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml'); cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise SystemExit('OpenCV could not open this video.')
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 0); rows=[]
    for idx in sorted(set(np.linspace(0,max(n-1,0),max(1,a.samples),dtype=int).tolist())):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok,frame=cap.read()
        if not ok: continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); faces=detector.detectMultiScale(gray,1.1,5,minSize=(40,40))
        timestamp=round(idx/fps,2) if fps>0 else None
        if len(faces)==0: rows.append({'frame':idx,'time_seconds':timestamp,'status':'NO_FACE_DETECTED'}); continue
        x,y,w,h=max(faces,key=lambda f:f[2]*f[3]); px,py=int(.18*w),int(.18*h)
        crop=frame[max(0,y-py):min(frame.shape[0],y+h+py),max(0,x-px):min(frame.shape[1],x+w+px)]
        rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB); tensor=TF(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad(): prob=float(torch.softmax(model(tensor),1)[0,fake_idx].item())
        rows.append({'frame':idx,'time_seconds':timestamp,'fake_probability':round(prob,4),'status':'LIKELY_FAKE' if prob>=threshold else 'LIKELY_REAL'})
    cap.release(); probs=[r['fake_probability'] for r in rows if 'fake_probability' in r]
    result={'video':str(video.resolve()),'sampled_frames':len(rows),'faces_scored':len(probs),'threshold':float(threshold),'frames':rows}
    if len(probs)<3: result.update(video_prediction='UNCERTAIN',reason='Fewer than three faces were scored.')
    else:
        mean=float(np.mean(probs)); result.update(mean_fake_probability=round(mean,4),video_prediction='LIKELY_FAKE' if mean>=threshold else 'LIKELY_REAL',reason='Mean probability across sampled detected faces; this is not forensic proof.')
    out=root/'outputs/reports'; out.mkdir(parents=True,exist_ok=True); report=out/f'{video.stem}_prediction.json'; report.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2)); print('Report saved:',report)
if __name__=='__main__': main()
