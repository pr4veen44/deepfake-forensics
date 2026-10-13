import argparse, cv2, numpy as np, pandas as pd
from tqdm import tqdm
from src.utils import project_root

def main():
    p=argparse.ArgumentParser(); p.add_argument('--frames-per-video',type=int,default=12); a=p.parse_args()
    if a.frames_per_video<1: raise SystemExit('frames-per-video must be >= 1')
    root=project_root(); cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    if cascade.empty(): raise SystemExit('OpenCV Haar cascade failed to load. Reinstall opencv-python==4.10.0.84.')
    total=0
    for split in ['train','val','test']:
        csv=root/'data/splits'/f'{split}.csv'
        if not csv.exists(): raise SystemExit('Run python -m src.data.split_videos first.')
        df=pd.read_csv(csv)
        for _,r in tqdm(df.iterrows(),total=len(df),desc=split):
            cap=cv2.VideoCapture(r.path)
            if not cap.isOpened(): print('Cannot open:',r.path); continue
            n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n<1: cap.release(); continue
            out=root/'data/processed'/split/r.label; out.mkdir(parents=True,exist_ok=True)
            for idx in sorted(set(np.linspace(0,n-1,a.frames_per_video,dtype=int).tolist())):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok,frame=cap.read()
                if not ok: continue
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces=cascade.detectMultiScale(gray,1.1,5,minSize=(40,40))
                if len(faces)==0: continue
                x,y,w,h=max(faces,key=lambda q:q[2]*q[3]); px,py=int(.18*w),int(.18*h)
                crop=frame[max(0,y-py):min(frame.shape[0],y+h+py),max(0,x-px):min(frame.shape[1],x+w+px)]
                if crop.size:
                    # Include parent-folder hash in name to reduce filename collisions.
                    import hashlib
                    tag=hashlib.md5(str(r.path).encode()).hexdigest()[:8]
                    if cv2.imwrite(str(out/f'{tag}_{idx:06d}.jpg'),crop): total+=1
            cap.release()
    print('Face crops saved:',total)
if __name__=='__main__': main()
