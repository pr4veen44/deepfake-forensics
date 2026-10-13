import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import project_root, VIDEO_EXTS, seed_everything

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    seed_everything(a.seed); root=project_root(); raw=root/'data/raw'; rows=[]
    for label in ['REAL','FAKE']:
        folder=raw/label
        vids=sorted(x for x in folder.rglob('*') if x.is_file() and x.suffix.lower() in VIDEO_EXTS) if folder.exists() else []
        rows += [{'path':str(x.resolve()),'label':label,'video_id':x.stem} for x in vids]
    df=pd.DataFrame(rows)
    if df.empty: raise SystemExit(f'No videos found. Put videos in {raw}/REAL and {raw}/FAKE.')
    if df.label.nunique()!=2 or df.label.value_counts().min()<3: raise SystemExit('Need at least 3 videos in each class.')
    train,temp=train_test_split(df,test_size=.30,random_state=a.seed,stratify=df.label)
    val,test=train_test_split(temp,test_size=.5,random_state=a.seed,stratify=temp.label)
    out=root/'data/splits'; out.mkdir(parents=True,exist_ok=True)
    for name,part in [('train',train),('val',val),('test',test)]:
        part.to_csv(out/f'{name}.csv',index=False); print(name,len(part),part.label.value_counts().to_dict())
    print('Video-level splits saved:',out)
if __name__=='__main__': main()
