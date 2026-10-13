import json, numpy as np, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,roc_auc_score
from src.utils import project_root
from src.models import build_model

def main():
    root=project_root(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    ds=datasets.ImageFolder(str(root/'data/processed/test'),transform=tf)
    if not len(ds): raise SystemExit('No test images found.')
    ck=root/'checkpoints/best_model.pt'
    if not ck.exists(): raise SystemExit('Train the model first; checkpoint missing.')
    model=build_model(False); model.load_state_dict(torch.load(ck,map_location=device)['state_dict']); model.to(device).eval()
    yt=[]; yp=[]; prob=[]
    with torch.no_grad():
        for x,y in DataLoader(ds,batch_size=16,shuffle=False):
            q=torch.softmax(model(x.to(device)),1)[:,ds.class_to_idx['FAKE']].cpu().numpy()
            prob.extend(q.tolist()); yt.extend(y.numpy().tolist()); yp.extend((q>=.5).astype(int).tolist())
    fake_idx=ds.class_to_idx['FAKE']; yfake=[int(y==fake_idx) for y in yt]
    pred_idx=[ds.class_to_idx['FAKE'] if q>=.5 else ds.class_to_idx['REAL'] for q in prob]
    m={'test_images':len(ds),'classes':ds.classes,'accuracy':float(accuracy_score(yt,pred_idx)),
       'precision_fake':float(precision_score(yfake,[int(p==fake_idx) for p in pred_idx],zero_division=0)),
       'recall_fake':float(recall_score(yfake,[int(p==fake_idx) for p in pred_idx],zero_division=0)),
       'f1_fake':float(f1_score(yfake,[int(p==fake_idx) for p in pred_idx],zero_division=0)),
       'confusion_matrix_labels':ds.classes,'confusion_matrix':confusion_matrix(yt,pred_idx,labels=list(range(len(ds.classes)))).tolist(),
       'classification_report':classification_report(yt,pred_idx,target_names=ds.classes,zero_division=0,output_dict=True)}
    try: m['roc_auc_fake']=float(roc_auc_score(yfake,prob))
    except ValueError: m['roc_auc_fake']=None
    out=root/'outputs/metrics'; out.mkdir(parents=True,exist_ok=True); (out/'test_metrics.json').write_text(json.dumps(m,indent=2))
    cm=np.array(m['confusion_matrix']); fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(cm)
    ax.set_xticks(range(len(ds.classes)),ds.classes); ax.set_yticks(range(len(ds.classes)),ds.classes); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('Test Confusion Matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=160); plt.close(fig); print(json.dumps(m,indent=2))
if __name__=='__main__': main()
