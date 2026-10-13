import argparse, json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from src.utils import project_root, seed_everything
from src.models import build_model, freeze_backbone

def loader(path,train,batch):
    ops=[transforms.Resize((224,224))]
    if train: ops += [transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=.12,contrast=.12)]
    ops += [transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
    ds=datasets.ImageFolder(str(path),transform=transforms.Compose(ops))
    if ds.classes!=['FAKE','REAL'] and set(ds.classes)!={'REAL','FAKE'}: raise SystemExit(f'Expected REAL and FAKE folders in {path}; got {ds.classes}')
    return DataLoader(ds,batch_size=batch,shuffle=train,num_workers=0),ds

def epoch(model,dl,loss_fn,opt,device,training):
    model.train(training); loss_sum=correct=n=0
    for x,y in tqdm(dl,leave=False):
        x,y=x.to(device),y.to(device)
        if training: opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits=model(x); loss=loss_fn(logits,y)
            if training: loss.backward(); opt.step()
        loss_sum+=loss.item()*len(y); correct+=(logits.argmax(1)==y).sum().item(); n+=len(y)
    return loss_sum/max(n,1),correct/max(n,1)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=8); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--unfreeze',action='store_true'); a=p.parse_args()
    root=project_root(); seed_everything(42); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    tr,ds=loader(root/'data/processed/train',True,a.batch_size); va,vds=loader(root/'data/processed/val',False,a.batch_size)
    if len(ds)==0 or len(vds)==0: raise SystemExit('No face images. Run face extraction and check processed folders.')
    model=build_model(True)
    if not a.unfreeze: freeze_backbone(model)
    model.to(device); criterion=nn.CrossEntropyLoss(); opt=torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],lr=a.lr,weight_decay=1e-4)
    best=float('inf'); history=[]; ckdir=root/'checkpoints'; ckdir.mkdir(exist_ok=True)
    for e in range(1,a.epochs+1):
        tl,ta=epoch(model,tr,criterion,opt,device,True); vl,vacc=epoch(model,va,criterion,None,device,False)
        row={'epoch':e,'train_loss':tl,'train_accuracy':ta,'val_loss':vl,'val_accuracy':vacc}; history.append(row); print(row)
        if vl<best:
            best=vl; torch.save({'state_dict':model.state_dict(),'classes':ds.classes,'threshold':.5},ckdir/'best_model.pt'); print('Saved best checkpoint')
    out=root/'outputs/metrics'; out.mkdir(parents=True,exist_ok=True); (out/'training_history.json').write_text(json.dumps(history,indent=2))
if __name__=='__main__': main()
