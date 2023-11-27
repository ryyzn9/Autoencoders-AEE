import tqdm
import torch
from tqdm import tqdm
from torchvision import transforms

def train(dataloaders, model,loss_fn,optim ,epochs,device, noisy= None,super_res= None):
    
    tqdm_iter = tqdm(range(epochs))#you cn
    train_dataLoader , test_dataLoader = dataloaders[0],dataloaders[1]

    for epoch in tqdm_iter: # you put batch inspite of epo
        model.train()#put the model into the treaning mode 
        train_loss =0.0
        test_loss = 0.0
        for batch in train_dataLoader:
            imgs ,labels = batch
            shapes = list(imgs.shape)
            if super_res is not None:
                shapes[2],shapes[3]=int(shapes[2]/super_res),int(shapes[3]/super_res)
                _transform = transforms.Resize((shapes[2],shapes[3]))
                imgs_transformed = _transform(imgs)
                imgs_transformed = imgs_transformed.to(device)
            imgs = imgs.to(device)
            labels = labels.to(device)

            
            if noisy is not None:
                noisy_tensor = noisy[0]
            else:
                noisy_tensor = torch.zeros(tuple(shapes)).to(device)
            #1
            if super_res is None:
                imgs_noisy = imgs + noisy_tensor
            else:
                imgs_noisy = imgs_transformed + noisy_tensor

            imgs_noisy = torch.clamp(imgs_noisy, 0., 1.)

             #limit the values from 0 to 1 like normalizeation
            preds = model(imgs_noisy)
            loss = loss_fn(preds,imgs)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item()
        model.eval()
        with torch.no_grad():
            for batch in test_dataLoader:
                imgs ,labels = batch
                shapes = list(imgs.shape)
                if super_res is not None:
                    shapes[2],shapes[3]=int(shapes[2]/super_res),int(shapes[3]/super_res)
                    _transform = transforms.Resize((shapes[2],shapes[3]))
                    imgs_transformed = _transform(imgs)
                    imgs_transformed = imgs_transformed.to(device)
                imgs = imgs.to(device)
                labels = labels.to(device)
                if noisy is not None :
                    noisy_tensor = noisy[1]
                else:
                    noisy_tensor = torch.zeros(tuple(shapes)).to(device)
                if super_res is None:   
                    imgs_noisy = imgs + noisy_tensor 
                else :
                     imgs_noisy = imgs_transformed + noisy_tensor
                imgs_noisy = torch.clamp(imgs_noisy,0.,1.) #limit the values from 0 to 1 like normalizeation
                preds = model(imgs_noisy)
                loss = loss_fn(preds,imgs)    
        train_loss /=len(train_dataLoader)
        test_loss /= len(test_dataLoader)
        tqdm_dic = {"train_loss:":train_loss,"test_loss:":test_loss}
        tqdm_iter.set_postfix(tqdm_dic,refresh =True)
        tqdm_iter.refresh()