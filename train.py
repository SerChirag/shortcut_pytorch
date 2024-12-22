from diffusers.models import AutoencoderKL

from PIL import Image
import numpy as np
import pandas as pd
import io
import os
from copy import deepcopy
from collections import OrderedDict
import math

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from torchmetrics.image.fid import FrechetInceptionDistance

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback

from model import DiT_B_2, EMACallback
from utils import create_targets, create_targets_naive

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EVAL_SIZE = 8

NUM_CLASSES = 1
CLASS_DROPOUT_PROB = 1.0

N_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOG_EVERY = 3
CFG_SCALE = 0.0


class CelebaHQDataset(Dataset):
    def __init__(self, parquet_path, transform=None, size=256):
        

        parquet_names = os.listdir(parquet_path)
        parquet_paths = [os.path.join(parquet_path, parquet_name) for parquet_name in parquet_names]

        if len(parquet_paths) < 1:
            return FileNotFoundError

        parquets = []

        for i in range(len(parquet_paths)):
            parquet_i = pd.read_parquet(parquet_paths[i])
            parquets.append(parquet_i)
            
        self.data = pd.concat(parquets, axis=0)
        self.size = size
        self.transform = transform

        # print(f"self.data: {self.data}")

    def __len__(self):
        
        return self.data.shape[0]

    def __getitem__(self, idx):

        data_i = self.data.iloc[[idx]]

        image_i = Image.open(io.BytesIO(data_i['image.bytes'].item())).resize((self.size, self.size))

        if self.transform is not None:
            image_i = self.transform(image_i)

        label_i = data_i['label'].item()

        return image_i, label_i

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def train_epoch(train_dataloader, dit, ema, vae, optimizer):

    loss_fn = torch.nn.MSELoss()
    
    dit.train()
    
    total_loss = 0.0
    for batch, (images, labels) in enumerate(tqdm(train_dataloader)):

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # print(f"torch.mean(images): {torch.mean(images)} | torch.std(images): {torch.std(images)}")

        with torch.no_grad():
            
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            
        print(f"latents.shape: {latents.shape}")
        # print(f"labels.shape: {labels.shape}")

        x_t, v_t, t, dt_base, labels_dropped = create_targets(latents, labels, dit)

        # print(f"x_t.shape: {x_t.shape}")
        # print(f"v_t.shape: {v_t.shape}")
        # print(f"t.shape: {t.shape}")
        # print(f"dt_base.shape: {dt_base.shape}")
        # print(f"labels_dropped.shape: {labels_dropped.shape}")

        # exit()

        v_prime = dit(x_t, t, dt_base, labels)

        # print(f"v_prime.shape: {v_prime.shape}")
        # print(f"v_prime: {v_prime}")

        loss = loss_fn(v_prime, v_t)

        total_loss += loss.item()
        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()

        update_ema(ema, dit)

    return total_loss / len(train_dataloader)

def evaluate(dit, vae, val_dataloader, epoch):

    dit.eval()

    images, labels_real = next(iter(val_dataloader))
    images, labels_real = images[:EVAL_SIZE].to(DEVICE), labels_real[:EVAL_SIZE].to(DEVICE)
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor


    labels_uncond = torch.ones_like(labels_real, dtype=torch.int32) * NUM_CLASSES

    fid = FrechetInceptionDistance().to(DEVICE)

    
    images = 255 * ((images - torch.min(images)) / (torch.max(images) - torch.min(images) + 1e-8))

    # all start from same noise
    eps = torch.randn_like(latents).to(DEVICE)

    

    denoise_timesteps_list = [1, 2, 4, 8, 16, 32, 128]
    # run at varying timesteps
    for i, denoise_timesteps in enumerate(denoise_timesteps_list):
        all_x = []
        delta_t = 1.0 / denoise_timesteps # i.e. step size
        fid.reset()

        x = eps

        for ti in range(denoise_timesteps):
            # t should in range [0,1]
            t = ti / denoise_timesteps

            t_vector = torch.full((eps.shape[0],), t).to(DEVICE)
            dt_base = torch.ones_like(t_vector).to(DEVICE) * math.log2(denoise_timesteps)


            # if i == len(denoise_timesteps_list)-1:
            #     with torch.no_grad():
            #         v_cond = dit.forward(x, t_vector, dt_base, labels_real)
            #         v_uncond = dit.forward(x, t_vector, dt_base, labels_uncond)

            #         v = v_uncond + CFG_SCALE * (v_cond - v_uncond)
            # else:
            #     # t is same for all latents
            #     with torch.no_grad():
            #         v = dit.forward(x, t_vector, dt_base, labels_real)

            with torch.no_grad():
                v = dit.forward(x, t_vector, dt_base, labels_real)

            x = x + v * delta_t

            if denoise_timesteps <= 8 or ti % (denoise_timesteps//8) == 0 or ti == denoise_timesteps-1:

                with torch.no_grad():
                    decoded = vae.decode(x/vae.config.scaling_factor)[0]
                
                decoded = decoded.to("cpu")

                all_x.append(decoded)
    

        if(len(all_x)==9):
            all_x = all_x[1:]

        # estimate FID metric
        # images_fake = torch.randint(low=0, high=255, size=images.shape).to(torch.uint8).to(DEVICE)
        decoded_denormalized = 255 * ((decoded - torch.min(decoded)) / (torch.max(decoded)-torch.min(decoded)+1e-8))
        
        # generated images
        fid.update(images.to(torch.uint8), real=True)
        fid.update(decoded_denormalized.to(torch.uint8).to(DEVICE), real=False)
        fid_val = fid.compute()
        print(f"denoise_timesteps: {denoise_timesteps} | fid_val: {fid_val}")

        all_x = torch.stack(all_x)

        def process_img(img):
            # normalize in range [0,1]
            img = img * 0.5 + 0.5
            img = torch.clip(img, 0, 1)
            img = img.permute(1,2,0)
            return img
    
        fig, axs = plt.subplots(8, 8, figsize=(30,30))
        for t in range(min(8, all_x.shape[0])):
            for j in range(8):        
                axs[t, j].imshow(process_img(all_x[t, j]), vmin=0, vmax=1)
        
        fig.savefig(f"log_images_tvanilla/epoch:{epoch}_denoise_timesteps:{denoise_timesteps}.png")
        # if i == len(denoise_timesteps_list)-1:
        #     fig.savefig(f"log_images_tvanilla/epoch:{epoch}_cfg.png")
        # else:
        #     fig.savefig(f"log_images_tvanilla/epoch:{epoch}_denoise_timesteps:{denoise_timesteps}.png")
        
        plt.close()

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


# single gpu: [00:55<10:24,  0.64it/s, v_num=5]
# 3 gpus: [03:35<00:00,  0.68it/s, v_num=7]
def main_lightning():

    # should be same as in flax implementation(83'653'863 params)
    
    
    
    # if load from checkpoint:
    # checkpoint_path = "/workspace/shortcut_pytorch/lightning_logs/version_11/checkpoints/last.ckpt"
    # checkpoint = torch.load(checkpoint_path)
    # dit.load_state_dict(checkpoint['state_dict'])    


    train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

    train_dataset = CelebaHQDataset('/workspace/shortcut_pytorch/celeba-hq/data/train', transform=train_transform)
    val_dataset = CelebaHQDataset('/workspace/shortcut_pytorch/celeba-hq/data/val', transform=train_transform)

    print(f"len(train_dataset): {len(train_dataset)}")
    print(f"len(val_dataset): {len(val_dataset)}")

    # good option is 2*num_gpus
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)


    images_i, labels_i = next(iter(train_dataloader))

    # 4 - 2d mean, 2d std
    latent_shape = (BATCH_SIZE, 4, images_i.shape[2]//8, images_i.shape[2]//8)

    dit = DiT_B_2(learn_sigma=False, 
                  num_classes=NUM_CLASSES, 
                  class_dropout_prob=CLASS_DROPOUT_PROB,
                  lightning_mode=True,
                  latent_shape=latent_shape,
                  training_type="naive")

    print(f"count_parameters(dit): {count_parameters(dit)}")


    callbacks = []

    # Define a checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filename="model-{epoch:02d}",
        save_last=True,
    )
    ema_callback = EMACallback(decay=0.999)
    
    callbacks.append(checkpoint_callback)
    callbacks.append(ema_callback)


    trainer = pl.Trainer(max_epochs=500,
                         accelerator="gpu",
                         num_sanity_val_steps=1,
                         check_val_every_n_epoch=5,
                         limit_val_batches=1,
                         devices=[0, 1, 2],
                         strategy="ddp_find_unused_parameters_true",
                         callbacks=callbacks)
    
    trainer.fit(model=dit, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


# single epoch takes: [15:45<00:00,  2.16s/it]
def main():

    train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

    train_dataset = CelebaHQDataset('/workspace/shortcut_pytorch/celeba-hq/data/train', transform=train_transform)
    val_dataset = CelebaHQDataset('/workspace/shortcut_pytorch/celeba-hq/data/val', transform=train_transform)

    print(f"len(train_dataset): {len(train_dataset)}")
    print(f"len(val_dataset): {len(val_dataset)}")

    # good option is 2*num_gpus
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)

    images_i, labels_i = next(iter(train_dataloader))

    # 4 - 2d mean, 2d std
    latent_shape = (BATCH_SIZE, 4, images_i.shape[2]//8, images_i.shape[2]//8)

    dit = DiT_B_2(learn_sigma=False, 
                  num_classes=NUM_CLASSES, 
                  class_dropout_prob=CLASS_DROPOUT_PROB,
                  latent_shape=latent_shape,
                  training_type="naive").to(DEVICE)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
    vae = vae.eval()
    vae.requires_grad_(False)

    print(f"count_parameters(dit): {count_parameters(dit)}")
    ema = deepcopy(dit).to(DEVICE)
    ema.requires_grad_(False)
    
    checkpoint_path = "dit_saved.pth"
    checkpoint = torch.load(checkpoint_path)
    dit.load_state_dict(torch.load(checkpoint_path))



    optimizer = torch.optim.AdamW(dit.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

    update_ema(ema, dit, decay=0)
    ema.eval()

    # evaluate(dit, vae, val_dataloader, 0)

    for i in range(N_EPOCHS):

        epoch_loss = train_epoch(train_dataloader, dit, ema, vae, optimizer)
        exit()
        print(f"epoch_loss: {epoch_loss}")

        if i%LOG_EVERY == 0 and i > 0:
            evaluate(dit, vae, val_dataloader, i)

            torch.save(dit.state_dict(), "dit_saved.pth")


if __name__ == "__main__":
    # main()
    main_lightning()