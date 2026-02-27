from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import torch
import argparse
from torch import autograd 
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
import datetime
from torch.nn import functional as F
#from apex import amp
import random
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default=r"D:\workSpace\SAM-GAN\workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="LiTS_GT_sam-med2d-2.5D_GAN", help="run model name")
    parser.add_argument("--epochs", type=int, default=120, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number")
    parser.add_argument("--data_path", type=str, default=r"D:\workSpace\dataset\Task03_Liver\SIF-SAMGAN_LiTS_data\divided_dataset", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume") 
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    
    
    parser.add_argument("--sam_checkpoint", type=str, default=r"D:\workSpace\SAM-GAN\pretrain_model\sam-med2d_b.pth", help="sam checkpoint")

    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--use_amp", type=bool, default=False, help="use amp")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=args.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    
    
    #fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    
    # Ensure grad_outputs matches the shape of d_interpolates
    fake = torch.ones_like(d_interpolates, requires_grad=False)
    
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# 模型定义
class Discriminator(nn.Module):
    def __init__(self, output_features=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1 * 256 * 256, 512),  # 假设输入图像平铺后的大小为65536
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, output_features),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return torch.sigmoid(validity)    

def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )

    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )
  
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    
    return masks, low_res_masks, iou_predictions


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, discriminator, optimizer_G, optimizer_D):
    train_loader = tqdm(train_loader)
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)  # 确保这与你的度量数量相匹配
    discriminator_steps = 5 # 指定每训练5次判别器后训练一次生成器 
    discriminator_count = 0 
    for batch, batched_input in enumerate(train_loader):
        batched_input = stack_dict_batched(batched_input)  # 假设的函数，你需要根据你的数据结构自定义
        batched_input = to_device(batched_input, args.device)  # 将数据移至正确的设备

        if random.random() > 0.5:
            batched_input["point_coords"] = None
            flag = "boxes"
        else:
            batched_input["boxes"] = None
            flag = "point"

        for n, value in model.image_encoder.named_parameters():
            value.requires_grad = ("Adapter" in n)

        labels = batched_input["label"]
        image_embeddings = model.image_encoder(batched_input["image"])
        B, _, _, _ = image_embeddings.shape
        image_embeddings_repeat = [image_embeddings[i].repeat(args.mask_num, 1, 1, 1) for i in range(B)]
        image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

        masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=False)
        loss = criterion(masks, labels, iou_predictions)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_validity = discriminator(labels)
        fake_validity = discriminator(masks.detach())
        d_loss = 1 - torch.mean(real_validity) + torch.mean(fake_validity)
        d_loss.backward()
        optimizer_D.step()
        discriminator_count += 1
        
        
        
        if discriminator_count % discriminator_steps == 0:
        # Train Generator
            optimizer_G.zero_grad()
            fake_validity = discriminator(masks)
            G_loss = 1 - torch.mean(fake_validity)
            total_loss = loss + G_loss
            total_loss.backward()
            optimizer_G.step()

            train_losses.append(total_loss.item())

        # Calculate metrics for batch
        batch_metrics = SegMetrics(masks, labels, args.metrics)  # 你需要定义这个函数
        train_iter_metrics = [train_iter_metrics[i] + batch_metrics[i] for i in range(len(args.metrics))]

        if (batch + 1) % 50 == 0:
            #print(f'Epoch: {epoch+1}, Batch: {batch+1}, Loss: {total_loss.item()}, Metrics: {batch_metrics}')
            # 假设 args.metrics 是一个包含度量名称的列表，例如 ['accuracy', 'loss', 'iou']
            # 并且 batch_metrics 是一个 numpy 数组，其中包含这些度量的值
            if isinstance(batch_metrics, np.ndarray):
                batch_metrics_str = ", ".join([f"{name}: {value:.4f}" for name, value in zip(args.metrics, batch_metrics)])
                print(f'Epoch: {epoch+1}, Batch: {batch+1}, Loss: {total_loss.item():.4f}, Metrics: {batch_metrics_str}')
            else:
                print(f'Epoch: {epoch+1}, Batch: {batch+1}, Loss: {total_loss.item():.4f}, Metrics: {batch_metrics}')
        torch.cuda.empty_cache()

    # Normalize metrics over all batches
    train_iter_metrics = [metric / len(train_loader) for metric in train_iter_metrics]

    return train_losses, train_iter_metrics


def main(args):
    
    discriminator = Discriminator(output_features=9).to(args.device)   
    model = sam_model_registry[args.model_type](args).to(args.device) 
    generator = model
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()
    # 定义优化器
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma = 0.5)
        print('*******Use MultiStepLR')

    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            print(f"*******load {args.resume}")

    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        print("*******Mixed precision with Apex")
    else:
        print('*******Do not use mixed precision')

    # 检查并打印 image_encoder 的参数及其 requires_grad 状态 
    # for n, value in model.image_encoder.named_parameters(): 
    #     print(f"Parameter: {n}, requires_grad: {value.requires_grad}")
    

    train_dataset = TrainingDataset(args.data_path, image_size=args.image_size, point_num=1,  requires_name = False)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=4)
    print('*******Train data:', len(train_dataset))   

    loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))

    best_loss = 1e10
    l = len(train_loader)
    best_dice = 0.0
    for epoch in range(0, args.epochs):
        model.train()
        train_metrics = {}
        start = time.time()
        os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)
        train_losses, train_iter_metrics = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion,discriminator,optimizer_G,optimizer_D)

        if args.lr_scheduler is not None:
            scheduler.step()

        
        # 正确计算 train_metrics
        train_metrics = {args.metrics[i]: train_iter_metrics[i] / len(train_loader) for i in range(len(train_iter_metrics))}
        

        # 安全地转换为字符串用于显示
        train_metrics_str = {k: f"{v:.4f}" if isinstance(v, (int, float)) else v for k, v in train_metrics.items()}
       
        #train_iter_metrics = [metric / l for metric in train_iter_metrics]
        #train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}
        #train_metrics_str = {k: f"{v:.4f}" for k, v in train_metrics.items()} 
        current_dice = float(train_metrics['dice']) # 确保'dice'是度量名称之一 
        
        average_loss = np.mean(train_losses)
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr
        loggers.info(f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics_str}")

        # # 检查是否达到新的最高Dice系数
        # if current_dice > best_dice:
        #     best_dice = current_dice
        #     save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch+1}_best_dice.pth")
        #     state = {'model': model.float().state_dict(), 'optimizer': optimizer.state_dict()}
        #     torch.save(state, save_path)
        #     loggers.info(f"New best dice score: {best_dice:.4f} saved to {save_path}")
        #     if args.use_amp:
        #         model = model.half()
        ##最佳loss保存
        #if average_loss < best_loss:
        #    best_loss = average_loss
        #    save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch+1}_sam.pth")
        #    state = {'model': model.float().state_dict(), 'optimizer': optimizer}
        #    torch.save(state, save_path)
        save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch+1}_sam.pth")
        state = {'model': model.float().state_dict(),'optimizer': optimizer}
        torch.save(state, save_path)
        if args.use_amp:
            model = model.half()

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))


if __name__ == '__main__':
    args = parse_args()
    main(args)


