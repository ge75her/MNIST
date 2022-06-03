

import datetime
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import utils
from data_augment import DataAugmentation
from loss import Loss, clip_gradients
from multicrop import DINOHead, MultiCropWrapper
from vit import VisionTransformer


def train_one_epoch(student,teacher,dino_loss,data_loader,optimizer,lr_schedule,wd_schedule,momentum_schedule,epoch
                    ,total_epochs,clip_grad,freeze_last_layer):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, total_epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 9, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda() for im in images]
        teacher_output=teacher(images)
        student_output=student(images)
        loss=dino_loss(student_output,teacher_output)
        

        optimizer.zero_grad()
        loss.backward()
        param_norms=clip_gradients(student,clip=clip_grad)
        utils.cancel_gradients_last_layer(epoch,student,freeze_last_layer)
        optimizer.step()


        #EMA update teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        #torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_dino(data_path,batch_size,lr,weight_decay,weight_decay_end,min_lr,out_dim,tau,total_epochs,warmup_epochs,momentum_teacher,output_dir,saveckp_freq,
            clip_grad,freeze_last_layer):

    transform = DataAugmentation()
    dataset = ImageFolder(data_path, transform=transform)
    data_loader = DataLoader(dataset,batch_size=32,drop_last=True,)
    print(f"Data loaded: there are {len(dataset)} images.")

    student =VisionTransformer(patch_size=7,drop_path_ratio=0.1,)  # stochastic depth
    teacher = VisionTransformer(patch_size=7)
    embed_dim = student.embed_dim

    student = MultiCropWrapper(student, DINOHead(embed_dim,out_dim=out_dim,use_bn=False,norm_last_layer=True,))
    teacher = MultiCropWrapper(teacher,DINOHead(embed_dim, out_dim=out_dim, use_bn=False),)
    # move networks to gpu
    student=student.cuda()
    teacher=teacher.cuda()
    params_groups = utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)
    # teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())
    dino_loss = Loss(tau=tau,out_dim=out_dim)
    #there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both vit network.")


    lr_schedule = utils.cosine_scheduler(lr * (batch_size* utils.get_world_size()) / 256.,min_lr, total_epochs, len(data_loader),warmup_epochs=warmup_epochs,)
    wd_schedule = utils.cosine_scheduler(weight_decay,weight_decay_end,total_epochs, len(data_loader),)
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(momentum_teacher, 1,total_epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    #start to training
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
    os.path.join(output_dir, "checkpoint.pth"),
    run_variables=to_restore,
    student=student,
    teacher=teacher,
    optimizer=optimizer,
    dino_loss=dino_loss,)
    
    start_epoch = to_restore["epoch"]
    start_time = time.time()
    print("Starting DINO training !")
    
    for epoch in range(start_epoch,total_epochs):
        train_stats=train_one_epoch(student,teacher,dino_loss,data_loader,optimizer,lr_schedule,wd_schedule,momentum_schedule,epoch,output_dir,
                                    total_epochs,clip_grad,freeze_last_layer)
        save_dict = {'student': student.state_dict(),'teacher': teacher.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch + 1,'loss': dino_loss.state_dict()}
        if epoch%saveckp_freq==0:
            utils.save_on_master(save_dict,os.path.join(output_dir,'checkpoint{epoch}.pth'))
        
        log_stats={**{f'train_{k}': v for k, v in train_stats.items()},'epoch': epoch}
        with (Path(output_dir)/'log.txt').open("a") as f:
            f.write(json.dumps(log_stats)+'\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    if __name__=='__main__':
        data_path='data/MNIST - JPG - training'
        batch_size=32
        lr=0.0005
        weight_decay=0.04
        weight_decay_end=0.4
        min_lr=1e-6

        total_epochs=100
        warmup_epochs=10
        momentum_teacher=0.995
        out_dim=16384
        tau=0.07
        saveckp_freq=10

        clip_grad=3.0
        freeze_last_layer=1
        output_dir='log'
        train_dino(data_path,batch_size,lr,weight_decay,weight_decay_end,min_lr,out_dim,tau,total_epochs,warmup_epochs,momentum_teacher,output_dir,saveckp_freq,
            clip_grad,freeze_last_layer)

