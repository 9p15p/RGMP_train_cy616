import os
import torch
import argparse, threading, time
import numpy as np
from tqdm import tqdm
from model import *
from utils import *
from tensorboardX import SummaryWriter
from train import *

if __name__ == '__main__':

    model = RGMP().cuda()
    Testset = DAVIS(DAVIS_ROOT, imset='2016/val.txt')
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=True, num_workers=2)

    best_iou = 0
    writer = SummaryWriter()
    for epoch in np.sort([int(d.split('.')[0]) for d in os.listdir('saved_models')]):
        d = '{}.pth'.format(epoch)
        # load saved model if specified
        print('Loading checkpoint {}@Epoch {}{}...'.format(font.BOLD, d, font.END))
        load_name = os.path.join('saved_models',
                                 '{}.pth'.format(epoch))
        state = model.state_dict()
        checkpoint = torch.load(load_name)
        start_epoch = checkpoint['epoch'] + 1
        checkpoint = {k: v for k, v in checkpoint['model'].items() if k in state}
        state.update(checkpoint)
        model.load_state_dict(state)
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            POOLING_MODE = checkpoint['pooling_mode']
        del checkpoint
        torch.cuda.empty_cache()
        print('  - complete!')

        criterion = torch.nn.BCELoss()

        # testing
        with torch.no_grad():
            print('[Val] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
            model.eval()
            loss = 0
            iOU = 0
            pbar = tqdm.tqdm(total=len(Testloader))
            for i, (all_F, all_M, info) in enumerate(Testloader):
                pbar.update(1)
                all_F, all_M = all_F[0], all_M[0]
                seq_name = info['name'][0]
                num_frames = info['num_frames'][0]
                num_objects = info['num_objects'][0]

                B, C, T, H, W = all_M.shape
                all_E = torch.zeros(B, C, T, H, W)
                all_E[:, 0, 0] = all_M[:, :, 0]

                msv_F1, msv_P1, all_M = ToCudaVariable([all_F[:, :, 0], all_E[:, 0, 0], all_M])
                ms = model.Encoder(msv_F1, msv_P1)[0]

                for f in range(0, all_M.shape[2] - 1):
                    output, ms = Propagate_MS(ms, model, all_F[:, :, f + 1], all_E[:, 0, f])
                    all_E[:, 0, f + 1] = output.detach()
                    loss = loss + criterion(output.permute(1, 2, 0), all_M[:, 0, f + 1].float()) / all_M.size(2)
                iOU = iOU + iou(torch.cat((1 - all_E, all_E), dim=1), all_M)

            pbar.close()

            loss = loss / len(Testloader)
            iOU = iOU / len(Testloader)
            writer.add_scalar('Val/BCE', loss, epoch)
            writer.add_scalar('Val/IOU', iOU, epoch)
            print('loss: {}'.format(loss))
            print('IoU: {}'.format(iOU))

            if best_iou < iOU:
                best_iou = iOU
                video_thread = threading.Thread(target=log_mask,
                                                args=(all_F, all_E, info, writer, 'Val/frames', 'Val/masks'))
                video_thread.start()
