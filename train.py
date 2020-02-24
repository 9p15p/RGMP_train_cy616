import torch
import argparse, threading, time
import numpy as np
from tqdm import tqdm
from model import *
from utils import *
import os
from torch.utils.tensorboard import SummaryWriter
import horovod.torch as hvd

# Constants
MODEL_DIR = 'saved_models'
NUM_EPOCHS = 1000
tensorboard_log = 'runs/demo_test'

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())


def Propagate_MS(ms, model, F2, P2):
    """

    :param ms: 参考帧支路的encoder输出的2048channels的feature maps
    :param model: 模型
    :param F2: 某一帧的RGB图像,如果是all_F[:,:,0]那么除了B通道有值,其他为0
    :param P2: 前景概率之和, shape为(1,H,W)
    :return:softmax后的值,r5
    """
    h, w = F2.size()[2], F2.size()[3]

    msv_F2, msv_P2 = ToCudaVariable([F2, P2])
    r5, r4, r3, r2 = model.module.Encoder(msv_F2, msv_P2)
    e2 = model.module.Decoder(r5, ms, r4, r3, r2)

    return F.softmax(e2[0], dim=1)[:, 1], r5


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='RGMP')
    parser.add_argument('--epochs', dest='num_epochs',
                        help='number of epochs to train',
                        default=NUM_EPOCHS, type=int)
    parser.add_argument('--bs', dest='bs',
                        help='batch_size',
                        default=1, type=int)  # 单卡改回1
    parser.add_argument('--num_workers', dest='num_workers',
                        help='num_workers',
                        default=1, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='display interval',
                        default=10, type=int)
    parser.add_argument('--eval_epoch', dest='eval_epoch',
                        help='interval of epochs to perform validation',
                        default=10, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output directory',
                        default=MODEL_DIR, type=str)

    # BPTT
    parser.add_argument('--bptt', dest='bptt_len',
                        help='length of BPTT',
                        default=12, type=int)
    parser.add_argument('--bptt_step', dest='bptt_step',
                        help='step of truncated BPTT',
                        default=4, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=1e-3, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # resume trained model
    parser.add_argument('--loadepoch', dest='loadepoch',
                        help='epoch to load model',
                        default=-1, type=str)

    args = parser.parse_args()
    return args


def log_mask(frames, masks, info, writer, F_name='Train/frames', M_name='Train/masks'):
    print('[tensorboard] Updating mask..')
    _, C, T, H, W = masks.size()
    #     (lh,uh), (lw,uw) = info['pad']
    vid = np.zeros((1, 3, T, H, W))

    print_dbg('[mask] mean: {}, max: {}'.format(torch.mean(masks[:, :, 1]), torch.max(masks[:, :, 1])))
    masks = torch.cat((1 - masks, masks), dim=1)
    for f in range(T):
        E = masks[0, :, f].cpu().data.numpy()
        # make hard label
        E = ToLabel(E)
        #         E = E[lh[0]:-uh[0], lw[0]:-uw[0]]

        # need to implement mask overlay

        img_E = Image.fromarray(E)
        img_E.putpalette(PALETTE)
        arr_E = np.array(E)
        vid[0, :, f, :, :] = np.array(img_E)

    vid_tensor = torch.FloatTensor(vid)
    writer.add_video(F_name, vid_tensor=frames)
    writer.add_video(M_name, vid_tensor=vid_tensor)
    print('[tensorboard] Mask updated')


def log_scalar(writer, name, tensor, iteration):
    value = tensor.cpu().data.numpy()
    writer.add_scalar(name, value, iteration)


def repackage_hidden(h, volatile=False, requires_grad=False):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data, volatile=volatile, requires_grad=requires_grad)
    else:
        return tuple(repackage_hidden(v, volatile=volatile,
                                      requires_grad=requires_grad) for v in h)


def bptt_hsm(data, hidden, target, model, criterion, bptt_len, bptt_step):
    hidden_v = repackage_hidden(hidden, volatile=True)
    data_v, _ = repackage_hidden(data, volatile=True)
    hsm = {-1: repackage_hidden(hidden)}
    intervals = list(enumerate(range(0, data.size(0), bptt_step)))
    # Record states at selective intervals and flag the need for grads.
    # Note we don't need to forward the last interval as we'll do it below.
    # This loop is most of the extra computation for this approach.
    for f_i, f_v in intervals[:-1]:
        output, hidden_v = model(data_v[f_v:f_v + args.bptt_step], hidden_v)
        hsm[f_i] = repackage_hidden(hidden_v, volatile=False,
                                    requires_grad=True)

    save_grad = None
    loss = 0
    for b_i, b_v in reversed(intervals):
        output, h = model(data[b_v:b_v + args.bptt_step], hsm[b_i - 1])
        iloss = criterion(output.view(-1, ntokens),
                          targets[b_v:b_v + args.bptt_step].view(-1))
        if b_v + args.bptt_step >= data.size(0):
            # No gradient from the future needed.
            # These are the hidden states for the next sequence.
            hidden = h
            iloss.backward()
        else:
            variables = [iloss]
            grad_variables = [None]  # scalar = None
            # Associate stored gradients with state variables for 
            # multi-variable backprop
            for l in h:
                variables.append(l)
                g = save_grad.popleft()
                grad_variables.append(g)
            torch.autograd.backward(variables, grad_variables)
        if b_i > 0:
            # Save the gradients left on the input state variables
            save_grad = collections.deque()
            for l in hsm[b_i - 1]:
                # If this fails, could be a non-leaf, in which case exclude;
                # its grad will be handled by a leaf
                assert (l.grad is not None)
                save_grad.append(l.grad)
        loss += iloss.data[0]

    av = 1 / (args.batch_size * args.bptt)
    loss *= av
    for g in model.parameters():
        g.grad.data.mul_(av)


if __name__ == '__main__':
    args = parse_args()
    # Trainset = DAVIS(DAVIS_ROOT, imset='2016/train.txt')
    # Trainloader = data.DataLoader(Trainset, batch_size=1, shuffle=True, num_workers=2)
    #
    # Testset = DAVIS(DAVIS_ROOT, imset='2016/val.txt')
    # Testloader = data.DataLoader(Testset, batch_size=1, shuffle=True, num_workers=2)

    # Define dataset...
    Trainset = DAVIS(DAVIS_ROOT, imset='2016/train.txt')
    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        Trainset, num_replicas=hvd.size(), rank=hvd.rank())
    Trainloader = torch.utils.data.DataLoader(Trainset, batch_size=args.bs, sampler=train_sampler)

    # Define dataset...
    Testset = DAVIS(DAVIS_ROOT, imset='2016/val.txt')
    # Partition dataset among workers using DistributedSampler
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        Testset, num_replicas=hvd.size(), rank=hvd.rank())
    Testloader = torch.utils.data.DataLoader(Testset, batch_size=args.bs, sampler=test_sampler)

    model = RGMP()
    if torch.cuda.is_available():
        model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 就这一行
        model = nn.DataParallel(model)

    writer = SummaryWriter(tensorboard_log)
    start_epoch = 0

    # load saved model if specified
    if args.loadepoch == "latest" or int(args.loadepoch) >= 0:
        print('Loading checkpoint {}@Epoch {}{}...'.format(font.BOLD, args.loadepoch, font.END))
        load_name = os.path.join(args.output_dir,
                                 '{}.pth'.format(args.loadepoch))
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

    # params
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            params += [{'params': [value], 'lr': args.lr, 'weight_decay': 4e-5}]

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)

    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    iters_per_epoch = len(Trainloader)
    for epoch in range(start_epoch, args.num_epochs):
        starttime = time.time()
        if epoch % args.eval_epoch == 1:
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
                    ms = model.module.Encoder(msv_F1, msv_P1)[0]

                    for f in range(0, all_M.shape[2] - 1):
                        output, ms = Propagate_MS(ms, model, all_F[:, :, f + 1], all_E[:, 0, f])
                        all_E[:, 0, f + 1] = output.detach()
                        # loss = loss + criterion(output.permute(1, 2, 0), all_M[:, 0, f + 1].float()) / all_M.size(2)
                        loss = loss + criterion(output, all_M[:, 0, f + 1].float()) / all_M.size(2)
                    iOU = iOU + iou(torch.cat((1 - all_E, all_E), dim=1), all_M)

                pbar.close()

                loss = loss / len(Testloader)
                iOU = iOU / len(Testloader)
                writer.add_scalar('Val/BCE', loss, epoch)
                writer.add_scalar('Val/IOU', iOU, epoch)
                print('loss: {}'.format(loss))
                print('IoU: {}'.format(iOU))

                video_thread = threading.Thread(target=log_mask,
                                                args=(all_F, all_E, info, writer, 'Val/frames', 'Val/masks'))
                video_thread.start()

        # training
        model.train()
        print('[Train] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
        for i, (all_F, all_M, info) in enumerate(Trainloader):
            optimizer.zero_grad()
            all_F, all_M = all_F[0], all_M[0]
            seq_name = info['name'][0]
            num_frames = info['num_frames'][0]
            num_objects = info['num_objects'][0]

            if (args.bptt_len < num_frames):
                start_frame = random.randint(0, num_frames - args.bptt_len)  # 随机从某一帧开始
                all_F = all_F[:, :, start_frame: start_frame + args.bptt_len]
                all_M = all_M[:, :, start_frame: start_frame + args.bptt_len]


            B, C, T, H, W = all_M.shape
            all_E = torch.zeros(B, C, T, H, W)
            all_E[:, 0, 0] = all_M[:, :, 0]

            msv_F1, msv_P1, all_M = ToCudaVariable([all_F[:, :, 0], all_E[:, 0, 0], all_M])  # msv是将加载的数据进行处理后用于程序中使用的意思
            ms = model.module.Encoder(msv_F1, msv_P1)[0]

            num_bptt = all_M.shape[2]
            loss = 0
            counter = 0
            for f in range(0, num_bptt - 1):
                output, ms = Propagate_MS(ms, model, all_F[:, :, f + 1], all_E[:, 0,
                                                                         f])  # training with recurrence 依次输入:参考帧feature maps,model,第f+1帧的原图,第f帧的mask
                all_E[:, 0, f + 1] = output.detach()
                # loss = loss + criterion(output.permute(1, 2, 0), all_M[:, 0, f + 1].float())
                loss = loss + criterion(output, all_M[:, 0, f + 1].float())
                counter += 1
                if (f + 1) % args.bptt_step == 0:
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    output.detach()
                    if f < num_bptt - 2:
                        loss = 0
                        counter = 0
            if loss > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # logging and display
            if (i + 1) % args.disp_interval == 0:
                writer.add_scalar('Train/BCE', loss / counter, i + epoch * iters_per_epoch)
                writer.add_scalar('Train/IOU', iou(torch.cat((1 - all_E, all_E), dim=1), all_M),
                                  i + epoch * iters_per_epoch)
                print('loss: {}'.format(loss / counter))

            if epoch % 10 == 1 and i == 0:
                video_thread = threading.Thread(target=log_mask, args=(all_F, all_E, info, writer))
                video_thread.start()

        if epoch % 10 == 0 and epoch > 0:
            save_name = '{}/{}.pth'.format(MODEL_DIR, epoch)
            latest_name = '{}/latest.pth'.format(MODEL_DIR)
            file_dir = os.path.split(save_name)[0]
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)  # 使用horovod多线程时候,必须自己创建,否则创建两遍文件夹,系统会报错
            if not os.path.exists(save_name):
                os.system(r'touch %s' % save_name)
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        },
                       save_name)
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        },
                       latest_name)
        endtime = time.time()
        dtime = endtime - starttime

        print("epoch%d运行时间：%.8s s" % (epoch, dtime))
