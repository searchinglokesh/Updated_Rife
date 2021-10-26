import math
import time
import argparse

from model.FastRIFE_GF import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from AverageMeter import *

from dataset import VimeoDataset


def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
    return 5e-4 * mul


def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


def train(model, local_rank):
    log_path = 'model_GF'

    step = 0
    nr_eval = 0
    dataset = VimeoDataset('train')
    sampler = None
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True,
                            sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    print(train_data.__len__())
    dataset_val = VimeoDataset('validation')
    val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=8)
    org_model = 'model_GF'
    evaluate(model, val_data, step, local_rank)
    model.load_model(org_model)
    model.save_model(log_path, local_rank)
    print('Loaded model from', org_model)
    print('TRAINING...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, flow_gt = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            flow_gt = flow_gt.to(device, non_blocking=True)
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            mul = np.cos(step / (args.epoch * args.step_per_epoch) * math.pi) * 0.5 + 0.5
            learning_rate = get_learning_rate(step)
            pred, merged_img, flow, loss_l1_temp, loss_flow, loss_ter, flow_mask = model.update(imgs, gt, learning_rate, mul,
                                                                                           True, flow_gt)
            for p in model.contextnet.parameters():
                if p.requires_grad is False:
                    print(p)
            for p in model.fusionnet.parameters():
                if p.requires_grad is False:
                    print(p)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if local_rank == 0 and i % 100 == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1_temp:{:.4e}'.format(epoch, i, args.step_per_epoch,
                                                                                data_time_interval, train_time_interval,
                                                                                loss_l1_temp))
            step += 1
        nr_eval += 1
        val_loss = evaluate(model, val_data, step, local_rank)
        model.save_model(log_path, local_rank)
        model.schedulerG.step(val_loss)


def evaluate(model, val_data, nr_eval, local_rank, writer_val=None):
    loss_l1_list = AverageMeter()
    time_stamp = time.time()
    for i, data in enumerate(val_data):
        data_gpu, flow_gt = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]

        with torch.no_grad():
            pred, merged_img, flow, loss_l1, loss_flow, loss_ter, flow_mask = model.update(imgs, gt, training=False)

        loss_l1_list.update(loss_l1.cpu().numpy().mean())

    eval_time_interval = time.time() - time_stamp
    if local_rank == 0:
        print('VAL eval time: {}, loss_l1:{:.4e}'.format(eval_time_interval, loss_l1_list.avg))
    return loss_l1_list.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='slomo')
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=32, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank, training=True)
    train(model, args.local_rank)

