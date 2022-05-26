import os
import torch
import logging
import time
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
from tensorboardX import SummaryWriter
from script import option
from script.lr_scheduler import get_cosine_schedule_with_warmup
from script.dataloader import LayoutDataset,get_raw_data
from utils import logger, path
from utils.draw import Painter
from script.misc import RenderMode
from script.criterion import MutiLoss
from script.model import MULTModel
from script.result_draw import get_result_print


def main(args):
    torch.set_printoptions(precision=2)
    
    def train(model, optimizer, criterion, loader):
        epoch_loss = 0
        model.train()
        net = nn.DataParallel(model)
        start_time = time.time()
        for i_batch, batch in enumerate(loader):
            optimizer.zero_grad()
            img, bbox, label, target= batch.img.to(device), batch.bbox.to(device), batch.label.to(device), batch.bbox_trg.to(device)
            pad_mask, mask = batch.pad_mask.to(device), batch.mask.to(device)
            output, _ = net(img, bbox, label, pad_mask, mask)
            loss = criterion(output, target, batch.n_tokens.to(device),  pad_mask)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            epoch_loss += loss.item()
            if i_batch % args.log_interval == 0 and i_batch > 0:
                avg_loss = epoch_loss / (i_batch+1)
                elapsed_time = time.time() - start_time
                print('Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(i_batch, len(loader), elapsed_time * 1000 / args.log_interval, avg_loss))
                start_time = time.time()
        epoch_avg_loss = epoch_loss / len(train_dataloader)
        return epoch_avg_loss

    def evaluate(model, criterion, loader, log_painter):
        epoch_loss = 0
        num_log_samples = 3
        log_iter = len(loader) // num_log_samples

        model.eval()
        net = nn.DataParallel(model)
        for i_batch, batch in enumerate(loader):
            with torch.no_grad():
                img, bbox, label, target = batch.img.to(device), batch.bbox.to(device), batch.label.to(device), batch.bbox_trg.to(device)
                pad_mask, mask = batch.pad_mask.to(device), batch.mask.to(device)
                output, _ = net(img, bbox, label, pad_mask, mask)
                loss = criterion(output, target, batch.n_tokens.to(device),  pad_mask)

                if i_batch%log_iter == 0:
                    get_result_print(batch, output, [epoch, args.n_epochs], log_painter, args.input_size)

                epoch_loss += loss.item()
                
        epoch_avg_loss = epoch_loss / len(eval_dataloader)
        return epoch_avg_loss

    logger.set_logger(os.path.join(args.log_root,'train.log.txt'))
    device = torch.device('cpu') if args.cpu is True else torch.device('cuda')
    
    raw_data = get_raw_data(args,use_buffer=True)
    dataset = LayoutDataset(args, raw_data, device) # Num fo samples: 3860
    train_dataset, eval_dataset = random_split(dataset,[int(0.9*len(dataset)), len(dataset) - int(0.9*len(dataset))])
    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=args.batch_size,collate_fn=dataset.collate_fn)
    eval_dataloader = DataLoader(eval_dataset,shuffle=False,batch_size=args.batch_size,collate_fn=dataset.collate_fn)
    logging.info(f'Num fo samples:{len(dataset)},train samples:{len(train_dataset)},evaluat samples:{len(eval_dataset)}')
    logging.info(f'Device:{device}')

    model = MULTModel(args)
    logging.info(args)
    
    criterion = MutiLoss()

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.n_warmup_epochs, num_training_steps=args.n_epochs)

    logging.info('Start training.')
    model = model.to(device)
    path.clear_folder(os.path.join(args.log_root, "runs"))
    writer = SummaryWriter(comment='layout', log_dir=os.path.join(args.log_root, "runs"))
    log_painter = Painter(os.path.join(args.log_root,'eval_log'), mode=RenderMode.IMAGE)
    
    # early stop
    best_perform = float('inf')
    last_epoch = args.n_epochs
    stop_count = 0

    for epoch in range(1, args.n_epochs + 1):
        logging.info(f'\nepoch_{epoch}/{args.n_epochs}:')

        start = time.time()
        train_loss = train(model, optimizer, criterion, train_dataloader)
        eval_loss = evaluate(model, criterion, eval_dataloader, log_painter)
        end = time.time()

        logging.info('Epoch {:2d} | Time {:5.4f} sec | train Loss {:5.4f} | eval Loss {:5.4f}'.format(epoch, end-start, train_loss, eval_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)
        writer.add_scalars('loss', {'valid': eval_loss}, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)
        scheduler.step()

        # early stop
        if(eval_loss < best_perform):
            best_perform = eval_loss
            stop_count = 0
            # model save
            torch.save(model.state_dict(), os.path.join(args.log_root,f'model.epoch_{epoch}_p.pth'))
            path.remove_file(os.path.join(args.log_root,f'model.epoch_{epoch-1}_p.pth'))
        else:
            if (stop_count > 5):
                last_epoch = epoch
                break
            stop_count = stop_count+1

    torch.save(model.state_dict(), os.path.join(args.log_root,f'model.epoch_{last_epoch}_f.pth'))
    writer.close()



def cli_main():
    args = option.get_trainning_args()
    main(args)


if __name__=='__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 如果设置的设备超过3个，DP在汇中返回output的时候会产生问题
    cli_main()