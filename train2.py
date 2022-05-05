import os
import torch
import logging
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tensorboardX import SummaryWriter
from script.lr_scheduler import get_cosine_schedule_with_warmup
from script.dataloader import LayoutDataset,get_raw_data
from utils import logger, option, path
from script.layout_process import box_cxcywh_to_xyxy,scale
from script.model import MULTModel
import time
import torch.nn as nn
'''
这个数据集（任务）和之前的不同之处：
图片需要编码
文字没有字数
'''

def get_result_print(batch, pred, step_info, painter,args):
    with torch.no_grad():
        # filter for PAD
        mask = batch.seq_mask[0].unsqueeze(-1).repeat(1,4)
        pred = torch.masked_select(pred[0],mask).reshape(-1,4)
        target = torch.masked_select(batch.bbox_trg[0],mask).reshape(-1,4)
        # scale&format back
        pred1 = box_cxcywh_to_xyxy(pred).cpu().numpy().tolist()
        bboxes = [scale(bbox,(1,1),args.input_size) for bbox in pred1]
        '''test0 = box_cxcywh_to_xyxy(batch.bbox_trg[0]).cpu().numpy().tolist()
        test = [scale(bbox,(1,1),args.input_size) for bbox in test0]'''
        base_framework = batch.framework[0]
        framework = {}
        framework['bboxes'] = bboxes
        framework.update({k:v for k,v in base_framework.items() if k not in framework})
        logging.info(f'epoch_{step_info[0]}/{step_info[1]}:')
        logging.info(f"framework_name: {framework['name']}")
        logging.info(f"framework_labels: {framework['labels']}")
        logging.info(f'decoder_output_label: {target.cpu().numpy().tolist()}')
        logging.info(f'decoder_output_pred: {pred.cpu().numpy().tolist()}')
        painter.log(framework, base_framework, f'epoch_{step_info[0]}_')

def main(args):

    def train(model, optimizer, criterion, loader):
        epoch_loss = 0
        model.train()
        net = nn.DataParallel(model)
        start_time = time.time()
        for i_batch, batch in enumerate(loader):
            optimizer.zero_grad()
            output = net(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            epoch_loss += loss.item()
            if i_batch % args.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(i_batch, len(loader), elapsed_time * 1000 / args.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
        epoch_avg_loss = epoch_loss / len(train_dataloader)
        return epoch_avg_loss

    def evaluate(model, criterion, loader):
        eval_losses = 0
        model.eval()
        net = nn.DataParallel(model)
        for i_batch, batch in enumerate(loader):
            with torch.no_grad():
                output = net(batch)
                loss = criterion(output, batch.bbox_trg, batch.n_tokens, batch.seq_mask)      
                eval_losses += loss.item()
        epoch_avg_loss = eval_losses / len(eval_dataloader)
        return epoch_avg_loss


    logger.set_logger(os.path.join(args.log_root,'train.log.txt'))
    device = torch.device('cpu') if args.cpu is True else torch.device('cuda:0')
    
    raw_data = get_raw_data(args,use_buffer=True)
    dataset = LayoutDataset(args, raw_data, device) # Num fo samples: 3860
    train_dataset, eval_dataset = random_split(dataset,[int(0.9*len(dataset)), len(dataset) - int(0.9*len(dataset))])
    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=args.batch_size,collate_fn=dataset.collate_fn)
    eval_dataloader = DataLoader(eval_dataset,shuffle=False,batch_size=args.batch_size,collate_fn=dataset.collate_fn)
    logging.info(f'Num fo samples:{len(dataset)},train samples:{len(train_dataset)},evaluat samples:{len(eval_dataset)}')
    logging.info(f'Device:{device}')

    model = MULTModel(args)
    logging.info(args)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.PAD,)
    # criterion = MutiLoss()

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.n_warmup_epochs, num_training_steps=args.n_epochs)

    logging.info('Start training.')
    model = model.to(device)
    path.clear_folder(os.path.join(args.log_root, "runs"))
    writer = SummaryWriter(comment='layout', log_dir=os.path.join(args.log_root, "runs"))
    # early stop
    best_perform = float('inf')
    last_epoch = args.n_epochs
    stop_count = 0

    for epoch in range(1, args.n_epochs + 1):
        logging.info(f'\nepoch_{epoch}/{args.n_epochs}:')

        start = time.time()
        train_loss = train(model, optimizer, criterion, train_dataloader)
        eval_loss = evaluate(model, criterion, eval_dataloader)
        end = time.time()

        logging.info('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Train Loss {:5.4f}'.format(epoch, end-start, train_loss, eval_loss))
        # logging.info(f'Train loss: {train_epoch_loss}, Eval loss: {eval_epoch_loss}')
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
    # TODO:使用此函数确定是否使用DDP进行训练
    args = option.get_trainning_args()
    main(args)


if __name__=='__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    cli_main()