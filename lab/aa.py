import os
import torch
import logging
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tensorboardX import SummaryWriter
from script.lr_scheduler import get_cosine_schedule_with_warmup
from script.dataloader import LayoutDataset
from model import transformer
from utils import logger, option, path
from utils.draw import LogPainter
from script.misc import RenderMode
args = option.get_trainning_args()
logger.set_logger(os.path.join(args.log_root,'train.log.txt'))
device = 'cpu' if args.cpu is True else 'cuda:0'

dataset = LayoutDataset(args,use_buffer=True) # Num fo samples: 3860
train_dataset, eval_dataset = random_split(dataset,[int(0.9*len(dataset)), len(dataset) - int(0.9*len(dataset))])
train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=args.batch_size,collate_fn=dataset.collate_fn)
eval_dataloader = DataLoader(eval_dataset,shuffle=False,batch_size=args.batch_size,collate_fn=dataset.collate_fn)
logging.info(f'Num fo samples:{len(dataset)},train samples:{len(train_dataset)},evaluat samples:{len(eval_dataset)}')
logging.info(f'Device:{device}')

args.src_vocab = dataset.layout_processor.vocab_size
args.tgt_vocab = dataset.layout_processor.vocab_size
model = transformer.make_model(args)
logging.info(args)

criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.PAD,) # TODO 虽然之前iou不可行，但是linear loss应该可行
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.n_warmup_epochs, num_training_steps=args.n_epochs)

logging.info('Start training.')
model = model.to(device)
path.clear_folder(os.path.join(args.log_root, "runs"))
writer = SummaryWriter(comment='layout', log_dir=os.path.join(args.log_root, "runs"))
log_painter = LogPainter(args, dataset.layout_processor,mode=RenderMode.IMAGE)
# early stop
best_perform = float('inf')
last_epoch = 0
stop_count = 0

for epoch in range(1, args.n_epochs + 1):
    logging.info(f'\nepoch_{epoch}/{args.n_epochs}:')

    # model train
    model.train()
    train_losses = 0
    for batch in tqdm(train_dataloader, desc='training'):
        logging.info(f'epoch_:')
        logging.info(f"framework_name: {batch.frameworks[0]['name']}")
        logging.info(f"framework_labels: {batch.frameworks[0]['labels']}")
        logging.info(f'src: {batch.src[0].cpu().numpy().tolist()}')
        logging.info(f'decoder_output_label: {batch.trg_y[0].cpu().numpy().tolist()}')
        log_painter.log(batch.frameworks[0], batch.trg_y[0].cpu().numpy().tolist(), f'epoch_')