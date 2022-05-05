
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import language_model
from utils import option
from utils.draw import LogPainter
from script.rawdata_load import load_specific_raw_data
from script.misc import RenderMode
from train2 import get_result_print

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

args = option.get_trainning_args()
xmlfiles = ['fashion_0039.xml','food_0601.xml','travel_0351.xml']
model_path = './experiment/buffer/model.tmp.pth'
log_painter = LogPainter(args, mode=RenderMode.IMAGE)

raw_data = []
for xmlfile in xmlfiles: 
    data = load_specific_raw_data(args,xmlfile)
    if data == 'break_file':
        continue
    else:
        raw_data.append(data)
device = torch.device('cpu') if args.cpu is True else torch.device('cuda:0')
eval_dataset = LayoutDataset(args, raw_data, device)
eval_dataloader = DataLoader(eval_dataset,shuffle=False,batch_size=1,collate_fn=eval_dataset.collate_fn)
args.src_vocab = eval_dataset.layout_processor.vocab_size
args.tgt_vocab = eval_dataset.layout_processor.vocab_size
model = language_model.make_model(args)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
for batch in tqdm(eval_dataloader, desc='evaluating'):
    with torch.no_grad():
        output = model(batch)
        get_result_print(batch, output, [1, 1], log_painter, args)
