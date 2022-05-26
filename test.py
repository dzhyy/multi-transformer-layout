
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from script.option import get_testing_args
from script.rawdata_load import load_specific_raw_data
from script.misc import RenderMode
from script.result_draw import get_result_print
from script.dataloader import LayoutDataset
from script.model import MULTModel
from utils.draw import Painter

print(f'gpu is availabel:{torch.cuda.is_available()}')
args = get_testing_args()
device = torch.device('cpu') if args.cpu else torch.device('cuda')
xmlfiles = ['fashion_0039.xml','food_0601.xml','travel_0351.xml']
model_path = './experiment/model.epoch_50_f.pth'
result_savepath = './experiment/eval_log'
log_painter = Painter(result_savepath, mode=RenderMode.IMAGE)

raw_data = []
for xmlfile in xmlfiles: 
    data = load_specific_raw_data(args,xmlfile)
    if data == 'break_file':
        continue
    else:
        raw_data.append(data)
eval_dataset = LayoutDataset(args, raw_data, device)
eval_dataloader = DataLoader(eval_dataset,shuffle=False,batch_size=1,collate_fn=eval_dataset.collate_fn)
model = MULTModel(args)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
for batch in tqdm(eval_dataloader, desc='infer'):
    with torch.no_grad():
        img, bbox, label, target= batch.img.to(device), batch.bbox.to(device), batch.label.to(device), batch.bbox_trg.to(device)
        pad_mask, mask = batch.pad_mask.to(device), batch.mask.to(device)
        output, _ = model(img, bbox, label, pad_mask, mask)
        get_result_print(batch, output, [1, 1], log_painter, args.input_size)
