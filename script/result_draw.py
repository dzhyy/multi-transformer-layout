
import torch
import logging
from script.layout_process import box_cxcywh_to_xyxy,scale

def get_result_print(batch, pred, step_info, painter, size):
    pred = pred.cpu()
    with torch.no_grad():
        # filter for PAD
        mask = batch.pad_mask[0].squeeze(0).unsqueeze(-1).repeat(1,4) # [bn,1,len]-> [len,4]
        pred = torch.masked_select(pred[0],mask).reshape(-1,4)
        target = torch.masked_select(batch.bbox_trg[0],mask).reshape(-1,4)
        # scale&format back
        pred1 = box_cxcywh_to_xyxy(pred).cpu().numpy().tolist()
        bboxes = [scale(bbox,(1,1), size) for bbox in pred1]
        base_framework = batch.framework[0]
        framework = {}
        framework['bboxes'] = bboxes
        framework.update({k:v for k,v in base_framework.items() if k not in framework})
        logging.info(f'epoch_{step_info[0]}/{step_info[1]}:')
        logging.info(f"framework_name: {framework['name']}")
        logging.info(f"framework_labels: {framework['labels']}")
        logging.info(f'decoder_output_label: {target.cpu().numpy().tolist()}')
        logging.info(f'decoder_output_pred: {pred.cpu().numpy().tolist()}')
        painter.draw(framework, base_framework, f'epoch_{step_info[0]}_')
