from utils.draw import Painter,RenderMode
from script.dataloader import get_raw_data
from script.rawdata_load import load_specific_raw_data
from script.option import get_preprocess_args 

def compose_raw_data(save_folder, filename_list = None):
    args = get_preprocess_args()
    if filename_list is not None:
        raw_data = load_specific_raw_data(filename_list)
    else:
        raw_data = get_raw_data(args,use_buffer=True)
    painter = Painter(save_folder, mode=RenderMode.IMAGE)
    for framework in raw_data:
        painter.draw(framework)


if __name__=='__main__':
    save_folder = './lab/'
    compose_raw_data(save_folder)