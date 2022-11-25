import sys
sys.path.append("..")
import yaml, easydict, pprint, os, cv2, torch, torchvision.transforms as transforms
from models.model_helper import build_network
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.dist_helper import setup_distributed
from utils.misc_helper import load_state, to_device
from utils.vis_helper import Visualizer

FILE_DIR    = "../data/FSC147_384_V2/images_384_VarV2"
FILE_NAME   = "5.jpg"
BOXES       = [[99, 511, 112, 520], [126, 310, 136, 316], [142, 237, 159, 250]]

FSC147  = "../experiments/FSC147"
CONFIG  = os.path.join(FSC147, "config.yaml")
WEIGHTS = "./" + os.path.join(FSC147, "checkpoints", "ckpt_best.pth.tar")
VERBOSE = False


def get_model(config):
    model      = build_network(config.net)
    model.cuda()
    local_rank = int(os.environ["LOCAL_RANK"])
    model      = DDP(model,
                    device_ids                = [local_rank],
                    output_device             = local_rank,
                    find_unused_parameters    = True)
    load_state(WEIGHTS, model)
    return model
    
def resize(image, boxes, size_rsz: tuple, size_orig: tuple):
    # adapted from
    # https://github.com/zhiyuanyou/SAFECount/blob/7069b49ef7302621c719706a841d644d25821174/datasets/base_dataset.py#L44
    h_orig, w_orig      = size_orig
    h_rsz, w_rsz        = size_rsz
    h_scale, w_scale    = h_rsz / h_orig, w_rsz / w_orig
    image               = cv2.resize(image, (w_rsz, h_rsz))
    boxes_rsz           = []
    for box in boxes:
        y_tl, x_tl, y_br, x_br = box
        y_tl = int(y_tl * h_scale)
        y_br = int(y_br * h_scale)
        x_tl = int(x_tl * w_scale)
        x_br = int(x_br * w_scale)
        boxes_rsz.append([y_tl, x_tl, y_br, x_br])
    return image, boxes_rsz

def load_sample(image_path: str, boxes: tuple, config):
    normalize_fn  = transforms.Normalize(mean = config["pixel_mean"], std = config["pixel_std"])
    image         = cv2.imread(image_path)
    image         = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    image, boxes  = resize(image, boxes, config["input_size"], (height, width))
    image         = transforms.ToTensor()(image)
    boxes         = torch.tensor(boxes, dtype = torch.float64)
    image         = normalize_fn(image)
    sample        = {"image": image.unsqueeze(0), 
                     "boxes": boxes.unsqueeze(0),
                     "height": height,
                     "width": width}
    sample        = to_device(sample, device = torch.device("cuda"))
    return sample
    

if __name__ == "__main__":
    with open(CONFIG) as f:
        config = easydict.EasyDict(yaml.load(f, Loader = yaml.FullLoader))
    if VERBOSE:
        print(pprint.pformat(config))
    
    config.port         = config.get("port", None)
    rank, world_size    = setup_distributed(port = config.port)    
    model               = get_model(config)
    sample              = load_sample(os.path.join(FILE_DIR, FILE_NAME), BOXES, config.dataset)
    
    outputs             = model(sample)
    density_pred        = outputs["density_pred"]
    pred_cnt            = torch.sum(density_pred).item()
    print("Predicted count:", pred_cnt)
    visualizer          = Visualizer("./", FILE_DIR, "sigmoid", True, True)
    visualizer.vis_result(FILE_NAME, "out.png", sample["height"], sample["width"], density_pred[0])
