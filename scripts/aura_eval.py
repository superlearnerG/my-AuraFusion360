import sys
sys.path.append(".")
from pathlib import Path
import glob
from natsort import natsorted
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.image_utils import masked_psnr
from tqdm import tqdm
import os
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[2] / "pretrained_models" / "torch"))
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
import torch
import numpy as np
from pytorch_fid import fid_score

spatial_lpips = lpips.LPIPS(net='vgg', spatial=True).cuda()
original_lpips = lpips.LPIPS(net='vgg').cuda()

scenes = ["carton", "cone", "cookie", "newcone", "plant", "skateboard", "sunflower"]

for scene in scenes:
    # gt & object_masks in testing views
    gt_imgs = natsorted(glob.glob(f"./data/360-USID/{scene}/test_images/*jpg"))
    object_masks = natsorted(glob.glob(f"output/360-USID/{scene}/test/ours_30000/object_mask/*.png"))
    
    # methods
    ours_imgs = natsorted(glob.glob(f"output/360-USID/{scene}/test/ours_10000_object_inpaint/renders/*png"))
    
    methods = [ours_imgs]
    method_names = ['ours']

    
    print("Scene: ", scene)
    with torch.no_grad():
        for j in range(len(methods)):
            method = methods[j]
            method_name = method_names[j]
            if not (len(gt_imgs) == len(object_masks) == len(method)):
                print(f"{len(gt_imgs)}, {len(object_masks)}, {len(method)} are not equal")
                breakpoint()
                continue
            
            ssims = []
            psnrs = []
            lpipss = []
            fids = []
            ssims_object = []
            psnrs_object = []
            lpipss_object = []
            for i in range(len(gt_imgs)):
                # prepocess gt, img, object_mask
                gt_img = Image.open(gt_imgs[i]).convert("RGB")
                img_name = img_name = os.path.basename(gt_imgs[i])
                object_mask = Image.open(object_masks[i])
                gt_tensor = tf.to_tensor(gt_img).unsqueeze(0)[:, :3, :, :].cuda()
                object_mask_tensor = tf.to_tensor(object_mask).unsqueeze(0)[:, :3, :, :].cuda()
                object_mask_tensor[object_mask_tensor > 0.5] = 1
                img = Image.open(method[i]).convert("RGB")
                img_tensor = tf.to_tensor(img).unsqueeze(0)[:, :3, :, :].cuda()
                
                ###### Eval w/ object_mask
                # psnr w/ object_mask
                psnrs_object.append(masked_psnr(img_tensor, gt_tensor, object_mask_tensor).item())
                # lpips w/ object_mask
                lpips_map = spatial_lpips(img_tensor, gt_tensor)
                lpips_score = torch.sum(lpips_map * object_mask_tensor) / torch.sum(object_mask_tensor)
                lpipss_object.append(lpips_score.item())
                # ssim w/ object_mask
                img_tensor_object = img_tensor * object_mask_tensor.repeat(1, 3, 1, 1)
                gt_tensor_object = gt_tensor * object_mask_tensor.repeat(1, 3, 1, 1)
                ssims_object.append(ssim(img_tensor_object, gt_tensor_object).item())
                    
                ##### Eval w/o object_mask
                ssims.append(ssim(img_tensor, gt_tensor).item())
                psnrs.append(psnr(img_tensor, gt_tensor).item())
                lpipss.append(original_lpips(img_tensor, gt_tensor).item())
                
                
            # fid = fid_score.calculate_fid_given_paths([os.path.dirname(gt_imgs[0]), os.path.dirname(method[0])], 50, 'cuda', 2048, 0)
            original_stdout = sys.stdout
            try:
                sys.stdout = open(os.devnull, 'w')
                fid = fid_score.calculate_fid_given_paths([os.path.dirname(gt_imgs[0]), os.path.dirname(method[0])], 50, 'cuda', 2048, 0)
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout
            
            
            print("Method Name: ", method_name)
            print(f"PSNR: {np.mean(psnrs):.4f}, SSIM: {np.mean(ssims):.4f}, LPIPS: {np.mean(lpipss):.4f}, FID: {fid:.4f}")
            print(f"PSNR OBJECT: {np.mean(psnrs_object):.4f}, SSIM OBJECT: {np.mean(ssims_object):.4f}, LPIPS OBJECT: {np.mean(lpipss_object):.4f}")

            print(f"{np.mean(psnrs):.3f}, {np.mean(ssims):.3f}, {np.mean(lpipss):.3f}, {fid:.3f}")
            print(f"{np.mean(psnrs_object):.3f}, {np.mean(ssims_object):.3f}, {np.mean(lpipss_object):.3f}")
            
            print("====================================")
            
