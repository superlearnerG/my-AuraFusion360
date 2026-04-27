# Copyright 2024 Marigold authors, PRS ETH Zurich. All rights reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information and citation instructions are available on the
# Marigold project website: https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import bitsandbytes as bnb
import torch
from diffusers import MarigoldDepthPipeline
from diffusers.utils.torch_utils import randn_tensor

from arguments import OptimizationParams


def l1l2_loss(prediction, incomplete_depth):
    """
    Args:
        prediction (torch.Tensor): The predicted depth map. Shape: [1, H, W].
        incomplete_depth (torch.Tensor): The incomplete depth map. Shape: [1, H, W].
    """
    loss = torch.nn.functional.l1_loss(prediction, incomplete_depth) + torch.nn.functional.mse_loss(prediction, incomplete_depth)
    return loss


def l2_loss(prediction, incomplete_depth):
    """
    Args:
        prediction (torch.Tensor): The predicted depth map. Shape: [1, H, W].
        incomplete_depth (torch.Tensor): The incomplete depth map. Shape: [1, H, W].
    """
    loss = torch.nn.functional.mse_loss(prediction, incomplete_depth)

    return loss

def hubor_loss(prediction, incomplete_depth, delta=0.3):
    """
    Args:
        prediction (torch.Tensor): The predicted depth map. Shape: [1, H, W].
        incomplete_depth (torch.Tensor): The incomplete depth map. Shape: [1, H, W].
    """
    loss = torch.nn.functional.huber_loss(prediction, incomplete_depth, "mean", delta=delta)
    return loss

def adaptive_loss(prediction, incomplete_depth, unseen_mask, opt, greater_zero_mask=None):
    """
    Args:
        prediction (torch.Tensor): The predicted depth map. Shape: [1, H, W].
        incomplete_depth (torch.Tensor): The incomplete depth map. Shape: [1, H, W].
        unseen_mask (torch.Tensor): The mask of the unseen region. Shape: [1, H, W].
    """
    def dilate_mask(mask, iterations=1, kernel_size=3):
        # Dilate the mask
        dilated_mask = mask.clone()
        for _ in range(iterations):
            dilated_mask = torch.nn.functional.max_pool2d(dilated_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        return dilated_mask[0]
    
    unmask_dilated = dilate_mask(unseen_mask, iterations=opt.dilate_iter, kernel_size=opt.kernel_size)
    if greater_zero_mask is not None:
        visible_mask = (1 - unseen_mask) * greater_zero_mask.bool()
    else:
        visible_mask = (1 - unseen_mask)

    y_indices, x_indices = torch.where(unmask_dilated > 0.5)
    y_min, y_max = y_indices.min().item(), y_indices.max().item()
    x_min, x_max = x_indices.min().item(), x_indices.max().item()
    cropped_prediction = prediction[0, y_min:y_max, x_min:x_max]
    cropped_gt_depth = incomplete_depth[0, y_min:y_max, x_min:x_max]
    cropped_mask = visible_mask[0, y_min:y_max, x_min:x_max]
    loss = torch.nn.functional.huber_loss(cropped_prediction*cropped_mask, cropped_gt_depth*cropped_mask, "mean", delta=opt.delta)
                            
    return loss


def exponential_decay_list(init_weight, decay_rate, num_steps):
    weights = [init_weight * (decay_rate ** i) for i in range(num_steps)]
    return weights


class AGDDv2(MarigoldDepthPipeline):
    """
    Pipeline for Marigold depth inpainting with latent optimization.
    """
    def __call__(
        self, 
        image: torch.Tensor, 
        incomplete_depth: torch.Tensor = None, 
        unseen_mask: torch.Tensor = None,
        num_inference_steps: int = 50, 
        processing_resolution: int = 768,
        resample_method_input: str = "bilinear",
        resample_method_output: str = "bilinear", 
        generator: torch.Generator = None,
        latents: torch.Tensor = None,
        is_latent_optimizing: bool = True,
        opt: OptimizationParams = None,
        tb_writer=None,
    ) -> torch.Tensor:

        """
        Args:
            image (torch.Tensor): The inpaintedRGB image. Shape: [3, H, W].
            incomplete_depth (torch.Tensor): The incomplete depth map. Shape: [1, H, W].
            unseen_mask (torch.Tensor): The mask of the unseen region. Shape: [1, H, W].
            num_inference_steps (int, optional): The number of inference steps. Defaults to 50.
            processing_resolution (int, optional): The processing resolution. Defaults to 768.
            generator (torch.Generator, optional): The generator. Defaults to None.
            latents (torch.Tensor, optional): The latents. Defaults to None.
            is_latent_optimizing (bool, optional): Whether to optimize the latent. Defaults to True. If False, it's normal Marigold.
            opt (OptimizationParams, optional): The optimization parameters. Defaults to None.
        Returns:
            torch.Tensor: The inpainted depth map. Shape: [1, H, W].
        """
        
        if is_latent_optimizing:
            assert incomplete_depth is not None and unseen_mask is not None, "incomplete_depth and unseen_mask must be provided if is_latent_optimizing is True"
        
        # 0. Resolving variables.
        device = self._execution_device
        dtype = self.dtype
        ensemble_size = 1
        batch_size = 1
        
        # 1. Check inputs. (skip)
        # num_images = self.check_inputs(
        #     image,
        #     num_inference_steps,
        #     ensemble_size,
        #     processing_resolution,
        #     resample_method_input,
        #     resample_method_output,
        #     batch_size,
        #     ensembling_kwargs,
        #     latents,
        #     generator,
        #     output_type,
        #     output_uncertainty,
        # )
            
        
        with torch.no_grad():
            # 2. Prepare empty text conditioning.
            # Model invocation: self.tokenizer, self.text_encoder.
            if self.empty_text_embedding is None:
                prompt = ""
                text_inputs = self.tokenizer(
                    prompt,
                    padding="do_not_pad",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)
                self.empty_text_embedding = self.text_encoder(text_input_ids)[0]  # [1,2,1024]
        
            # 3. Preprocess input images. This function loads input image or images of compatible dimensions `(H, W)`,
            # optionally downsamples them to the `processing_resolution` `(PH, PW)`, where
            # `max(PH, PW) == processing_resolution`, and pads the dimensions to `(PPH, PPW)` such that these values are
            # divisible by the latent space downscaling factor (typically 8 in Stable Diffusion). The default value `None`
            # of `processing_resolution` resolves to the optimal value from the model config. It is a recommended mode of
            # operation and leads to the most reasonable results. Using the native image resolution or any other processing
            # resolution can lead to loss of either fine details or global context in the output predictions.
            image, padding, original_resolution = self.image_processor.preprocess(
                image, processing_resolution, resample_method_input, device=device, dtype=self.dtype
            )  # [N,3,PPH,PPW]
            
            # 4. Encode input image into latent space. At this step, each of the `N` input images is represented with `E`
            image_latent, pred_latent = self.prepare_latents(
                image, latents, generator, ensemble_size, batch_size
            )  # [N*E,4,h,w], [N*E,4,h,w]
        
        _, _, height, width = image_latent.shape
        del image
        
        
        # 4, prepare input
        # Because batch_size is 1, I directly use the image_latent and pred_latent as batch_image_latent and batch_pred_latent
        batch_image_latent = image_latent 
        batch_pred_latent = pred_latent
        batch_empty_text_embedding = self.empty_text_embedding.to(device=device, dtype=dtype).repeat(
            batch_size, 1, 1
        )  # [B,1024,2]
        text = batch_empty_text_embedding
        
        # 5. if it's for latent optimizing (agdd), prepare optimized parameters and optimizer
        optimizer, optimizer_scheduler = None, None
        if is_latent_optimizing:
            batch_pred_latent = torch.nn.Parameter(batch_pred_latent)            
            optimizer = bnb.optim.AdamW8bit([batch_pred_latent], lr=opt.agdd_lr, weight_decay=0.0)
            optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            
            greater_zero_mask = incomplete_depth > 0
            
            for p in self.unet.parameters():
                p.requires_grad_(False)
            for p in self.vae.parameters():
                p.requires_grad_(False)
        
        
        # Denoising process
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps, leave=False, desc="Diffusion steps...")): # default is 50
            if is_latent_optimizing and i >= 0:
                ##### Start optimizing for: t -> t - 1 #####
                for k in range(opt.optimize_iter): # default is 8
                    # Concat the image latent and the predicted latent
                    batch_latent = torch.cat([batch_image_latent, batch_pred_latent], dim=1)
                    
                    # Predict the noise
                    noise = self.unet(
                        batch_latent, t, encoder_hidden_states=text, return_dict=False
                    )[0]
                    
                    # Get the predicted latent at timestep 0 from timestep t
                    batch_pred_latent_t0 = self.scheduler.step(
                        noise, t, batch_pred_latent, generator=generator
                    ).pred_original_sample # [B,4,h,w]
                    
                    # Decode the predict latent to Depth
                    prediction = self.decode_prediction(batch_pred_latent_t0)
                    prediction = self.image_processor.unpad_image(prediction, padding)  # [N*E,1,PH,PW]
                    prediction = self.image_processor.resize_antialias(
                        prediction, original_resolution, resample_method_output, is_aa=False
                    )[0]  # [1,H,W]

                    # visualize the prediction
                    with torch.no_grad():
                        if i % 5 == 0:
                            vis = self.image_processor.visualize_depth(prediction.squeeze(), val_min=prediction.min(), val_max=prediction.max())[0]
                            vis.save(f"tmp/predecoded_depth_{i}.jpg")
                    
                    # Step4. Compute gradient of loss with respect to latent, and update the noise
                    if opt.not_adaptive:
                        loss = l2_loss((prediction*(1 - unseen_mask))[greater_zero_mask], (incomplete_depth*(1 - unseen_mask))[greater_zero_mask].detach())
                    else:
                        loss = adaptive_loss(prediction, incomplete_depth, unseen_mask, opt, greater_zero_mask) * opt.agdd_loss_scale
                    
                    
                    # Opt1. Directly update the latent
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    
                    if tb_writer is not None:
                        tb_writer.add_scalar('Loss/optimization', loss.item(), i * opt.optimize_iter + k)
                        tb_writer.add_scalar('Gradient/norm', batch_pred_latent.grad.norm().item(), i * opt.optimize_iter + k)
                    
                    print(
                        f"loss {loss.item():.5f} | "
                        f"∥grad∥ {batch_pred_latent.grad.norm().item():.4e}"
                    )
                ##### End optimizing for: t -> t - 1 #####
                # step to t - 1
                with torch.no_grad():
                    batch_latent = torch.cat([batch_image_latent, batch_pred_latent], dim=1)  # [B,8,h,w]
                    noise = self.unet(batch_latent, t, encoder_hidden_states=text, return_dict=False)[0]  # [B,4,h,w]
                    batch_pred_latent.data.copy_(
                        self.scheduler.step(noise, t, batch_pred_latent).prev_sample
                    )

                    if opt.use_renoise and i >= opt.renoise_start_iter and i < num_inference_steps - 1 and i % opt.renoise_step == 0:
                        latents_to_average = []
                        current_batch_pred_latent = batch_pred_latent.detach().clone() # timestep t - 1, renoise iter: m - 1
                        for m in range(opt.infer_iter): # IMPORTANT NOTE: if reference image has more details, set infer_iter larger, which use noise to adujust the distribution shift caused by latent optimization.
                            # 1. add noise
                            noise = randn_tensor(
                                current_batch_pred_latent.shape, 
                                generator=generator, 
                                dtype=current_batch_pred_latent.dtype
                            ).to(device)
                            
                            alpha_bar_t     = self.scheduler.alphas_cumprod[t]
                            prev_t = self.scheduler.timesteps[i + 1]
                            alpha_bar_prev  = self.scheduler.alphas_cumprod[prev_t]
                            alpha = alpha_bar_t / alpha_bar_prev
                            noisy_batch_pred_latent = (alpha.sqrt() * current_batch_pred_latent) + ((1 - alpha).sqrt() * noise)
                            
                            # 2. denoise
                            batch_latent = torch.cat([batch_image_latent, noisy_batch_pred_latent], dim=1)  # [B,8,h,w]
                            predicted_noise = self.unet(batch_latent, t, encoder_hidden_states=text, return_dict=False)[0]  # [B,4,h,w] 
                            current_batch_pred_latent_m = self.scheduler.step(
                                predicted_noise, t, current_batch_pred_latent, generator=generator
                            ).prev_sample # timestep t - 1, renoise iter: m
                            latents_to_average.append(current_batch_pred_latent_m)
                            
                        batch_pred_latent.data.copy_(
                            torch.mean(torch.stack(latents_to_average), dim=0)
                        )
                    
                if optimizer_scheduler is not None:
                    optimizer_scheduler.step()
                if i % 5 == 0:
                    print(f"lr: {optimizer.param_groups[0]['lr']:.4f}")
                    
            else:                
                # w/o latent optimizing
                with torch.no_grad():
                    batch_latent = torch.cat([batch_image_latent, batch_pred_latent], dim=1)
                    noise = self.unet(
                        batch_latent, t, encoder_hidden_states=text, return_dict=False
                    )[0]  # [1,4,h,w]
                
                    batch_pred_latent.data = self.scheduler.step(
                        noise, t, batch_pred_latent, generator=generator
                    ).prev_sample # [B,4,h,w] 
            

        # finalize
        pred_latent = batch_pred_latent
        
        del (
            image_latent,
            batch_empty_text_embedding,
            batch_image_latent,
            batch_pred_latent,
            text,
            batch_latent,
            noise,
        )     
        
        ##### Decoding Latent to Pixel Space #####
        with torch.no_grad():
            # 6. Decode predictions from latent into pixel space. The resulting `N * E` predictions have shape `(PPH, PPW)`,
            # which requires slight postprocessing. Decoding into pixel space happens in batches of size `batch_size`.
            # Model invocation: self.vae.decoder.
            prediction = self.decode_prediction(pred_latent)
            
            # 7. Remove padding. The output shape is (PH, PW).
            prediction = self.image_processor.unpad_image(prediction, padding)  # [N*E,1,PH,PW]
        
            prediction = self.image_processor.resize_antialias(
                prediction, original_resolution, resample_method_output, is_aa=False
            )
            
            # 11. Offload all models
            self.maybe_free_model_hooks()
            
        # visualize the inpainted depth map
        vis = self.image_processor.visualize_depth(prediction.squeeze(), val_min=prediction.squeeze().min(), val_max=prediction.squeeze().max())[0]
        import os
        os.makedirs("tmp", exist_ok=True)
        vis.save("tmp/depth_vis.jpg")
        
        # return torch.Tensor [1, H, W]
        return prediction.squeeze(0)
    
        
            
            
            
            
            