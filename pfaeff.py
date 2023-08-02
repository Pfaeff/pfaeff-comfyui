import os
import torch
import nodes
from nodes import MAX_RESOLUTION
import cv2
import numpy as np
from PIL import Image
import tempfile
from collections import Counter

from diffusers import StableDiffusionInpaintPipeline
from diffusers.models import AutoencoderKL
import subprocess


class AstropulsePixelDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE", ),
                              "max_colors": ("INT", {"default": 128})
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    CATEGORY = "Pfaeff/image"    

    def __init__(self):
        current_script_dir = os.path.dirname(os.path.realpath(__file__))
        self.pixeldetector_path = os.path.join(current_script_dir, 'pixeldetector', 'pixeldetector.py')


    def run(self, image, max_colors):

        if image.shape[0] > 1:
            raise Exception("Batches are not supported since output images can have different resolutions!")

        numpy_image = (image[0, :, :, :].cpu().detach().numpy() * 255.0).astype(np.uint8)

        # Create a temporary file for the input image
        temp_input_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_input_path = temp_input_file.name
        Image.fromarray(numpy_image).save(temp_input_path)
        temp_input_file.close()

        # Create a temporary file for the output image
        temp_output_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_output_path = temp_output_file.name
        temp_output_file.close()

        # Construct the command for calling the pixeldetector script
        command = [
            "python", self.pixeldetector_path,
            "--input", temp_input_path,
            "--output", temp_output_path
        ]
        if max_colors:
            command.extend(["--max", str(max_colors)])
        
        command.extend(["--palette"]) # TODO make this a parameter

        # Run the command
        subprocess.run(command)

        # Read the processed image
        processed_image = Image.open(temp_output_path)
        processed_image = np.array(processed_image)
        # Close and delete the temporary files
        os.unlink(temp_input_path)
        os.unlink(temp_output_path)

        # Convert back to torch
        image_result = torch.from_numpy(processed_image).float() / 255.0
        image_result = image_result[None, :, :, :]
        image_result = image_result.to(image.device)

        return (image_result, )


class BackgroundRemover:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE", ),
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    CATEGORY = "Pfaeff/image"        

    def detect_most_common_color(self, image):
        # Reshape the image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Convert pixels to tuples so they can be hashed
        pixel_tuples = [tuple(pixel) for pixel in pixels]

        # Find the most common color using a Counter
        most_common_color = Counter(pixel_tuples).most_common(1)[0][0]
        
        return np.array(most_common_color)
    

    def run(self, image):
        batch_size, height, width, channels = image.size()

        if channels != 3:
            raise Exception("Input must be a 3-channel image")

        image_result = torch.zeros((batch_size, height, width, 4), dtype=torch.float32)

        for b in range(image.shape[0]):
            numpy_image = (image[b, :, :, :].cpu().detach().numpy() * 255.0).astype(np.uint8)

            # Find the most common color in the image
            most_common_color = self.detect_most_common_color(numpy_image)

            # Create an output image with an additional alpha channel (shape will be (height, width, 4))
            output_image = np.zeros((numpy_image.shape[0], numpy_image.shape[1], 4))

            # Iterate through the image and set the RGB channels to match the original image
            # If the pixel matches the most common color, set the alpha channel to 0; otherwise, set it to 255
            for i in range(numpy_image.shape[0]):
                for j in range(numpy_image.shape[1]):
                    if np.array_equal(numpy_image[i, j], most_common_color):
                        output_image[i, j, :3] = numpy_image[i, j]
                        output_image[i, j, 3] = 0
                    else:
                        output_image[i, j, :3] = numpy_image[i, j]
                        output_image[i, j, 3] = 255
        
            image_result[b, :, :, :] = torch.from_numpy(output_image).float() / 255.0
            image_result = image_result.to(image.device)

        return (image_result, )


class InpaintingPipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": ("STRING", {"default": "stabilityai/stable-diffusion-2-inpainting"}),
                              "vae_name": ("STRING", {"default": "stabilityai/sd-vae-ft-mse"}),
                             }}
    RETURN_TYPES = ("INPAINT_PIPELINE",)
    FUNCTION = "load_inpainting_pipeline"

    CATEGORY = "Pfaeff/loaders"

    def load_inpainting_pipeline(self, model_name, vae_name):
        vae = AutoencoderKL.from_pretrained(vae_name)
        inpaint = StableDiffusionInpaintPipeline.from_pretrained(
           model_name, vae=vae
        )

        if torch.cuda.is_available():
            inpaint.to("cuda")      

        return (inpaint,)
    

class Inpainting:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "inpainting_pipeline": ("INPAINT_PIPELINE", ),
                              "image": ("IMAGE", ),
                              "mask": ("MASK", ),
                              "text": ("STRING", {"multiline": True}),
                              "num_inference_steps": ("INT", {"default": 20}),
                              "guidance_scale": ("FLOAT", {"default": 8.0}),
                              }}    
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "inpaint"

    CATEGORY = "Pfaeff/inpainting"    

    def inpaint(self, inpainting_pipeline, image, mask, text, num_inference_steps, guidance_scale):

        extra_kwargs = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }

        image_result = torch.zeros_like(image)

        for i in range(image.shape[0]):

            numpy_image = (image[i, :, :, :].cpu().detach().numpy() * 255.0).astype(np.uint8)
            init_image = Image.fromarray(numpy_image)

            numpy_mask = (mask.cpu().detach().numpy() * 255.0).astype(np.uint8)
            mask_image = Image.fromarray(numpy_mask)

            inpainted = inpainting_pipeline(
                prompt=text,
                image=init_image,
                mask_image=mask_image,
                width=image.shape[2],
                height=image.shape[1],
                **extra_kwargs,
            )["images"]

            image_result[i, :, :, :] = torch.from_numpy(np.array(inpainted[0])).float() / 255.0

        image_result = image_result.to(image.device)            

        return (image_result,)
 

class ImagePadForBetterOutpaint:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "top": ("INT", {"default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "right": ("INT", {"default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "bottom": ("INT", {"default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "inpaint_radius": ("INT", {"default": 5, "min": 3, "max": 128, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    FUNCTION = "expand_image"

    CATEGORY = "Pfaeff/outpainting"

    def add_padding_and_create_mask(self, image, left, top, right, bottom):
        # Add padding around the image
        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        # Create a mask with same size as padded image
        h, w = padded_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Set the padded region in the mask to 255
        mask[:top, :] = 255
        mask[-bottom:, :] = 255
        mask[:, :left] = 255
        mask[:, -right:] = 255

        return padded_image, mask
    

    def expand_image(self, image, left, top, right, bottom, inpaint_radius):
        batch_size, height, width, channels = image.size()

        image_result = torch.zeros((batch_size, height + top + bottom, width + left + right, channels), dtype=torch.float32)
        mask_result = torch.zeros((height + top + bottom, width + left + right), dtype=torch.float32)

        for i in range(batch_size):
            img = (image[i, :, :, :].cpu().detach().numpy() * 255.0).astype(np.uint8)

            img, mask = self.add_padding_and_create_mask(img, left, top, right, bottom)

            inpainted = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_NS)

            image_result[i, :, :, :] = torch.from_numpy(inpainted).float() / 255.0

            if i == 0:
                mask_result[:, :] = torch.from_numpy(mask).float() / 255.0

        image_result = image_result.to(image.device)
        mask_result = mask_result.to(image.device)

        masked_image = torch.zeros_like(image_result)
        for i in range(image_result.shape[-1]):
            masked_image[:, :, :, i] = image_result[:, :, :, i] * (mask < 0.5)

        return (image_result, mask_result, masked_image)
    

NODE_CLASS_MAPPINGS = {
    "AstropulsePixelDetector": AstropulsePixelDetector,
    "BackgroundRemover": BackgroundRemover,
    "ImagePadForBetterOutpaint": ImagePadForBetterOutpaint,
    "InpaintingPipelineLoader": InpaintingPipelineLoader,
    "Inpainting": Inpainting
}