import uuid
import os
from io import BytesIO
import base64
import requests
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import numpy as np
import boto3


os.environ['SPCONV_ALGO'] = 'native'

class InferlessPythonModel:
    @staticmethod
    def download_image(url):
        response = requests.get(url)
        return Image.open(BytesIO(response.content)).convert("RGB")
    
    @staticmethod
    def convert_base64(file_name):
        with open(file_name, 'rb') as file:
            file_content = file.read()
        base64_encoded = base64.b64encode(file_content)
        base64_string = base64_encoded.decode('utf-8')
        os.remove(file_name)
        return base64_string

    def initialize(self):
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self.pipeline.cuda()

        aws_region = 'us-west-1'  # e.g., 'us-west-1'
        self.s3_client = boto3.client('s3', region_name=aws_region, aws_access_key_id=os.getenv("AWS_KEYS"), aws_secret_access_key=os.getenv("AWS_SECRETS") )

    def infer(self, inputs):
        image_url = inputs["image_url"]
        trial_id = inputs["task_id"]
        return_render = inputs["return_render"]
        seed =  int(inputs.get("seed",0))
        ss_guidance_strength =  float(inputs.get("ss_guidance_strength",7.5))
        ss_sampling_steps =  int(inputs.get("ss_sampling_steps",12))
        slat_guidance_strength =  float(inputs.get("slat_guidance_strength",3))
        slat_sampling_steps = int(inputs.get("slat_sampling_steps",12))
        glb_extraction_simplify = float(inputs.get("glb_extraction_simplify",0.0))
        glb_extraction_texture_size = int(inputs.get("glb_extraction_texture_size",1024))
        preprocess_image = bool(inputs.get("preprocess_image",False))

        image = InferlessPythonModel.download_image(image_url).resize((512, 512))
        # Run the pipeline
        outputs = self.pipeline.run(
            image,
            seed=seed,
            preprocess_image=preprocess_image,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )

        # Render the outputs
        s3_bucket_name = 'genies-ml-rnd'
        s3_key_prefix = '3D-generation/trellis/output_tasks'  # Folder path in your S3 bucket (optional)

        if return_render:
            video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
            video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
            video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
            buffer = BytesIO()
            imageio.mimsave(buffer, video, fps=30, format='mp4')
            buffer.seek(0)
            s3_key = f"{s3_key_prefix}/{trial_id}/{trial_id}_video.mp4"
            self.s3_client.upload_fileobj(buffer, s3_bucket_name, s3_key)
            key_video = s3_key
        else:
            video_path = None

        # GLB files can be extracted from the outputs
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=glb_extraction_simplify,          
            texture_size=glb_extraction_texture_size,  
        )
        
        glb.export(f"{trial_id}.glb") 
        key_glb = f"{s3_key_prefix}/{trial_id}/{trial_id}_model.glb"
        self.s3_client.upload_file(f"{trial_id}.glb", s3_bucket_name, key_glb)
        key_glb = key_glb

        return {
            "task_id": trial_id,
            "model_glb": key_glb,
            "video_path": video_path
        }

    def finalize(self):
        self.pipeline = None
