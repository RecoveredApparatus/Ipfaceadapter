import torch
from torch import cuda
import spaces
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from transformers import AutoFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID, IPAdapterFaceIDPlus
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import gradio as gr
import cv2

base_model_path = "stablediffusionapi/samaritan-3d-cartoon"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid_sd15.bin", repo_type="model")
ip_plus_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid-plusv2_sd15.bin", repo_type="model")

safety_model_id = None
safety_feature_extractor = None
safety_checker = None

device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=safety_feature_extractor,
    safety_checker=safety_checker
)

#pipe.load_lora_weights("h94/IP-Adapter-FaceID", weight_name="ip-adapter-faceid-plusv2_sd15_lora.safetensors")
#pipe.fuse_lora()

ip_model = IPAdapterFaceID(pipe, ip_ckpt, device)
ip_model_plus = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_plus_ckpt, device)

@spaces.GPU(enable_queue=True)
def generate_image(images, prompt, negative_prompt, preserve_face_structure, face_strength, likeness_strength, num_samples, guidance_scale, nfaa_negative_prompt, progress=gr.Progress(track_tqdm=True)):
    print(cuda.memory_summary())
    pipe.to(device)
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print(cuda.memory_summary())
    faceid_all_embeds = []
    first_iteration = True
    for image in images:
        face = cv2.imread(image)
        faces = app.get(face)
        faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        faceid_all_embeds.append(faceid_embed)
        if(first_iteration and preserve_face_structure):
            face_image = face_align.norm_crop(face, landmark=faces[0].kps, image_size=224) # you can also segment the face
            first_iteration = False
            
    average_embedding = torch.mean(torch.stack(faceid_all_embeds, dim=0), dim=0)
    
    total_negative_prompt = f"{negative_prompt} {nfaa_negative_prompt}"
    
    if(not preserve_face_structure):
        print("Generating normal")
        image = ip_model.generate(
            prompt=prompt, negative_prompt=total_negative_prompt, faceid_embeds=average_embedding,
            scale=likeness_strength, width=512, height=512, num_inference_steps=30
        )
    else:
        print("Generating plus")
        image = ip_model_plus.generate(
            prompt=prompt, negative_prompt=total_negative_prompt, faceid_embeds=average_embedding,
            scale=likeness_strength, face_image=face_image, shortcut=True, s_scale=face_strength, num_samples=num_samples, guidance_scale=guidance_scale, width=512, height=512, num_inference_steps=30
        )
    
    print(cuda.memory_summary())
    print(image)
    return image

def change_style(style):
    if style == "Photorealistic":
        return(gr.update(value=True), gr.update(value=1.3), gr.update(value=1.0))
    else:
        return(gr.update(value=True), gr.update(value=0.1), gr.update(value=0.8))

def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
css = '''
h1{margin-bottom: 0 !important}
'''
with gr.Blocks(css=css) as demo:
    gr.Markdown("# IP-Adapter-FaceID Plus demo")
    gr.Markdown("Demo for the [h94/IP-Adapter-FaceID model](https://huggingface.co/h94/IP-Adapter-FaceID) - Non-commercial license")
    with gr.Row():
        with gr.Column():
            files = gr.Files(
                        label="Drag 1 or more photos of your face",
                        file_types=["image"]
                    )
            uploaded_files = gr.Gallery(label="Your images", visible=False, columns=5, rows=1, height=125)
            with gr.Column(visible=False) as clear_button:
                remove_and_reupload = gr.ClearButton(value="Remove and upload new ones", components=files, size="sm")
            prompt = gr.Textbox(label="Prompt",
                       info="Try something like 'a photo of a man/woman/person'",
                       placeholder="A photo of a [man/woman/person]...")
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="low quality")
            style = gr.Radio(label="Generation type", info="For stylized try prompts like 'a watercolor painting of a woman'", choices=["Photorealistic", "Stylized"], value="Photorealistic")
            submit = gr.Button("Submit")
            with gr.Accordion(open=False, label="Advanced Options"):
                preserve = gr.Checkbox(label="Preserve Face Structure", info="Higher quality, less versatility (the face structure of your first photo will be preserved). Unchecking this will use the v1 model.", value=True)
                face_strength = gr.Slider(label="Face Structure strength", info="Only applied if preserve face structure is checked", value=1.3, step=0.1, minimum=0, maximum=3)
                likeness_strength = gr.Slider(label="Face Embed strength", value=1.0, step=0.1, minimum=0, maximum=5)
                #seed = gr.Slider(label="seed", value=1000, step=100, minimum=100, maximum=2000)
                guidance_scale = gr.Slider(label="CFG", value=1.0, step=0.5, minimum=0, maximum=20) 
                num_samples = gr.Slider(label="samples", info="number of generated images", value=1, step=1, minimum=1, maximum=16)
                nfaa_negative_prompts = gr.Textbox(label="Appended Negative Prompts 4 realistic vision model", info="Negative prompts to steer generations towards safe for all audiences outputs", value="deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck")    
        with gr.Column():
            gallery = gr.Gallery(label="Generated Images")
        style.change(fn=change_style,
                    inputs=style,
                    outputs=[preserve, face_strength, likeness_strength])
        files.upload(fn=swap_to_gallery, inputs=files, outputs=[uploaded_files, clear_button, files])
        remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files, clear_button, files])
        submit.click(fn=generate_image,
                    inputs=[files,prompt,negative_prompt,preserve, face_strength, likeness_strength, num_samples, guidance_scale, nfaa_negative_prompts],
                    outputs=gallery)
    
    gr.Markdown("safety filter is off, enable in lines 20-23")
print(cuda.memory_summary())   
demo.launch(share=True)

