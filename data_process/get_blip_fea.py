import torch
import numpy as np
from tqdm import tqdm
from torchvision import datasets,transforms
device = "cuda:0" if torch.cuda.is_available() else "cpu"
from transformers import AutoProcessor, Blip2Model
blip_loc = "/pretrained_models/blip2/"
processor = AutoProcessor.from_pretrained(blip_loc)
model = Blip2Model.from_pretrained(blip_loc, torch_dtype=torch.float16)
model.to(device)

file_name= "./images"
dataset = datasets.ImageFolder(file_name, transform=processor)
out_name="./blip_fea/"
bz = 8
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bz, shuffle=False)
print(len(dataloader.dataset.samples))

for i, (images, labels) in tqdm(enumerate(dataloader),total=len(dataloader)):
    images["pixel_values"]=images["pixel_values"][0]
    images = images.to(device, torch.float16)
    image_embeds = model.get_image_features(**images)[0]

    # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
    image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_outputs = model.qformer(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask
    )
    image_features = query_outputs[0]
    for j, image_feature in enumerate(image_features):
        sample_fname, _ = dataloader.dataset.samples[i*bz+j]
        out_file_name="./blip_fea/"+sample_fname.split("/")[-2]+"/"+sample_fname.split("/")[-1]
        image_save = image_feature.cpu().detach().numpy()
        np.save(out_file_name,image_save)