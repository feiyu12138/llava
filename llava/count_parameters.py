from transformers import Blip2Model
import torch
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

vision_encoder = model.vision_model
qformer = model.qformer

print("Parameters in vision encoder:", count_parameters(vision_encoder))
print("Parameters in qformer:", count_parameters(qformer))