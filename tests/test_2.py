import scenedetect, transformers, torch, accelerate
print(scenedetect.__version__, transformers.__version__, torch.__version__, accelerate.__version__, sep='\n')

# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
# from PIL import Image
# img = Image.open("photo.jpg")
# model_id = "Qwen/Qwen2-VL-2B-Instruct"
# proc = AutoProcessor.from_pretrained(model_id, use_fast=True)
# model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, device_map="cpu").eval()

# prompt = proc.apply_chat_template(
#     [{"role":"user",
#       "content":[
#         {"type":"image","image":img},
#         {"type":"text","text":"Descride picture in one line."}
#       ]}],
#     tokenize=False,
#     add_generation_prompt=True
# )
# inputs = proc(text=prompt, images=[img], return_tensors="pt")
# out = model.generate(**inputs, max_new_tokens=32)
# print(proc.decode(out[0], skip_special_tokens=True))
