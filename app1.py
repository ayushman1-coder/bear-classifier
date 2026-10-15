#!/usr/bin/env python
# coding: utf-8

# In[2]:


#hide
get_ipython().system('pip install -Uqq fastbook')
import fastbook
fastbook.setup_book()


# In[3]:


#hide
from fastbook import *
from fastai.vision.widgets import *


# In[4]:


search_images_ddg


# In[5]:


from ddgs import DDGS as ddgs


# In[6]:


from ddgs import DDGS
from fastai.vision.all import *

search = DDGS()
img = list(search.images("damaged cars", max_results=10))
urls = [item["image"] for item in img]

from IPython.display import Image, display,HTML
# display(Image(url=img[2]["image"]))
display(HTML("".join(
    f'<img src="{item["image"]}" width="150" style="margin:5px;">'
    for item in img[:5]
)))

path = Path("bears")
path.mkdir(exist_ok=True)

download_images(path, urls=urls)


# In[7]:


# fns = get_image_files(path)
# im = PILImage.create(fns[2])
# im.to_thumb(64, 64)


# In[8]:


fns = get_image_files(path)
print(len(fns))


# In[9]:


dest ='bears/2026b0d6-79ea-4952-a5fb-a1cd4a4221e9.jpeg'
from PIL import Image
im = Image.open(dest)
im.thumbnail((128,128))


# In[10]:


display(im)


# In[11]:


bear_types = 'grizzly_bear','black_bear','bearlike_photo'
path = Path('bear')


# In[12]:


if not path.exists():
    path.mkdir()
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        img = list(search.images(o, max_results=150))
        urls = [item["image"] for item in img] 
        download_images(dest, urls=urls)


# In[13]:


fns = get_image_files(path)
print(len(fns))
print(fns[3])


# In[14]:


failed = L(
    f for f in fns
    if not verify_image(f)
)

print("Invalid images:", len(failed))
failed.map(Path.unlink)
print("Remaining images:", len(get_image_files(path)))


# In[15]:


fns = get_image_files(path)
for f in fns:
    im = PILImage.create(f)
    display(im.to_thumb(128, 128))


# In[16]:


im = PILImage.create(fns[10])
display(im.to_thumb(64, 64))


# In[17]:


print(len(fns))
print(fns[1])


# In[18]:


dest =r'bear\bearlike_photo\172cb240-54e2-42ab-84f6-e305d5974681.jpg'
from PIL import Image
im = Image.open(dest)
im.thumbnail((128,128))
display(im)


# In[19]:


dblock=DataBlock(blocks=(ImageBlock,CategoryBlock),
              get_y=parent_label,
               splitter=RandomSplitter(valid_pct=.2,seed=23),
               item_tfms=Resize(128),
               get_items=get_image_files
                )


# In[20]:


dls=dblock.dataloaders(path,bs=16,device='cuda')
dls.train.drop_last=False


# In[21]:


print("Training images:", len(dls.train_ds))
print("Validation images:", len(dls.valid_ds))

images, labels = dls.one_batch()

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)


# In[22]:


dls.train.show_batch(max_n=5, nrows=1)


# In[23]:


fns = get_image_files(path)


# In[24]:


print("Total images:", len(fns))
print("Training images:", len(dls.train_ds))
print("Validation images:", len(dls.valid_ds))
print("Training batches:", len(dls.train))
print("Training batches:", len(dls.valid))


# In[25]:


print("Drop last batch:", dls.train.drop_last)

for i, (images, labels) in enumerate(dls.train):
    print(f"Batch {i+1}: {len(images)} images")


# In[26]:


bears = dblock.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=6, nrows=1)


# In[27]:


bears = dblock.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = bears.dataloaders(path,bs=8)
dls.train.show_batch(max_n=5, nrows=1)


# In[28]:


bears = dblock.new(item_tfms=RandomResizedCrop(128, min_scale=0.4))
dls = bears.dataloaders(path,bs=8)
dls.valid.show_batch(max_n=5, nrows=1, unique=True)


# In[29]:


bears =dblock.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path,bs=8)
dls.train.show_batch(max_n=8, nrows=1, unique=True)


# In[30]:


print("Training images:", len(dls.train_ds))
print("Validation images:", len(dls.valid_ds))

print("Training batches:", len(dls.train))
print("Validation batches:", len(dls.valid))
for i, (images, labels) in enumerate(dls.train):
    print(f"Batch {i+1}: {len(images)} images")


# In[31]:


bears = dblock.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path,bs=16)
dls.train.drop_last=False


# In[32]:


for i, (images, labels) in enumerate(dls.train):
    print(f"Batch {i+1}: {len(images)} images")
print("Drop last batch:", dls.train.drop_last)    


# In[33]:


learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)


# In[34]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[35]:


interp.plot_top_losses(2, nrows=1)


# In[36]:


interp = ClassificationInterpretation.from_learner(learn,dl=dls.train)
interp.plot_confusion_matrix()


# In[37]:


interp.plot_top_losses(3, nrows=1)


# In[38]:


cleaner = ImageClassifierCleaner(learn)
cleaner


# In[39]:


for idx in cleaner.delete(): cleaner.fns[idx].unlink()
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)


# In[40]:


import huggingface_hub
print(huggingface_hub.__version__)


# In[42]:


learn.export()


# In[43]:


path = Path()
path.ls(file_exts='.pkl')


# In[46]:


learn.predict(r"bears\09be4373-d0a8-4d1f-b4fd-a88840679f8e.jpg")


# In[56]:


learn.predict(r"bear\grizzly_bear\1ec910c4-4a84-4ce7-bf70-0fea59521107.png")


# In[56]:


learn.predict(r"bear\grizzly_bear\1ec910c4-4a84-4ce7-bf70-0fea59521107.png")


# In[59]:


learn.dls.vocab


# In[101]:


category=('bearlike_photo', 'black_bear', 'grizzly_bear')
def classify_image(img):
    pred,idx,probs=learn.predict(img)
    return dict(zip(category,map(float,probs)))


# In[102]:


classify_image(r"bear\grizzly_bear\1ec910c4-4a84-4ce7-bf70-0fea59521107.png")


# In[95]:


fns = get_image_files(path/'bear')


# In[97]:


import gradio as gr


# In[103]:


image = gr.Image(
    type="pil",
    label="Upload a bear image"
)
label = gr.Label(num_top_classes=3)

examples = [[str(f)] for f in fns[:5]]
imgf=gr.Interface(fn=classify_image,inputs=image,outputs=label,examples=examples)


# In[104]:


imgf.launch()


# In[108]:


from nbdev import nbdev_export

nbdev_export()


# In[109]:


import os
print(os.getcwd())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[88]:


# %pip install gradio


# In[60]:


#hide_output
btn_upload = widgets.FileUpload()
btn_upload


# In[67]:


btn_upload = SimpleNamespace(data = [r"bear\bearlike_photo\de36ce58-da2f-4d43-9e14-1fb1c80294d1.jpg"])


# In[68]:


img = PILImage.create(btn_upload.data[-1])


# In[69]:


display(img.to_thumb(128, 128))


# In[70]:


out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
out_pl


# In[72]:


pred,pred_idx,probs = learn.predict(img)
#hide_output
lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred


# In[74]:


#hide_output
btn_run = widgets.Button(description='Classify')
btn_run
def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)


# In[75]:


btn_upload = widgets.FileUpload()


# In[76]:


#hide_output
VBox([widgets.Label('Select your bear!'), 
      btn_upload, btn_run, out_pl, lbl_pred])


# In[41]:


# from fastai.vision.all import *

# path = Path("damaged_cars")

# dls = ImageDataLoaders.from_folder(
#     path,
#     train="train",
#     valid="valid",
#     valid_pct=0.2,
#     seed=42,
#     vocab=None,
#     item_tfms=Resize(128),
#     batch_tfms=Normalize.from_stats(*imagenet_stats),
#     bs=16,
#     shuffle=True,
#     device=None
# )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




