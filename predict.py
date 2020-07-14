
import argparse
import numpy as np
from PIL import Image


import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
tfds.disable_progress_bar()
import json

parser = argparse.ArgumentParser(description='prdict the Flower classification.')


parser.add_argument('--image_path', default='./test_images/hard-leaved_pocket_orchid.jpg', help=' path of image to predict')
parser.add_argument('--model', default='best_model.h5', help=' model path',type=str)
parser.add_argument('--top_k', default=5, help=' the number of top_k result',type=int)
parser.add_argument('--classes', default='label_map.json', help=' class namees')


args = parser.parse_args()
img = args.image_path
model= args.model
top_k = args.top_k
classes = args.classes

with open(classes, 'r') as f:
    class_names = json.load(f)

modlel_load = tf.keras.models.load_model(model,custom_objects={'KerasLayer':hub.KerasLayer})


def process_image(image):
    t_imge = tf.convert_to_tensor(image)
    resize_image = tf.image.resize(t_imge, (224, 224))
    normalizing_image = resize_image /255
    convert_img  = normalizing_image.numpy()
    return convert_img


class_names_new = dict()
for key in class_names:
    class_names_new[str(int(key)-1)] = class_names[key]
def predict(img,modlel_load,topk):
    
    
    img = Image.open(img)
    test_img = np.asarray(img)
    transform_img = process_image(test_img)
    redim_img = np.expand_dims(transform_img,axis=0)
    prob_pred= modlel_load.predict(redim_img)
    #print(type(prob_pred))
    prob_pred = prob_pred.tolist()
    #print(prob_pred)
    values, indices = tf.math.top_k(prob_pred, k=top_k)
    probs=values.numpy().tolist()[0]
    classes= indices.numpy().tolist()[0]
    
    label_names = [class_names_new[str(names)] for names in classes]
    
    print(label_names,'\n',probs)
    print("the high Possibilities is ",label_names[0])
    #return probs , classes
if __name__ == "__main__":
     predict(img, modlel_load, top_k)
        
