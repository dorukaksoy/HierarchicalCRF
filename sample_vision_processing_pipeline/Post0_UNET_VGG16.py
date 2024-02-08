# ---------------------------------------------------------------------------
# --- Prerequisite(s) --
# An image
# --- Description --
# A machine vision model is provided based on U-Net architecture with a VGG-16 backend
# ------------------
# Authors: Doruk Aksoy; University of California, Irvine
# Contact: (daksoy@uci.edu)
# Date: 02/07/2024
# ------------------
# Version: Python 3.8
# Execution: python3 Post0_UNET_VGG16.py
# ---------------------------------------------------------------------------
import numpy as np
import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from PIL import Image
import PostProcessingPlots as pplot

# Check if a display environment exists
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
    VISUALIZATIONS=False
else: 
    VISUALIZATIONS=True # False if no visualization environment exists
import matplotlib.pyplot as plt
# %%  Define the Custom convolution block and decoder
# (From https://saifgazali.medium.com/retina-blood-vessel-segmentation-using-vgg16-unet-7262f97e1695
from tensorflow.keras import layers
from tensorflow.keras import initializers
def conv_block(inputs,num_filters):
  x = layers.Conv2D(num_filters,3,padding='same')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.Conv2D(num_filters,3,padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  return x

def define_decoder(inputs,skip_layer,num_filters):
  init = initializers.RandomNormal(stddev=0.02)
  x = layers.Conv2DTranspose(num_filters,(2,2),strides=(2,2),padding='same',kernel_initializer=init)(inputs)
  g = layers.Concatenate()([x,skip_layer])
  g = conv_block(g,num_filters)
  return g

# %% Define Custom Metrics
def iou(y_true,y_pred):
  def f(y_true,y_pred):
    intersection = (y_true*y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    x = (intersection + 1e-15) / (union + 1e-15)
    x = x.astype(np.float32)
    return x
  return tf.numpy_function(f,[y_true,y_pred],tf.float32)

smooth = 1e-15
def dice_coef(y_true,y_pred):
  y_true = tf.keras.layers.Flatten()(y_true)
  y_pred = tf.keras.layers.Flatten()(y_pred)
  intersection = tf.reduce_sum(y_true*y_pred)
  return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

def dice_loss(y_true,y_pred):
  return 1.0 - dice_coef(y_true,y_pred)

# %% Load model (This can be replaced with any model that outputs a binary mask)
model_id='Machine_Vision'
print('Loading model: UNET_VGG16_{}'.format(model_id))
model = tf.keras.models.load_model('./{}.model'.format(model_id),custom_objects={'conv_block':conv_block,'define_decoder':define_decoder,'iou':iou,'dice_coef':dice_coef,'dice_loss':dice_loss})

# %% Predict from full-resolution images
# Create masks from predictions as possibilities
def createMaskFromMean(y_pred):
    assert y_pred.ndim==2, "Expected array of shape (H, W)"
    # pred_im = y_pred.reshape(input_h, input_w)
    pred_im_norm = y_pred * 1e04
    thresh = np.mean(pred_im_norm)
    pred_im_norm[pred_im_norm<thresh]=0
    pred_im_norm[pred_im_norm>=thresh]=1
    return pred_im_norm.astype(bool)

print("Predicting the segmentation mask using the computer vision model...")
sample_img = np.asarray(plt.imread('./img.tif'), dtype='uint8')

X_new = sample_img[np.newaxis,...,np.newaxis] # pretend we have new images
y_pred = model.predict(X_new)[0,:,:,0]
    
predicted_mask = createMaskFromMean(y_pred)
print("Mask is created.")

if VISUALIZATIONS:    
    def overlayImageandMask(background, foreground, color='red'):
        assert background.ndim==3, "Expected array of shape (H, W, C)"                            
        assert foreground.ndim==2, "Expected array of shape (H, W)"

        # Preserve original blacks from background
        original_blacks = np.nonzero(background == 0)
        
        image_fg = ~np.expand_dims(foreground, axis=-1)
        image_bg = background
        overlay = np.multiply(image_bg,image_fg)

        r_ch = np.copy(overlay[:,:,0])
        r_ch[np.where(r_ch == 0)] = 255
        g_ch = np.copy(overlay[:,:,1])
        g_ch[np.where(g_ch == 0)] = 255
        b_ch = np.copy(overlay[:,:,2])
        b_ch[np.where(b_ch == 0)] = 255
        if color=='red' or color=='r':
            overlay[:,:,0] = r_ch
        elif color=='yellow' or color=='y':
            overlay[:,:,0] = np.copy(r_ch)
            overlay[:,:,1] = np.copy(g_ch)
        elif color=='white' or color=='w':
            overlay[:,:,0] = np.copy(r_ch)
            overlay[:,:,1] = np.copy(g_ch)
            overlay[:,:,2] = np.copy(b_ch)
        elif color=='black' or color=='b':
            pass

        # Put original blacks back
        overlay[original_blacks] = 0
        
        return (overlay)
        
        # Plot image and predicted mask side-by-side
        pplot.plot_multiple_images_single_row([sample_img,
                  overlayImageandMask(sample_img,predicted_mask, color='y')
                    ])             

# Save the predicted mask
pred_msk = Image.fromarray(predicted_mask)
pred_msk.save('pred.png')
print("Predicted mask, 'pred.png', is saved to the current directory.")