##################################################################
#
#    Deal with 2D images as numpy arrays of shape (C,H,W)
#    C = 3, indicating RGB.
##################################################################
import os
import torch
import numpy as np
from torchvision.transforms import transforms
import cv2
import matplotlib.pyplot as plt
import PIL
from scipy.interpolate import griddata
import torch.nn.functional as F



def resize_img_np(image, w,h):
    '''
    :param image: np, CHW,RGB
    :param w:
    :param h:
    :return:
    '''
    image = torch.tensor(image)
    image = transforms.Resize((h, w))(image)
    image = image.numpy()

    # image = np.transpose(image, (1, 2, 0))
    # image = cv2.resize(image, (w, h))
    # image = np.transpose(image, (2, 0, 1))
    return image

def resize_CHW_RGB_img_interp(img,new_height,new_width,mode='bilinear'):
    '''
    mode: 'bilinear', 'nearest'
    '''
    new_img = img.unsqueeze(0)
    if mode =='nearest':
        new_img = F.interpolate(new_img, size=(new_height, new_width), mode='nearest')

    elif mode =='bilinear':
        new_img = F.interpolate(new_img, size=(new_height, new_width), mode='bilinear', align_corners=False)

    # back to [3,H,W]
    new_img = new_img.squeeze(0)
    return new_img
    

def cat_images(img1, img2, margin=10, horizon = True):
    '''
    cat the img1 and img horizontally if horizon=True, else vertically.
    Resize img2 to fit the width of height of img1 using transforms.Resize() of pytorch like transforms.Resize((w, h))(img2)
    Both input images are np arrays of shape [C,H,W], RGB, ranging from 0 to 1. so as the result.

    :param img1:
    :param img2:
    :param margin:
    :param horizon:
    :return:
    '''
    # Convert numpy arrays to PyTorch tensors
    img1 = torch.tensor(img1)
    img2 = torch.tensor(img2)


    # Resize img2 to match the dimensions of img1
    _,h1,w1 = img1.shape
    _,h2,w2 = img2.shape
    if horizon:
        img2 = transforms.Resize((h1, int(w2 * h1 / h2)))(img2)
        _, h2, w2 = img2.shape
        width = w1 + margin + w2
        height = h1
    else:
        img2 = transforms.Resize((int(h2 * w1 / w2), w1))(img2)
        _, h2, w2 = img2.shape
        width = w1
        height = h1 + margin + h2

    # Create an empty numpy array to hold the concatenated image
    concat_img = np.ones((img1.shape[0], height, width))

    # print('img1.shape',img1.shape)
    # print('img2.shape',img2.shape)
    # print('concat_img.shape',concat_img.shape)

    # Copy img1 into the left or top side of the concatenated image
    if horizon:
        concat_img[:, :img1.shape[1], :img1.shape[2]] = img1
    else:
        concat_img[:, :img1.shape[1], :img1.shape[2]] = img1

    # Copy img2 into the right or bottom side of the concatenated image
    if horizon:
        concat_img[:, :img2.shape[1], img1.shape[2] + margin:] = img2
    else:
        concat_img[:, img1.shape[1] + margin:, :img2.shape[2]] = img2

    return concat_img





def display_CHW_RGB_img_np_cv2(img):
    if isinstance(img,torch.Tensor):
        img = img.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)  # from C,H,W to H,W,C
    pc_img = img[:, :, ::-1]  # from RGB to BGR
    cv2.imshow('img', pc_img)
    key = cv2.waitKey(0)  # Wait for keyboard input
    if key == 27:  # Check if Escape key was pressed
        cv2.destroyAllWindows()

def display_CHW_RGB_img_np_matplotlib(img):
    img = img.transpose(1, 2, 0)  # from C,H,W to H,W,C
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def display_CHW_RGB_img_np(img):
    display_CHW_RGB_img_np_cv2(img)

def save_CHW_RGB_img(img,file_name):
    img=img.transpose(1,2,0) #  # from C,H,W to H,W,C
    mask = (img==0).astype(np.uint8)
    img*=255 # from [0,1] to [0,255]
    img = img.clip(0, 255).astype(np.uint8)
    # img *= mask
    # img = np.ascontiguousarray(img[::-1, :, :])
    img = np.ascontiguousarray(img)
    # PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
    #     os.path.join(save_root, f'{name.split("/")[1]}.png'))


    image_pil = PIL.Image.fromarray(img,'RGB')

    image_pil.save(file_name)

def save_CHW_RGBA_img(img,file_name):
    img=img.transpose(1,2,0) #  # from C,H,W to H,W,C
    mask = (img==0).astype(np.uint8)
    img*=255 # from [0,1] to [0,255]
    img = img.clip(0, 255).astype(np.uint8)
    # img *= mask
    # img = np.ascontiguousarray(img[::-1, :, :])
    img = np.ascontiguousarray(img)
    # PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
    #     os.path.join(save_root, f'{name.split("/")[1]}.png'))


    image_pil = PIL.Image.fromarray(img,'RGBA')

    image_pil.save(file_name)

def load_CHW_RGB_img(file_name):
    '''
    CHW,RGB, float, 0-1
    :param file_name:
    :return:
    '''

    img = PIL.Image.open(file_name)
    if img.mode != 'RGB':
        img = img.convert('RGB')  # this is important
    img = torch.from_numpy(np.array(img))  # H,W,C = 4
    img = img[:, :, :3]  # H,W,C = 3
    img = img.float() /255.

    
    img = img.permute(2,0,1)
    return img

def load_CHW_RGBA_img_np(file_name,H=None,W=None):
    '''
    CHW,RGB, float, 0-1
    :param file_name:
    :return:
    '''

    img = PIL.Image.open(file_name)

    assert img.mode == 'RGBA'

    

    if H is not None and W is not None:
        img = img.resize((H,W))
    img = np.array(img)  # H,W,C = 4
    foreground_mask = img[:,:,3:] # H,W,C = 1
    img = img[:, :, :3]  # H,W,C = 3
    img = img.astype(np.float32)/255.

    img = img.transpose(2,0,1) # CHW
    foreground_mask = foreground_mask.transpose(2,0,1)[0] # HW
    return img,foreground_mask

def load_CHW_RGBA_img(file_name):
    '''
    CHW,RGB, float, 0-1
    :param file_name:
    :return:
    '''

    img = PIL.Image.open(file_name)
 
    
    assert img.mode == 'RGBA'
   

    img = torch.from_numpy(np.array(img))  # H,W,C = 4
    foreground_mask = img[:,:,3:]
    img = img[:, :, :3]  # H,W,C = 3
    img = img.float() /255.
 
    img = img.permute(2,0,1)
    return img,foreground_mask


######################## numpy
def paint_pixels(img, pixel_coords, pixel_colors, point_size):
    '''
    :param img: numpy array of shape [3,res,res]
    :param pixel_coords: [N,2], 2 for H and W
    :param pixel_colors: [N,3]
    :param point_size: paint not only the given pixels, but for each pixel, paint its neighbors whose distance to it is smaller than (point_size-1).
    :return:
    '''
    N = pixel_coords.shape[0]
    C = img.shape[0]
    # print('img.shape',img.shape)
    # print('pixel_coords.shape',pixel_coords.shape)
    # print('pixel_colors.shape',pixel_colors.shape)
    if point_size == 1:
        img[:, pixel_coords[:, 0], pixel_coords[:, 1]] = pixel_colors.T
    else:
        pixel_coords = np.round(pixel_coords).astype(int)
        if point_size > 1:
            xx, yy = np.meshgrid(np.arange(-point_size + 1, point_size, 1), np.arange(-point_size + 1, point_size, 1))
            grid = np.stack((xx, yy), 2).reshape(point_size * 2 - 1, point_size * 2 - 1, 2) # grid_res,grid_res,2
            grid_res = grid.shape[0]
            grid = grid + pixel_coords.reshape(N, 1, 1, 2) # [N,grid_res,grid_res,2]
            pixel_colors = np.repeat(pixel_colors[:, np.newaxis, np.newaxis, :], grid_res, axis=1)  # [N,3] -> [N,grid_res,1,3]
            pixel_colors = np.repeat(pixel_colors[:, :, :, :], grid_res, axis=2)  # [N,3] -> [N,grid_res,grid_res,3]
            mask = (grid[:, :, :, 0] >= 0) & (grid[:, :, :, 0] < img.shape[1]) & \
                   (grid[:, :, :, 1] >= 0) & (grid[:, :, :, 1] < img.shape[2])  # [N,grid_res,grid_res],
            grid = grid[mask]
            # print('pixel_colors.shape',pixel_colors.shape)
            # print('mask.shape',mask.shape)
            pixel_colors = pixel_colors[mask]
            indices = grid.astype(int)
            img[:, indices[:,  0], indices[:, 1]] = pixel_colors.transpose((1,  0))

    return img

def fill_hole(binary_img,kernel_size = 7):
    '''
    :param binary_img: [H,W]
    :return:  [H,W]
    '''
    ''' by contours'''
    # contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # out_img = np.zeros((binary_img.shape[0], binary_img.shape[1]))
    #
    # for i in range(len(contours)):
    #     cnt = contours[i]
    #     cv2.fillPoly(out_img, [cnt], color=255)
    # return out_img
    '''morphology  close'''
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    out_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    return out_img

def naive_inpainting(img,mask2,method='linear'):
    '''
    Numpy
    :param img: C,H,W
    :param mask2: H,W.  (~to_be_painted)
    :param method: 'linear' or 'nearest'
    :return: C,H,W
    '''
    # import kiui
    # temp = img.copy()
    # temp = temp.transpose(1, 2, 0)
    # temp[~mask2,0] = 1
    # temp[~mask2,1] = 0
    # temp[~mask2,2] = 1
    # cat = np.concatenate([temp,img.transpose(1,2,0)],1)
    # kiui.vis.plot_image(cat[...,:3])
    
    res = img.shape[1]

    # mask2 = mask2[0]
    need_to_fill_mask = ~(mask2.astype(np.bool_))

    # Create a grid of pixel coordinates
    y_coords, x_coords = np.indices(img.shape[1:])
    coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))

    # Flatten the image and mask arrays
    img_flat = img.reshape(img.shape[0], -1)
    mask_flat = need_to_fill_mask.ravel().astype(np.bool_)

    # Filter the image array to only include valid pixels
    valid_pixels = img_flat[:, ~mask_flat]
    valid_coords = coords[~mask_flat]

    x = np.arange(res)
    y = np.arange(res)
    xx, yy = np.meshgrid(x, y, indexing='xy')  # xy or ij

    # interpolated_pixels = griddata(valid_coords, valid_pixels.T, coords, input_pc_generate_method='nearest') # res*res,3
    # print('interpolated_pixels.shape', interpolated_pixels.shape)
    # filled_img = interpolated_pixels.transpose(1, 0)
    # filled_img = filled_img.reshape(-1, res, res)
    # # print('filled_img.shape',filled_img.shape)
    # save_CHW_RGB_img(filled_img, img_file_name)

    interpolated_pixels = griddata(valid_coords, valid_pixels.T, (xx, yy), method=method)  # res,res,3 # linear, nearest
    # print('interpolated_pixels.shape',interpolated_pixels.shape)
    filled_img = interpolated_pixels.transpose(2, 0, 1)
    # print('filled_img.shape',filled_img.shape)
    # save_CHW_RGB_img(filled_img, img_file_name)
    return filled_img

def smooth_img_researving_edges(img):
    '''
    :param img: res,res (gray img)
    :return:
    '''
    out_img = cv2.bilateralFilter(img.astype(np.float32), d=11, sigmaColor=10, sigmaSpace=150)
    out_img = cv2.bilateralFilter(out_img.astype(np.float32), d=11, sigmaColor=10, sigmaSpace=150)
    # src,d,sigmaColor,sigmaSpace,borderType
    # d: kernel size. size of neighbor
    # sigmaColor: only filter when |current pixel value - neighbor pixel value| smaller than this value
    # SigmaSpace: only work when d <=0. When d<=0, will use a d depending on sigmaSpace.
    return out_img



def detect_edges_in_gray_by_scharr(gray_img_uint8):
    '''

    :param gray_img_uint8: numpy array, uint8, [res,res], ranging from 0 to 255
    :return:
    '''
    im1x = cv2.Scharr(gray_img_uint8, cv2.CV_64F, 1, 0)
    im1y = cv2.Scharr(gray_img_uint8, cv2.CV_64F, 0, 1)
    im1x = cv2.convertScaleAbs(im1x)
    im1y = cv2.convertScaleAbs(im1y)
    edges = cv2.addWeighted(im1x, 0.5, im1y, 0.5, 0)
    return edges






#################################### pytorch
@torch.no_grad()


def detect_edges_in_gray_by_scharr_torch_batch(gray_imgs_float32):
    '''
    :param gray_imgs_float32: torch tensor, 
    :return: edges: torch tensor, 
    '''
    device = gray_imgs_float32.device

    # Calculate the x and y gradients using the Scharr kernel
    # We use the Scharr kernel with ddepth = -1 to get the gradients in the same data type as the input
    kernel = torch.tensor([[[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]], dtype=torch.float32).to(device)
    im1x = torch.nn.functional.conv2d(gray_imgs_float32, kernel, padding=1) # BCHW

    im1x =  torch.abs(im1x)

    kernel = torch.tensor([[[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]]], dtype=torch.float32).to(device)
    im1y = torch.nn.functional.conv2d(gray_imgs_float32, kernel, padding=1)
    im1y = torch.abs(im1y)


    # Compute the edge map by combining the x and y gradients with equal weights
    edges = torch.add(im1x, im1y) / 2.0

    # vis = True
    # if vis:
    #     cat = cat_images(im1x[0].clip(0,255).repeat(3,1,1).detach().cpu().numpy()/255.0,
    #                      im1y[0].clip(0, 255).repeat(3, 1, 1).detach().cpu().numpy()/255.0,
    #                      )
    #     display_CHW_RGB_img_np_matplotlib(cat)
    
    # this will make the outer edge white, so deal with it now
    mask = torch.zeros_like(edges, dtype=torch.bool)  
    mask[..., 0, :] = True  # 
    mask[..., -1, :] = True  #
    mask[..., :, 0] = True  # 
    mask[..., :, -1] = True  # 
    
    edges[mask] = 0
    
    # for i in range(H):
    #     for j in range(W):
    #         if i==0 or  i==H-1 or j==0 or j==W-1:
    #             edges[:,i,j] = 0
    return edges

def scharr_edge_RGB_torch(img_BCHW):

    B,C,H,W = img_BCHW.shape
    edge_img_BCHW = torch.zeros_like(img_BCHW).to(img_BCHW.device)
    for c in range(C):
        temp1 = edge_img_BCHW[:,c,:,:]
        temp2 = img_BCHW[:,c,:,:]
        temp3 = detect_edges_in_gray_by_scharr_torch_batch(img_BCHW[:,c:c+1,:,:])
       
        edge_img_BCHW[:,c:c+1,:,:] = detect_edges_in_gray_by_scharr_torch_batch(img_BCHW[:,c:c+1,:,:])
        
    edges_NHW = edge_img_BCHW.max(dim=1)[0] # pick the biggest one among RGB channels
    edges_NHW /= edges_NHW.max() # normalize to 0 and 1
    edge_img_BCHW = edges_NHW.unsqueeze(1).repeat(1,C,1,1)
    return edge_img_BCHW

def sobel_edge_torch(img_BCHW):
    ''''
    input: image tensors of shape BCHW
    
    output:
    edge strength tensor of shape BCHW
    '''
    C = img_BCHW.shape[1]
    # Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(1, C, 1, 1).to(img_BCHW.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(1, C, 1, 1).to(img_BCHW.device)

    # Convolution
    grad_x = F.conv2d(img_BCHW, sobel_x, padding=1)
    grad_y = F.conv2d(img_BCHW, sobel_y, padding=1)

    # Calculate edge strength
    edge_strength = torch.sqrt(grad_x ** 2 + grad_y ** 2) # N1HW
    import kiui
    kiui.lo(edge_strength)
    edge_strength = edge_strength.repeat(1, C, 1, 1) # NCHW
    kiui.lo(edge_strength)
    return edge_strength

def dilate_torch_batch(binary_img_batch, kernel_size):
    """
    dilate white pixels like cv2.dilate
    :param img_batch: [B,  H, W] ,B indicate batch size. each element is either 0 or 1
    :param kernel_size:
    :return: [B,  H, W] same size as before
    """

    pad = (kernel_size - 1) // 2
    bin_img = F.pad(binary_img_batch.unsqueeze(1), pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=kernel_size, stride=1, padding=0)
    out = out.squeeze(1)
    return out

def get_forground_inner_edge_mask(foreground_mask,method = 'dilate'):
    '''
    :param foreground_mask: res,res
    :param method: 'shift' or 'dilate'. 'dilate' produces more smooth_depth_map result
    :return:
    '''
    if method == 'shift':
        # Create masks for the neighbors in each direction
        top_mask = torch.roll(foreground_mask, shifts=1, dims=0)
        bottom_mask = torch.roll(foreground_mask, shifts=-1, dims=0)
        left_mask = torch.roll(foreground_mask, shifts=1, dims=1)
        right_mask = torch.roll(foreground_mask, shifts=-1, dims=1)

        # Find the edge pixels by checking if any neighbor is 0
        # edge_mask = (top_mask | bottom_mask | left_mask | right_mask) & foreground_mask
        top_edge_mask = foreground_mask ^ (top_mask& foreground_mask)
        bottom_edge_mask = foreground_mask ^ (bottom_mask& foreground_mask)
        left_edge_mask = foreground_mask ^ (left_mask& foreground_mask)
        right_edge_mask = foreground_mask ^ (right_mask& foreground_mask)
        edge_mask = top_edge_mask | bottom_edge_mask | left_edge_mask | right_edge_mask
        edge_mask = edge_mask * foreground_mask

    elif method == 'dilate':
        dilated_back_mask = torch.nn.functional.max_pool2d((~foreground_mask).unsqueeze(0).unsqueeze(0).float(),
                                                      kernel_size=3, stride=1, padding=1).squeeze().bool() # res,res
        edge_mask = dilated_back_mask & foreground_mask


    vis = False
    if vis:
        foreground_mask_img = foreground_mask.unsqueeze(0).repeat(3, 1, 1)
        edge_mask_img = edge_mask.unsqueeze(0).repeat(3, 1, 1)
        cat = cat_images(foreground_mask_img.cpu().numpy(), edge_mask_img.cpu().numpy())

        display_CHW_RGB_img_np_matplotlib(cat)
    return edge_mask


def modify_bg_color_of_rgba_np(img_HW4_np, old_bg_color, new_bg_color):
    """
    Modify the background color of an RGBA image represented as a NumPy array.
    
    Parameters:
    - img_HW4_np: A NumPy array representing the image with shape (H, W, 4) or (B, H, W, 4).
    - old_bg_color: A tuple or list representing the old background color in RGBA format.
    - new_bg_color: A tuple or list representing the new background color in RGBA format.
    
    Returns:
    - A NumPy array representing the modified image with the new background color.
    """
    
    # Convert input colors to NumPy arrays for vectorized operations
    old_bg_color = np.array(old_bg_color)
    new_bg_color = np.array(new_bg_color)
    
    # Extract the alpha channel from the image
    alpha_channel = img_HW4_np[..., 3:]
    
    # Identify transparent pixels (alpha == 0)
    transparent_pixels = (alpha_channel == 0)
    
    # Avoid division by zero warning by adding a small epsilon value to the alpha channel
    epsilon = np.finfo(float).eps
    safe_alpha_channel = np.where(transparent_pixels, epsilon, alpha_channel)
    
    # Calculate the original color (without considering the background contribution)
    # For transparent pixels, use the old background color directly
    original_color = np.where(transparent_pixels, old_bg_color, 
                              (img_HW4_np[..., :3] - (1 - alpha_channel) * old_bg_color) / safe_alpha_channel)
    
    # # Ensure no division by zero errors (in case alpha is exactly 0)
    # # This step is redundant since we have already handled this above
    # original_color = np.where(alpha_channel == 0, old_bg_color, original_color)
    
    # Create the new image with the updated background color
    new_image = alpha_channel * original_color + (1 - alpha_channel) * new_bg_color
    
    # Stack the new image channels with the alpha channel to get the final result
    modified_image = np.concatenate([new_image, alpha_channel], axis=-1)
    
    # import kiui
    # kiui.lo(new_image)
    # strange_mask = new_image<0
    # temp = strange_mask.sum(-1).astype(np.bool_)
    # temp = alpha_channel[temp]
    # kiui.lo(temp)
    # kiui.vis.plot_image(temp)
    # kiui.vis.plot_image(strange_mask)
    modified_image = np.clip(modified_image, 0, 1)
    return modified_image