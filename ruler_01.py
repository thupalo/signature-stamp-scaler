## %%
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

SMALL_WIDTH = 300
DEBUG = False

### %%
# calculate fourier transformation by rows using window size 512 and overlap 256
def calculate_fourier_transform(image, window_size=512, overlap=256):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get image dimensions
    height, width = img_gray.shape
    # Prepare for results
    results = []
    # Slide window across the image
    for y in range(height):
        if width % overlap != 0:
            new_width = width + (overlap - (width % overlap))
            # print(f"Padding row {y} from {width} to {new_width}")
            line = np.pad(img_gray[y, :], (0, new_width - width), mode='constant')
        else:
            line = img_gray[y, :]
        l = None
        for start in range(0, new_width - window_size + 1, overlap):
            segment = line[start:start + window_size]
            # Apply a window function (e.g., Hamming window) to reduce spectral leakage
            window = np.hamming(window_size)
            segment_windowed = segment * window

            # Compute the FFT of the windowed segment
            f_transform = np.fft.fft(segment_windowed)
            f_transform_shifted = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_transform_shifted)[window_size//2+1:]
            magnitude_spectrum = np.log(magnitude_spectrum / np.max(magnitude_spectrum) + 1e-10)
            if l is None:
                l = magnitude_spectrum
            else:
                l = l + magnitude_spectrum
        results.append(l)
    results = np.array(results)
    return results[:, 1:]

def find_ruler(image, horizontal=True):
    if not horizontal:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    results = calculate_fourier_transform(image)
    filtered = cv2.GaussianBlur(results, (3,15), 0)
    coordinates = np.unravel_index(np.argmax(filtered, axis=None), filtered.shape)
    y, f1 = coordinates
    f1 = f1 + 1
    ruler = filtered[y, :]
    f2 = np.argmax(filtered[y, f1+f1//2:2*f1+f1//2]) + f1 + f1//2
    f3 = np.argmax(filtered[y, 2*f1+f1//2:3*f1+f1//2]) + 2*f1 + f1//2
    f4 = np.argmax(filtered[y, 3*f1+f1//2:4*f1+f1//2]) + 3*f1 + f1//2
    f5 = np.argmax(filtered[y, 4*f1+f1//2:5*f1+f1//2]) + 4*f1 + f1//2
    f = (f1 + f2/2 + f3/3 + f4/4 + f5/5) / 5
    std_err = np.std([f1, f2/2, f3/3, f4/4, f5/5])

    return {"f": f, "std_err": std_err, "y": y}

#%%
def detect_rulers_and_crop(pimage):

    h_ruler = find_ruler(pimage, horizontal=True)
    if DEBUG:
        print("Horizontal ruler data:", h_ruler)
    v_ruler = find_ruler(pimage, horizontal=False)
    if DEBUG:
        print("Vertical ruler data:", v_ruler)

    image = pimage
    h_scale = None
    v_scale = None
    if h_ruler["f"] > 5 and h_ruler["f"] < 40 and h_ruler["std_err"] < 2:
        # Crop the image using the ruler information
        y = h_ruler["y"]
        image = image[:y, :]
        h_scale = h_ruler["f"] * 2 # pix/mmm
    if v_ruler["f"] > 5 and v_ruler["f"] < 40 and v_ruler["std_err"] < 2:
        # Crop the image using the ruler information
        x = v_ruler["y"]
        image = image[:, x:]
        v_scale = v_ruler["f"] * 2 # pix/mmm
    if h_scale is None and v_scale is not None:
        h_scale = v_scale
    if v_scale is None and h_scale is not None:
        v_scale = h_scale

    if DEBUG:
        print("Image size after cropping:", image.shape)
        print("Horizontal scale (pix/mm):", h_scale)
        print("Vertical scale (pix/mm):", v_scale)
        cv2.imwrite("debug_info_1.png", image)
        print("Cropped image saved as debug_info_1.png")

    return image, (h_scale, v_scale)

## %%
def crop_to_content(img1):

    global SMALL_WIDTH

    img = img1.copy()
    SMALL_HEIGHT = int(SMALL_WIDTH * img.shape[0] / img.shape[1])
    img = cv2.resize(img, (SMALL_WIDTH, SMALL_HEIGHT), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Otsu's thresholding
    _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    kernel = np.ones((9,9),np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = ~img 

    # contour extraction for each combinded region in the mask the outer contour will be extracted
    contours,_ = cv2.findContours(img, mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)

    found = False
    for contour in contours:
        # computed bounding rect of contour, contour contains a list of points
        roi = cv2.boundingRect(contour)
        _left, _top, _right, _bottom = roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]
        if _left == 0 or _top == 0 or _right >= img.shape[1] or _bottom >= img.shape[0]:
            continue

        if not found:
            left, top, right, bottom = _left, _top, _right, _bottom
            found = True
        else:
            left = min(left, _left)
            top = min(top, _top)
            right = max(right, _right)
            bottom = max(bottom, _bottom)

    if not found:
        print("No valid bounding box found.")
        return img1

    else:
        # scale up
        left = int(left * img1.shape[1] / SMALL_WIDTH)
        top = int(top * img1.shape[0] / SMALL_HEIGHT)
        right = int(right * img1.shape[1] / SMALL_WIDTH)
        bottom = int(bottom * img1.shape[0] / SMALL_HEIGHT)

        # crop img1
        img2 = img1[top:bottom, left:right]
        if DEBUG:
            print("Bounding box:", (left, top, right, bottom))
            cv2.imwrite("debug_info_2.png", img2)
            print("Cropped image saved as debug_info_2.png")

        return img2


def finalize(img2, h_scale, v_scale, output_path):

    img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _,mask = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    mask = mask / 255

    # create PNG image with transparent background
    # Convert to RGBA format
    # img2  = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_rgba = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2RGBA)
    # Set alpha channel based on mask (where mask == 1, make transparent)
    img_rgba[:, :, 3] = (1 - mask) * 255

    dpmm = 300 /25.4
    h_scale_final =  dpmm / h_scale
    v_scale_final = dpmm / v_scale
    final_width = int(img_rgba.shape[1] * h_scale_final)
    final_height = int(img_rgba.shape[0] * v_scale_final)
    # scale the image
    img_rgba = cv2.resize(img_rgba, (final_width, final_height), interpolation=cv2.INTER_CUBIC)
    from PIL import Image
    # Convert to PIL Image and save
    img_pil = Image.fromarray(img_rgba)
    img_pil.save(output_path, dpi=(300,300))
    if DEBUG:
        print("Final image shape (with scale):", img_rgba.shape)

## %%

def process(input_path, output_path):

    image = cv2.imread(input_path)
    img1, (h_scale, v_scale) = detect_rulers_and_crop(image)
    img2 = crop_to_content(img1)
    finalize(img2, h_scale, v_scale, output_path=output_path)
    image = cv2.imread(input_path)
    img1, (h_scale, v_scale) = detect_rulers_and_crop(image)
    img2 = crop_to_content(img1)
    finalize(img2, h_scale, v_scale, output_path=output_path)


def main(input_path, output_path, overwrite=False):

    if not os.path.exists(input_path):
        print("Input file does not exist.")
        return
    
    if os.path.exists(output_path) and not overwrite:
        # ask user if they want to overwrite
        response = input(f"Output file {output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
        
    process(input_path, output_path)

    # check output
    if os.path.exists(output_path):
        print("Output image saved to:", output_path)

## %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect ruler, crop and save image with correct scale.")
    parser.add_argument("input_path", help="Path to input image (jpeg or png).")
    parser.add_argument("output_path", help="Path to output PNG file.")
    # primary correct flag is --debug; keep --debig as an alias for backward compatibility
    parser.add_argument("-d" , "--debug", dest="debug", action="store_true",
                        help="Enable debug output (sets DEBUG=True).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists without prompting.")
    args = parser.parse_args()

    # set debug flag
    DEBUG = args.debug

    # Validate input extension
    in_ext = os.path.splitext(args.input_path)[1].lower()
    if in_ext not in ('.png', '.jpg', '.jpeg'):
        print("Input file extension not supported. Use .jpg, .jpeg or .png.")
        raise SystemExit(1)

    out_path = args.output_path
    # Ensure output is PNG (user requested check for '.png' in output_path)
    if not out_path.lower().endswith('.png'):
        print("Output path does not end with '.png'. Appending '.png' to output path.")
        out_path = out_path + '.png'

    main(args.input_path, out_path, overwrite=args.overwrite)
## %%
