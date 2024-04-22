import os
import cv2
import numpy as np
import random

numSamplesPerCase = 50
patchSize = 256

# Prompt the user for input folder paths
inputImgFolder1 = input("Enter the input image folder path 1: ")
inputImgFolder2 = input("Enter the input image folder path 2: ")

# Prompt the user for the output folder path
outputFolder = input("Enter the output folder path: ")

files = [file for file in os.listdir(inputImgFolder1) if file.endswith('_snapshot.png')]


for i, sampleName in enumerate(files, start=1):
    print(sampleName)
    im = cv2.imread(os.path.join(inputImgFolder1, sampleName))
    labelName = sampleName.replace('_snapshot.png', '_labels.png')
    im2 = cv2.imread(os.path.join(inputImgFolder2, labelName))
    mask = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) > 200
    indices = np.where(mask)

    if len(indices[0]) == 0:
        continue
    elif len(indices[0]) < numSamplesPerCase:
        numSamplesPerCaseI = len(indices[0])
    else:
        numSamplesPerCaseI = numSamplesPerCase

    # Get the size of the current image
    img_height, img_width, _ = im.shape

    rand_indices = random.sample(range(len(indices[0])), numSamplesPerCaseI)
    
    for j in rand_indices:
        ri, ci = indices[0][j], indices[1][j]
        
        if ri + patchSize - 1 > img_height or ci + patchSize - 1 > img_width:
            continue

        patchi = im[ri:ri+patchSize, ci:ci+patchSize]
        patchi2 = im2[ri:ri+patchSize, ci:ci+patchSize]

        # Calculate the average color of the patch
        avg_color = np.mean(patchi)

        # Define a threshold for "mostly white" (adjust as needed)
        white_threshold = 200  # You can adjust this threshold

        if avg_color < white_threshold:
            cv2.imwrite(os.path.join(outputFolder, 'im', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}.png'), patchi)
            cv2.imwrite(os.path.join(outputFolder, 'mask', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}.png'), patchi2)

            # Rest of your augmentations (Reflection, Rotation, Jitter, Blur)
            cv2.imwrite(os.path.join(outputFolder, 'im', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}.png'), patchi)
            cv2.imwrite(os.path.join(outputFolder, 'mask', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}.png'), patchi2)

            # Reflection
            tform_refl = cv2.getRotationMatrix2D((patchSize / 2, patchSize / 2), 0, 1)
            imAugmented_relf1 = cv2.warpAffine(patchi, tform_refl, (patchSize, patchSize))
            imAugmented_relf2 = cv2.warpAffine(patchi2, tform_refl, (patchSize, patchSize))
            cv2.imwrite(os.path.join(outputFolder, 'im', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}_refl.png'), imAugmented_relf1)
            cv2.imwrite(os.path.join(outputFolder, 'mask', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}_refl.png'), imAugmented_relf2)

            # Rotation
            angle = random.uniform(0, 90)
            tform_rot = cv2.getRotationMatrix2D((patchSize / 2, patchSize / 2), angle, 1)
            imAugmented_rot1 = cv2.warpAffine(patchi, tform_rot, (patchSize, patchSize))
            imAugmented_rot2 = cv2.warpAffine(patchi2, tform_rot, (patchSize, patchSize))
            cv2.imwrite(os.path.join(outputFolder, 'im', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}_rot.png'), imAugmented_rot1)
            cv2.imwrite(os.path.join(outputFolder, 'mask', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}_rot.png'), imAugmented_rot2)

            # Rotation 2
            angle2 = random.uniform(-90, 0)
            tform_rot2 = cv2.getRotationMatrix2D((patchSize / 2, patchSize / 2), angle2, 1)
            imAugmented_rot12 = cv2.warpAffine(patchi, tform_rot2, (patchSize, patchSize))
            imAugmented_rot22 = cv2.warpAffine(patchi2, tform_rot2, (patchSize, patchSize))
            cv2.imwrite(os.path.join(outputFolder, 'im', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}_rot2.png'), imAugmented_rot12)
            cv2.imwrite(os.path.join(outputFolder, 'mask', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}_rot2.png'), imAugmented_rot22)

            # Rotation 3
            angle3 = random.uniform(180, 270)
            tform_rot3 = cv2.getRotationMatrix2D((patchSize / 2, patchSize / 2), angle3, 1)
            imAugmented_rot13 = cv2.warpAffine(patchi, tform_rot3, (patchSize, patchSize))
            imAugmented_rot23 = cv2.warpAffine(patchi2, tform_rot3, (patchSize, patchSize))
            cv2.imwrite(os.path.join(outputFolder, 'im', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}_rot3.png'), imAugmented_rot13)
            cv2.imwrite(os.path.join(outputFolder, 'mask', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}_rot3.png'), imAugmented_rot23)

            # Jitter Brightness
            hsv_patchi = cv2.cvtColor(patchi, cv2.COLOR_BGR2HSV)
            hsv_patchi[:, :, 2] = np.clip(hsv_patchi[:, :, 2] - random.uniform(0.1, 0.1), 0, 255)
            imJittered_bright = cv2.cvtColor(hsv_patchi, cv2.COLOR_HSV2BGR)
            cv2.imwrite(os.path.join(outputFolder, 'im', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}_bright.png'), imJittered_bright)
            cv2.imwrite(os.path.join(outputFolder, 'mask', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}_bright.png'), patchi2)

            # Jitter Contrast
            hsv_patchi = cv2.cvtColor(patchi, cv2.COLOR_BGR2HSV)
            hsv_patchi[:, :, 1] = np.clip(hsv_patchi[:, :, 1] * random.uniform(1.2, 1.4), 0, 255)
            imJittered_contrast = cv2.cvtColor(hsv_patchi, cv2.COLOR_HSV2BGR)
            cv2.imwrite(os.path.join(outputFolder, 'im', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}_contrast.png'), imJittered_contrast)
            cv2.imwrite(os.path.join(outputFolder, 'mask', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}_contrast.png'), patchi2)

            # Blur
            sigma = 1 + 1.1 * random.random()
            imBlurred = cv2.GaussianBlur(patchi, (0, 0), sigma)
            cv2.imwrite(os.path.join(outputFolder, 'im', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}_blur.png'), imBlurred)
            cv2.imwrite(os.path.join(outputFolder, 'mask', sampleName.replace(' ', '_') + f'_r{ri}_c{ci}_blur.png'), patchi2)

    print(i)
