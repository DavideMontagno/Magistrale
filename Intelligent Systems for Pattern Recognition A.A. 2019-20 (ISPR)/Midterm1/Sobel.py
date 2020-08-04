from skimage.exposure import rescale_intensity
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


#image: gray scale, #kernel: Sobel Filter or Blur
def convolutionOperation(image, kernel):
	# define the dimension of image and kernel
	(iH, iW) = image.shape[:2] #same as image.shape
	(kH, kW) = kernel.shape[:2] #same as kernel.shape
	# create output image taking care of image borders
	pad = (kW - 1) // 2
	#cv.imshow("Original",image) #show the original image 
	image = cv.copyMakeBorder(image, pad, pad, pad, pad,
		cv.BORDER_REPLICATE) #replicate borders if i'm in a border 

	output = np.zeros((iH, iW), dtype="float32") #create output image as original image
	
	# slinding kernel over the image
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):

			# extracting the center region of the current (x, y)-coordinates
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			
			# perform convolution by element-wise multiplication 
			D = np.zeros((kH, kW), dtype="float32")
			for i in range(kH):
				for j in range(kW):
					D[i][j] = np.dot(roi[i][j],kernel[i][j]) # the same with --- roi[i][j] * kernel[i][j] ---

			# store the convolved value in the center coordinate of the output image
			
			#compute sum by myself
			'''
			sum = 0
			D_y, D_x = D.shape[:2] #get dimension for the result
			for i in range(D_y):
				for j in range(D_x):
					sum = sum + D[i][j] #sum all pixel to get the result
			output[y - pad, x - pad] = sum
			'''
			output[y - pad, x - pad] = D.sum()

	
 	# rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")
	# return the output image
	return output           
           
    

blur = np.ones((3, 3), dtype="float") * (1.0 / (3 * 3))

# construct the Sobel x-axis kernel
sobelX = np.array((
	[+1, 0, -1],
	[+2, 0, -2],
	[+1, 0, -1]), dtype="int")
# construct the Sobel y-axis kernel
sobelY = np.array((
	[+1, +2, +1],
	[0, 0, 0],
	[-1, -2, -1]), dtype="int")


images = ['face4','tree3']


for image in images:
	for i in range(0,2):
		#take each image
		gray = cv.cvtColor(cv.imread("./analyze/"+image+".bmp"), cv.COLOR_BGR2GRAY)
		if(i==0):
			gray = convolutionOperation(gray, blur) #apply filter blur
			
		#compute convolution
		intensityX = convolutionOperation(gray, sobelX) #gradient X
		intensityY = convolutionOperation(gray, sobelY) #gradient Y
		own_magnitude = np.hypot(intensityX, intensityY) #magnitude
		gradientDirection = np.arctan(np.divide(intensityY,intensityX)) #direction gradient
		initialGradient = np.hypot(convolutionOperation(gray, sobelX*(1/8)),convolutionOperation(gray, sobelY*(1/8))) #initial gradient
		directionFinalMe = cv.phase(np.float64(intensityX),np.float64(intensityY),angleInDegrees=True) #take the gradient direction (in Degree)

		#save file
		if(i==0):
			cv.imwrite("./results_blur/decomposte/"+image+"_Sobel_X_blur.jpg",intensityX )
			cv.imwrite("./results_blur/decomposte/"+image+"_Sobel_Y_blur.jpg", intensityY)
			cv.imwrite("./results_blur/decomposte/"+image+"_Sobel_X-SG_blur.jpg",convolutionOperation(gray, sobelX*(1/8)))
			cv.imwrite("./results_blur/decomposte/"+image+"_Sobel_Y-SG_blur.jpg", convolutionOperation(gray, sobelY*(1/8)))
			cv.imwrite("./results_blur/intere/"+image+"_Sobel_Filter_blur.jpg", np.float32(own_magnitude))
			cv.imwrite("./results_blur/intere/"+image+"_SG_blur.jpg",np.float32(initialGradient))
			plt.imsave("./directions/"+image+"_angles_blur.jpg",directionFinalMe)
			plt.close()
			
		else:
			cv.imwrite("./results/decomposte/"+image+"_Sobel_X.jpg",intensityX )
			cv.imwrite("./results/decomposte/"+image+"_Sobel_Y.jpg", intensityY)
			cv.imwrite("./results/decomposte/"+image+"_Sobel_X-SG.jpg",convolutionOperation(gray, sobelX*(1/8) ))
			cv.imwrite("./results/decomposte/"+image+"_Sobel_Y-SG.jpg", convolutionOperation(gray, sobelY*(1/8)))
			cv.imwrite("./results/intere/"+image+"_Sobel_Filter.jpg", np.float32(own_magnitude))
			cv.imwrite("./results/intere/"+image+"_SG.jpg",np.float32(initialGradient))
			plt.imsave("./directions/"+image+"_angles.jpg",directionFinalMe)
			plt.close()
		
	