import pywt 
import matplotlib.pyplot as plt
import imageio
import numpy as np

#  (cA, (cH, cV, cD))
#        <--->
# | cA(LL) | cH(LH) |
# | cV(HL) | cD(HH) |
#    
#• a - LL, low-low coefficients
#• h - LH, low-high coefficients
#• v - HL, high-low coefficients
#• d - HH, high-high coefficients

#For 2D decomposition ('aa', 'ad', 'da', 'dd')
# arr[coeff_slices[0]] is (LL, 'aa')
# arr[coeff_slices[n]['x']] is the n level detail coefficients where x ='ad', 'da' or 'dd' for 2D case. 

def PrintReconstructions(coeffs, n):
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    #Removing Details
    for i in range(n,len(coeff_slices)):
        arr[coeff_slices[i]['ad']] = 0
        arr[coeff_slices[i]['dd']] = 0
        arr[coeff_slices[i]['da']] = 0
        
    D1 = pywt.array_to_coeffs(arr, coeff_slices)
    dCat = pywt.waverecn(D1, wavelet)
    
    plt.figure()
    plt.title('Reconstructed with level %i of details' %(n-1))
    plt.imshow(dCat,cmap=colormap)
    return

colormap=plt.get_cmap('gray')
cat = imageio.imread('/home/az/Desktop/Wavelets/Imagem/Im.jpg')
cat = cat[:,:,0]

plt.figure()
plt.title('Original Image')
plt.imshow(cat,cmap=colormap)

wavelet = 'db2'
lv = 7
coeffs = pywt.wavedecn(cat, wavelet, level=lv)

arr, coeff_slices = pywt.coeffs_to_array(coeffs)

for n in range(1,len(coeff_slices)):
    PrintReconstructions(coeffs,n)

plt.figure()
vec = [np.linalg.norm(arr[coeff_slices[0]])]
for i in range(1,7):
    vec.append(np.linalg.norm(arr[coeff_slices[i]['dd']]))

vec = vec/np.linalg.norm(vec)

plt.plot([0,1,2,3,4,5,6], vec, 'o')
plt.grid()
plt.show()
