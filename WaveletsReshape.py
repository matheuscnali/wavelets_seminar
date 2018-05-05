import pywt 
import matplotlib.pyplot as plt
import imageio
import sys
from scipy import misc
from scipy.io import wavfile
from PIL import Image
from scipy import ndimage
from pywt import wavedec
import numpy as np

def exclui_detalhes(coeffs):
    print("Coeffs:\n",coeffs[:2])
    
    novo_coeffs = coeffs[:1]
   # print("CoeffsN:\n", novo_coeffs)
    for cH,cV,cD in coeffs[1:]:novo_coeffs.append((cH,cV,cD*0))
    return novo_coeffs

def plot_detalhes(coeffs,n):
    plt.figure()
    plt.title('detalhes do nível %d'%n)
    plt.imshow(coeffs[n][2].reshape(len(coeffs[n][0]),len(coeffs[n][1])),cmap=colormap)

colormap=plt.get_cmap('gray')
cat = imageio.imread('/home/az/Desktop/Wavelets/Imagem/Im.jpg')
cat = cat[:,:,0]
#
plt.figure()
plt.title('Original Image')
plt.imshow(cat,cmap=colormap)

wavelet = 'db2'
lv = 7
coeffs = pywt.wavedec2(cat, wavelet, level=lv)

#arr, coeff_slices = pywt.coeffs_to_array(coeffs)
#
#for n in range(1,len(coeff_slices)):
#    PrintReconstructions(coeffs,n)

for i in range(lv-2):
    plot_detalhes(coeffs,i+1)
    
#plt.figure()
plt.title('Reconstituída sem os detalhes')
cat2 = pywt.waverec2(remove_details(arr, coeff_slices),wavelet)
plt.imshow(cat2,cmap=colormap)





















#
#
#
#
#
#
###########    catR = pywt.waverecn(NoDetcoeffs, wavelet)
#    plt.imshow(catR,cmap=colormap)################################3
# -Wavelet list of a Family
#print("\n",pywt.wavelist('db'))
#
# -Informations about haar Wavelet
#w = pywt.Wavelet('ga')
#print(w)
#
# -Filter Bank (w.dec_lo, w.dec_hi, w.rec_lo, w.rec_hi)
# -Lowpass and highpass decomposition filters.
# -Lowpass and highpass reconstruction filters.
#
#w.filter_bank == (w.dec_lo, w.dec_hi, w.rec_lo, w.rec_hi)
#
# -Number of vanishing moments for the scaling function phi (vanishing_moments_phi)
# -and the wavelet function psi (vanishing_moments_psi) associated with the filters.
#print(w.vanishing_moments_phi)
#print(w.vanishing_moments_psi)
#
## -Wavefun!
#fs, data = wavfile.read('/home/az/Desktop/Wavelets/Audio/1.wav')
##data = data[len(data)//10:(len(data)//10)+1000]
##w = pywt.Wavelet('db2')
##(phi, psi, x) = w.wavefun(level=10)
##plt.plot(np.arange(len(psi)), psi)
##plt.plot(np.arange(len(phi)), phi)
#
# -Example of discrete Wavelet transform using haar Wavelet
#
##x = [1,5,9,8,7,2,5,6,3,2,1,5,9,9,8,8,7,4,5,2,1]
#coeffs = wavedec(data.T[0], 'db20', level=2)
##cA, cD = data2 = pywt.dwt(data, 'sym2')
#cA2, cD2, cD1 = coeffs
#
#plt.figure()
##plt.xlim(0,1000)
#plt.plot(np.arange(len(data.T[0])), data.T[0])
##
##plt.figure()
###plt.xlim(0,250)
##plt.plot(np.arange(len(cA4)), cA4)
##
##plt.figure()
###plt.xlim(0,250)
##plt.plot(np.arange(len(cD4)), cD4)
###cA2,cD2,cD1 = coeffs
##
##plt.figure()
###plt.xlim(0,500)
##plt.plot(np.arange(len(cD3)), cD3)
#
#datarec = pywt.waverec([cA2, cD2, cD1], 'db20')
#plt.figure()
#plt.plot(np.arange(len(datarec)), datarec)
#
##wavfile.write('1aprrox.wav',44100 ,datarec)
##print("Approximation Coefficients:",cA,"\nApproximation Details:",cD) 
## -Approximation Coefficients
##plt.plot(np.arange(len(cA)), cA)
## -Details Coefficients
##plt.plot(np.arange(len(cD)), cD)
#
## -Now, the inverse:
##print(pywt.idwt(cA, cD, 'haar'))
