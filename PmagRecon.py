#Euler Pole addition function to be used in python

import pmag, pmagplotlib
import pylab
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def EulerADD(Elat1, Elong1, Eang1, Elat2, Elong2, Eang2):
	'''
	Calculates an Euler pole from the addition of two Euler poles. ORDER MATTERS!!!!!!!!!!!!!
	Elat1,Elong1,Eang1 = the values of the first Euler, all given in degrees
	Elat2,Elong2,Eang2 = the values of the second Euler, all given in degrees
	'''
	#Eulers are defined by a Latitude, Longitude, and rotation amount in degrees (CCW being positive)
	M1,M2 = np.zeros((3,1)), np.zeros((3,1))
	M1[0] = Elat1
	M1[1] = Elong1
	M1[2] = Eang1
	M2[0] = Elat2
	M2[1] = Elong2
	M2[2] = Eang2

	Eu1 = M1
	Eu2 = M2
	Eul1,Eul2,Eul3 = np.zeros((3,1)),np.zeros((3,1)),np.zeros((3,1))   

	Eul1[0] = Eu1[0]*np.pi/180. #convert units to radians
	Eul1[1] = Eu1[1]*np.pi/180.
	Eul1[2] = Eu1[2]*np.pi/180.
	Eul2[0] = Eu2[0]*np.pi/180.
	Eul2[1] = Eu2[1]*np.pi/180.
	Eul2[2] = Eu2[2]*np.pi/180.

	Eul1c,Eul2c,Eul3c = np.zeros((3,1)),np.zeros((3,1)),np.zeros((3,1))
	#convert location of Euler poles into Cartesian from Lat/Long
	Eul1c[0] = np.cos(Eul1[0])*np.cos(Eul1[1])
	Eul1c[1] = np.cos(Eul1[0])*np.sin(Eul1[1])
	Eul1c[2] = np.sin(Eul1[0])
	Eul2c[0] = np.cos(Eul2[0])*np.cos(Eul2[1])
	Eul2c[1] = np.cos(Eul2[0])*np.sin(Eul2[1])
	Eul2c[2] = np.sin(Eul2[0])
	#Make matrices out of cartesian Euler poles
	Eul1m,Eul2m,Eul3m = np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))
	Eul1m[0,0] = (Eul1c[0]**2.)*(1.-np.cos(Eul1[2]))+np.cos(Eul1[2])
	Eul1m[1,1] = (Eul1c[1]**2.)*(1.-np.cos(Eul1[2]))+np.cos(Eul1[2])
	Eul1m[2,2] = (Eul1c[2]**2.)*(1.-np.cos(Eul1[2]))+np.cos(Eul1[2])
	Eul1m[0,1] = Eul1c[0]*Eul1c[1]*(1.-np.cos(Eul1[2]))-Eul1c[2]*np.sin(Eul1[2])
	Eul1m[0,2] = Eul1c[0]*Eul1c[2]*(1.-np.cos(Eul1[2]))+Eul1c[1]*np.sin(Eul1[2])
	Eul1m[1,0] = Eul1c[1]*Eul1c[0]*(1.-np.cos(Eul1[2]))+Eul1c[2]*np.sin(Eul1[2])
	Eul1m[1,2] = Eul1c[1]*Eul1c[2]*(1.-np.cos(Eul1[2]))-Eul1c[0]*np.sin(Eul1[2])
	Eul1m[2,0] = Eul1c[2]*Eul1c[0]*(1.-np.cos(Eul1[2]))-Eul1c[1]*np.sin(Eul1[2])
	Eul1m[2,1] = Eul1c[2]*Eul1c[1]*(1.-np.cos(Eul1[2]))+Eul1c[0]*np.sin(Eul1[2])
	Eul2m[0,0] = (Eul2c[0]**2.)*(1.-np.cos(Eul2[2]))+np.cos(Eul2[2])
	Eul2m[1,1] = (Eul2c[1]**2.)*(1.-np.cos(Eul2[2]))+np.cos(Eul2[2])
	Eul2m[2,2] = (Eul2c[2]**2.)*(1.-np.cos(Eul2[2]))+np.cos(Eul2[2])
	Eul2m[0,1] = Eul2c[0]*Eul2c[1]*(1.-np.cos(Eul2[2]))-Eul2c[2]*np.sin(Eul2[2])
	Eul2m[0,2] = Eul2c[0]*Eul2c[2]*(1.-np.cos(Eul2[2]))+Eul2c[1]*np.sin(Eul2[2])
	Eul2m[1,0] = Eul2c[1]*Eul2c[0]*(1.-np.cos(Eul2[2]))+Eul2c[2]*np.sin(Eul2[2])
	Eul2m[1,2] = Eul2c[1]*Eul2c[2]*(1.-np.cos(Eul2[2]))-Eul2c[0]*np.sin(Eul2[2])
	Eul2m[2,0] = Eul2c[2]*Eul2c[0]*(1.-np.cos(Eul2[2]))-Eul2c[1]*np.sin(Eul2[2])
	Eul2m[2,1] = Eul2c[2]*Eul2c[1]*(1.-np.cos(Eul2[2]))+Eul2c[0]*np.sin(Eul2[2])
	
	#Multiply matricies (Or arrays, rather, use sp.dot for correct treatment of arrays as matricies)
	#Eul3m = sp.dot(Eul1m,Eul2m) #Something is not correctly calculated using this function...am I mixing arrays and matricies?
	#Multiply matricies manually instead:
	Eul3m[0,2] = Eul2m[0,0]*Eul1m[0,2]+Eul2m[0,1]*Eul1m[1,2]+Eul2m[0,2]*Eul1m[2,2] 
	Eul3m[0,0] = Eul2m[0,0]*Eul1m[0,0]+Eul2m[0,1]*Eul1m[1,0]+Eul2m[0,2]*Eul1m[2,0]
	Eul3m[0,1] = Eul2m[0,0]*Eul1m[0,1]+Eul2m[0,1]*Eul1m[1,1]+Eul2m[0,2]*Eul1m[2,1]
	Eul3m[1,0] = Eul2m[1,0]*Eul1m[0,0]+Eul2m[1,1]*Eul1m[1,0]+Eul2m[1,2]*Eul1m[2,0]
	Eul3m[1,1] = Eul2m[1,0]*Eul1m[0,1]+Eul2m[1,1]*Eul1m[1,1]+Eul2m[1,2]*Eul1m[2,1]
	Eul3m[1,2] = Eul2m[1,0]*Eul1m[0,2]+Eul2m[1,1]*Eul1m[1,2]+Eul2m[1,2]*Eul1m[2,2]
	Eul3m[2,0] = Eul2m[2,0]*Eul1m[0,0]+Eul2m[2,1]*Eul1m[1,0]+Eul2m[2,2]*Eul1m[2,0]
	Eul3m[2,1] = Eul2m[2,0]*Eul1m[0,1]+Eul2m[2,1]*Eul1m[1,1]+Eul2m[2,2]*Eul1m[2,1]
	Eul3m[2,2] = Eul2m[2,0]*Eul1m[0,2]+Eul2m[2,1]*Eul1m[1,2]+Eul2m[2,2]*Eul1m[2,2]

	#Convert Euler matrix back to Euler pole
	sqrt_term = np.sqrt((Eul3m[2,1]-Eul3m[1,2])**2+(Eul3m[2,0]-Eul3m[0,2])**2+(Eul3m[1,0]-Eul3m[0,1])**2)
	Eul3c[0] = np.arcsin((Eul3m[1,0]-Eul3m[0,1])/sqrt_term)
	Eul3c[1] = np.arctan((Eul3m[0,2]-Eul3m[2,0])/((Eul3m[2,1]-Eul3m[1,2])))
	Eul3c[2] = np.arctan(sqrt_term/(Eul3m[0,0]+Eul3m[1,1]+Eul3m[2,2]-1))
	#Convert radians back to degrees
	Eul3 = np.zeros((3,1))
	Eul3[0] = Eul3c[0]*180./np.pi
	Eul3[1] = Eul3c[1]*180./np.pi
	Eul3[2] = Eul3c[2]*180./np.pi

	#Deal with polarity and negative long/angle
	if Eul3[2] < 0:
	    Eul3[2] = Eul3[2]+180
	if Eul3m[2,1]-Eul3m[1,2] > 0:
	    Eul3[1] = np.mod(Eul3[1],360)
	else:
	    Eul3[1] = np.mod(Eul3[1]+180,360)
        
	print('Resulting "Third" Euler Pole')
	print(Eul3)


    
    
