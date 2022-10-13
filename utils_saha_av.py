#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:23:41 2021

@author: jcampbellwhite001
"""

from numpy import *
from scipy.special import gamma
from matplotlib import *
from matplotlib.pyplot import *
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import optimize
#from meanclip import meanclip
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
from scipy.stats import spearmanr
from astropy.io import fits
import time
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import utils_shared_variables as USH


starttime=time.time()

def saha_av(em_lines_fit,N_range=[3,15],T_range=[1000,15500]):
    #subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.97, wspace=0.1, hspace=0.05)
    matplotlib.rc('font', family='serif',size=14)
    
    #----------------------------------------------
    
    #natural constants
    clightkms=299792. #km/s
    clight=29979245800.0 #cm/s 
    eVtoK=1.16045221e4	#from NIST conversion from eV to K
    kboltzmann= 1.3806e-16	#erg/K
    hplanck=6.626e-27	#erg s
    me=9.109e-28	#electron mass in g
    xi_Fe= 7.9024681	#ionization potential of Fe in eV
    xi_He=24.58741
    xiFe=xi_Fe*eVtoK	#ionization potential of Fe in K
    xiHe=xi_He*eVtoK	#ionization potential of He in K
    
    #numberobs=3	#number of observations
    
    #minaki=1e5	#Minimum Aki to be considered
    
    #new: here I put the N and T ranges so I can find them easier.
    N=10**arange(3,15,0.05)	#for real
    T=arange(1000.,12500.,25.)
    #N=10**arange(3,15,0.5) #
    #T=arange(1000.,12500.,500.)
    
    N=10**arange(N_range[0],N_range[1],0.3)	
    T=arange(T_range[0],T_range[1],25)
    
    #read in dataframe generated from STAR_MELT gauss_stats()
    # D = pd.read_csv('saha_av_df_test.csv',delimiter=',',index_col=0)
    
    # #min Aki
    # D.drop(D[D.Aki < 1e5].index, inplace=True)
    
    # D.drop(D[D.int_flux > 100].index, inplace=True)
    
    # D.drop(D[D.w0==8217.48].index, inplace=True)
    
    # D.drop(D[D.element=='Fe II'].index, inplace=True)
    
    D=em_lines_fit
    
    #remove any duplicate entries for the same w0 and mjd
    D.drop_duplicates(subset=['w0','mjd'], inplace=True)
    print('Saha calculation for',D.element[0])
    if D.element[0] != 'Fe' and D.element[0] != 'He':
        print('ERROR, only have partition fn for Fe or He')
    
    if D.element[0]=='Fe':
        xi=xiFe
    elif D.element[0]=='He':
        xi=xiHe
    
    #only keep lines that have data for all dates given in D
    numberobs=len(unique(D.mjd))
    value_counts = D['w0'].value_counts()
    to_remove = value_counts[value_counts < numberobs].index
    D = D[~D.w0.isin(to_remove)]
    dates=D.mjd.to_numpy()
    
    #create df of the fluxes and the erros for each wl
    int_fluxs=D.pivot(index='w0',columns='mjd',values='int_flux').to_numpy()
    errors=D.pivot(index='w0',columns='mjd',values='gof').to_numpy()
    
#     if D['element'].str.contains('FeI').any() == False:
#         D['ele'] = np.where(
#             D['sp_num'] == 1, 'FeI', np.where(
#             D['sp_num'] == 2, 'FeII', -1)) 
#     else:
#         D['ele']=D['element']
    
    #keep one of each line for theoretical caluclations
    D_lines=D.drop_duplicates(subset=['w0']).sort_values('w0')
    
    ele=D_lines.element.to_numpy()
    sp=D_lines.sp_num.to_numpy()
    wl=D_lines.w0.to_numpy()
    aki=D_lines.Aki.to_numpy()
    gi=D_lines.g_k.to_numpy()
    gj=D_lines.g_i.to_numpy()
    ej=D_lines.Ei.to_numpy()
    qual=D_lines.Acc.to_numpy()
    
    # t1=int_fluxs[:,0]
    # e1=errors[:,0]
    # t2=int_fluxs[:,1]
    # e2=errors[:,1]
    # t3=int_fluxs[:,2]
    # e3=errors[:,2]
    
    t1=int_fluxs
    e1=errors
    
    
    qualo=zeros([size(wl)])
    
    for i in range(0, size(wl)):
        if qual[i]=='AAA':
            qualo[i]=0.01
        if qual[i]=='A+':
            qualo[i]=0.02
        if qual[i]=='AA':
            qualo[i]=0.02
        if qual[i]=='A':
            qualo[i]=0.03
        if qual[i]=='B+':
            qualo[i]=0.07
        if qual[i]=='B':
            qualo[i]=0.1
        if qual[i]=='C+':
            qualo[i]=0.18
        if qual[i]=='C':
            qualo[i]=0.25
        if qual[i]=='D+':
            qualo[i]=0.4
        if qual[i]=='D':
            qualo[i]=0.5
        if qual[i]=='E':
            qualo[i]=0.5
        else:
            qualo[i]=0.1
    
    #From now on, I only work with the "faint" lines in the hope they won't be saturated.
    
    #Calculate the scale factor: since data are not calibrated.
    
    # scal=mean([t1,t2,t3])
    # scal1=mean(t1)
    # scal2=mean(t2)
    # scal3=mean(t3)
    
    # t1s=t1/scal1
    # e1s=e1/scal1
    # t2s=t2/scal2
    # e2s=e2/scal2
    # t3s=t3/scal3
    # e3s=e2/scal3
    
    scal=mean(t1)
    t1s=t1/scal
    e1s=e1/scal
    
    
    """errorbar(wl,t1s,yerr=e1s, fmt='b.',ecolor='b')
    errorbar(wl,t2s,yerr=e2s, fmt='r.',ecolor='r')
    errorbar(wl,t3s,yerr=e3s, fmt='g.',ecolor='g')
    errorbar(wl,t4s,yerr=e4s, fmt='m.',ecolor='m')
    errorbar(wl,t5s,yerr=e5s, fmt='c.',ecolor='c')
    errorbar(wl,t6s,yerr=e6s, fmt='k.',ecolor='k')
    errorbar(wl,t7s,yerr=e7s, fmt='w.',ecolor='grey')
    errorbar(wl,t8s,yerr=e8s, fmt='y.',ecolor='Orange')"""
    
    
    #-----------------------------
    
    #Read in partition functions, taken from NIST, for Fe I and Fe II:
    #The partition function is trickier, so I am using some values from NIST
    #measured for various T and then interpolating
    #You get the values from NIST level form https://physics.nist.gov/PhysRefData/ASD/levels_form.html
    
    #For Fe I
    t_part_eV=[0.05,0.1,0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0,4.5,6,]
    t_part=transpose(transpose(t_part_eV)*eVtoK)	#transposing or it complains in unit conversion
    z_part_FeI=[12.79,16.57,21.71,30.89,47.99,78.54,127.43,196.65,388.89,628.86,889.55, 1152.77, 1408.13,1650.38 , 2284.12] 
    z_part_HeI=[1,1,1,1,1,1,1,1,1.01,1.08,1.42,2.33,4.16,7.18,24.66]
    
    if D.element[0]=='Fe':
        U=interpolate.interp1d(t_part,z_part_FeI)
    elif D.element[0]=='He':
        U=interpolate.interp1d(t_part,z_part_HeI)
        
    
    #For Fe II
    t_part_eV=[0.05,0.1,0.25,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,6]
    t_part=transpose(transpose(t_part_eV)*eVtoK)	#transposing or it complains in unit conversion
    z_part_FeII=[15.00,21.19,33.78,46.73,76.51,121.10,184.22,270.66,381.79,515.11,666.02,829.49, 1353.32] #FeII
    z_part_HeII=[2,2,2,2,2,2,2,2,2,2,2,2.01,2.19]
    
    if D.element[0]=='Fe':
        U1=interpolate.interp1d(t_part,z_part_FeII)
    elif D.element[0]=='He':
        U1=interpolate.interp1d(t_part,z_part_HeII)
    
    #Define the functions that I need here:
    
    def B_einst(Aul,gu,gl,wavelength):		#remember I am giving lambda in AA
    	return Aul*(gu/gl)*(wavelength*1e-8)**3/(2*hplanck*clight)
    
    def saha(t, ne, xj):	#gives the ratio nj+1/nj, assumes xj is in K or needs a kboltzmann*t
    	return  (2*np.pi*me*kboltzmann*t/hplanck**2)**(3./2.)*(1/ne)*(2*U1(t)/U(t))*exp(-xj/t)
    
    def boltzmann(t,gi,gj,xj):	#gives the ratio between i and j, assumes xj is in K or needs a kboltzmann*t
    	return ((1.0*gi)/(1.0*gj))* exp(-xj/t)
    
    def opthinratio(a1,a2,l1,l2):	#optically thin limit of the intensity ratio for lines with wavelentths l1,l2 and aki a1, a2 respectively
    	return (a1*l2)/(a2*l1)  #intensity ratio of line 1/ line2
    
    #All the ratios are relative. I need to give them with respect to the lowest 
    
    #Minimum for the Fe I transitions
    fol=(sp!=2)
    try:
    	ej_feI_min=min(ej[fol])
    except(ValueError):
    	ej_feI_min=nan
    #and for the Fe II transitions:
    fil=(sp==2)
    try:
    	ej_feII_min=min(ej[fil])
    except(ValueError):
    	ej_feII_min=nan
    
    #Now what I need for the Saha part is the ionization potential (OK, that's xiFe in K or xi in eV)
    #but for the Boltzmann relation I need the excitation between the levels.
    #This is relative, so I need to choose one (e.g. the lowest ionization potential for upper level)
    #and make them all relative to this one.
    
    #The energy of the upper level is the lower level plus hnu=h*c/lambda
    eup= ej + (hplanck*clight/(wl*1.e-8))*(1/(kboltzmann*eVtoK))
    
    #Then I get the minimum excitation energy, which is the one I can use as a reference for Boltzmann.
    fol=(sp!=2)
    try:
    	ei_feI_min=min(eup[fol])
    except(ValueError):
    	ei_feI_min=nan
    #and for the Fe II transitions:
    fil=(sp==2)
    try:
    	ei_feII_min=min(eup[fil])
    except(ValueError):
    	ei_feII_min=nan
    
    #Note: this means that at the end everything will be in terms of the lowest energy level
    #but since I will normalize things again, that should be fine.
    #Thus, the excitation energy relative to the lower one:
    
    x_rel=zeros([size(wl)])
    nFeII=0.	#to count the number of II
    
    for i in range(0, size(wl)):
    	if sp[i]==2:
    		x_rel[i]= (eup[i]-ei_feII_min)*eVtoK
    		nFeII=nFeII+1.
    	else:
    		x_rel[i]=(eup[i]-ei_feI_min)*eVtoK
    
    #-----------------------------
    
    #Now, for all the densities and temperatures in the array, calculate the result.
    #I call it ratioline although it is the non-scaled flux per line.
    
    ratiolines=zeros([size(wl), size(T), size(N)])	#first index is the line, second one is the T, third is N.
    fornorm=zeros([size(T), size(N)])
    
    
    for k in range(0, size(wl)):	#for the line
    	for i in range(0, size(T)):	#for the temperature
    		for j in range(0, size(N)):	#for the density
    
    			#now the line ratio is obtained by taking into account Saha (for ions), Boltzmann (for all) and the natural ratio (for all)
    			if sp[k]==2:	# and aki[k]<1e5:
    				ratiolines[k,i,j]=(aki[k]/(1.e-8*wl[k]))*saha(T[i], N[j], xi)*boltzmann(T[i],gi[k],gj[k], x_rel[k])			
    				#print 'For FeII, T, N, line ratio', wl[k],T[i], log10(N[j]), ratiolines[k,i,j]
    				#print 'Aki term, saha, boltzmann', (aki[k]/wl[k]), saha(T[i], N[j], xi), boltzmann(T[i],gi[k],gj[k], x_rel[k])
    			
    			#else			
    			if sp[k]!=2:	# and aki[k]<1e5:	#this is for the FeI, which I also use for normalization			
    				ratiolines[k,i,j]=(aki[k]/(1.e-8*wl[k]))*boltzmann(T[i],gi[k],gj[k], x_rel[k])
    				#print 'For FeI, T, N, line ratio', wl[k],T[i], log10(N[j]), ratiolines[k,i,j]
    				#print 'Aki term,  boltzmann', (aki[k]/wl[k]), boltzmann(T[i],gi[k],gj[k], x_rel[k])				
    				fornorm[i,j]=fornorm[i,j]+ratiolines[k,i,j]	#sum them all FeIs for normalization
    
    
    #Now that all the line ratios are done, I normalize them so that they are kind of reasonable:
    
    fornorm=fornorm/(1.*size(wl)-1.*nFeII)	#per line normalization, count only Fe I lines
    
    #print 'Old ratiolines', ratiolines
    
    for k in range(0, size(wl)):	#for the line
    	for i in range(0, size(T)):	#for the temperature
    		for j in range(0, size(N)):	#for the density
    			if fornorm[i,j]>0:
    				#print 'Old ratiolines', ratiolines[k,i,j]
    				ratiolines[k,i,j]=ratiolines[k,i,j]/fornorm[i,j]
    				#print 'New ratiolines', ratiolines[k,i,j]
    			else:
    				ratiolines[k,i,j]=nan
    #print 'New ratiolines', ratiolines
    #Note: The Fe I ratiolines are quite reasonable, the Fe II go crazy in the extremes.
    
    for l in range(0, size(wl)):
    	print('Range of line ratios for line lambda=', l, ele[l],int(sp[l]),np.round(wl[l],2), np.min(ratiolines[l,:,:]),'-', np.max(ratiolines[l,:,:]))
    
    # print('As of here, I have ratiolines which are the theoretical fluxes')
    # print('for each line, T, N, and normalized to some decent number.')
    # print('What I do now is to compare them with the observed ones.')
    # print('Leaving aside normalizations, the best one will be the one for which the ratios obs/theo are about the same')
    # print('which is, the std of the ratios will be smallest.')
    
    #-----------------------------
    
    #Now it is the time to compare the data and the models.
    #Note that things are still not calibrated against each other. I choose to fix one line
    #(so for this line the error will be zero) and compare them all to this one.
    #I may need to check other lines too in case this is bad.
    #Which line to choose? The one for which the data on more nights seem robust enough.
    
    # ts=[t1s,t2s,t3s]	#put all values in the same array
    # ts=transpose(ts)
    # #now, ts[:,l] is the set of all the lines for observation number l
    # es=[e1s,e2s,e3s]	#same for the errors
    # es=transpose(es)
    
    ts=t1s
    es=e1s
    
    #I do the exercise for all observations independently and for the average.
    
    ts_average=zeros([size(wl)])
    es_average=zeros([size(wl)])
    
    for k in range(0, size(wl)):
    	ts_average[k]=average(ts[k,:], weights=es[k,:])
    	es_average[k]=sqrt(average((ts[k,:]-ts_average[k])**2,weights=es[k,:])) #sqrt of weighted variance as weigthed error
    	#I checked the weigthed average vs normal average, they are not too different.
    	
    
    #This below is just to find which line is best.
    
    """std_lines=zeros([size(wl)])
    
    for i in range(0, size(wl)):
    	if (t1s[i]>0 or t2s[i]>0 or t3s[i]>0 or t4s[i]>0 or t5s[i]>0 or t6s[i]>0 or t7s[i]>0 or t8s[i]>0):	
    		#std_lines[i]=std([t1s[i],t2s[i],t3s[i],t4s[i],t5s[i],t6s[i],t7s[i],t8s[i]])/average([t1s[i],t2s[i],t3s[i],t4s[i],t5s[i],t6s[i],t7s[i],t8s[i]])
    		#Better: weighted std to account for errors:
    		aver=np.average(ts[i,:], weights=es[i,:])
    		vari=np.average((ts[i,:]-aver)**2,weights=es[i,:])
    		std_lines[i]=sqrt(vari)/aver	#normalize to the total or it's not representative!
    			
    	else:
    		std_lines[i]=1e6	#randomly big number
    
    #select best line as the most consistent of them all with aki<1e5:
    
    bestline=argmin(std_lines)	#index for the less variable line in relative std
    
    #I checked that this is a FeI line, or things will be funny when there is not enough ionization.
    
    
    print 'Bestline', sp[bestline],wl[bestline]"""
    
    ###
    
    #Now calculate the ratios:
    
    diffe=zeros([numberobs,size(wl), size(T), size(N)])
    ediffe=zeros([numberobs,size(wl), size(T), size(N)]) #to put the errors
    
    #Note that now diffe contains the ratio for each line, observed/modeled
    
    diffe_aver=zeros([size(wl), size(T), size(N)]) #and this is for the average values
    ediffe_aver=zeros([size(wl), size(T), size(N)])
    
    
    for i in range(0, size(T)):	#for the temperature
    	for j in range(0, size(N)):	#for the density
    		for k in range(0, size(wl)):	#for the line, I estimate the equivalent to chi2
    			diffe_aver[k,i,j]=ts_average[k]/ratiolines[k,i,j]
    			ediffe_aver[k,i,j]=sqrt(es_average[k]**2/ratiolines[k,i,j]**2+(qualo[k]*ts_average[k])**2/ratiolines[k,i,j]**2)
    			for l in range(0, numberobs):	
    				#if ts[k,l]>0 and es[k,l]>0:				
    				diffe[l,k,i,j]=ts[k,l]/ratiolines[k,i,j]
    				#propagated error:
    				ediffe[l,k,i,j]=sqrt(es[k,l]**2/ratiolines[k,i,j]**2+ (qualo[k]*ts[k,l])**2/ratiolines[k,i,j]**2)
    				#print "calculating difs", diffe[l,k,i,j],ediffe[l,k,i,j]				
    				#else:	#If no value available, then I divide by the max error for that one line
    				#	diffe[l,i,j]=diffe[l,i,j]+(ratiolines[k,i,j]*norma-ts[k,l])**2/max(es[k,:])**2
    				#if(aki[k]<1e5):				
    				#	diffe[l,i,j]=diffe[l,i,j]+(ratiolines[k,i,j]*norma-ts[k,l])**2
    				#else:
    				#	diffe[l,i,j]=diffe[l,i,j]
    			#print diffe[l,i,j]
    
    print('Calculated all the obs/theo ratios') #, diffe
        
    #Note: as of here, I check that both the results for the average and non-average data are
    #consistent, as I would expect. They are also consistent with the max, min values in each range.
    #The only potential issue is that the variance of the observed lines is quite high for some
    #lines, larger than the error (so we do have real line variability).
    
    """#This plot is very hard to visualize
    subplot(1,2,1)
    
    for i in range(0, size(T)):	#for the temperature
    	for j in range(0, size(N)):	#for the density
    		plot(wl,diffe[0,:,i,j], 'b-')
    		plot(wl,diffe[1,:,i,j], 'r-')
    		plot(wl,diffe[2,:,i,j], 'g-')
    		plot(wl,diffe[3,:,i,j], 'y-')
    		plot(wl,diffe[4,:,i,j], 'm-')
    		plot(wl,diffe[5,:,i,j], 'c-')
    		plot(wl,diffe[6,:,i,j], 'k-')
    		plot(wl,diffe[7,:,i,j], color='Grey')"""
    
    
    #ax = fig.gca(projection='3d')
    #Axes3D.plot_wireframe(N,T,diffe)
    #ax.plot_surface(T,N,diffe[0,:,:], cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    #now diffe is a big mattrix that contains the ratio between observed and predicted values
    #for something to be a good fit, the ratio will need to be about the same for each group of observations range(l,numberobs)
    
    stdratio=zeros([numberobs,size(T),size(N)])
    stdratio_average=zeros([size(T),size(N)])
    averatio=zeros([numberobs,size(T),size(N)]) #to keep track of the average value in case this diverges 
    
    for i in range(0, size(T)):	#for the temperature
    	for j in range(0, size(N)):	#for the density
    		#For the average values
    		a=np.average(diffe_aver[:,i,j], weights=ediffe_aver[:,i,j])		
    		a=np.median(diffe_aver[:,i,j])		
    		b=np.average((diffe_aver[:,i,j]-a)**2,weights=ediffe_aver[:,i,j])
    		aa=np.median(diffe_aver[:,i,j])		
    		stdratio_average[i,j]=sqrt(b)/aa		#Don't forget to normalize!
    		for l in range(0, numberobs):	
    			#stdratio[l,i,j]=std(diffe[l,:,i,j])/mean(diffe[l,:,i,j])	#std of the difference divided by mean for normalization
    			#Better: weighted average to give more importance to good lines acc to NIST and errors
    			#first calculate the weights as the errors of the "diffe" diffe=ts/ratioline
    			aver=np.average(diffe[l,:,i,j], weights=ediffe[l,:,i,j]) #weighted mean of observation number l compared to model i,j
    			aver=np.median(diffe[l,:,i,j])			
    			#print 'Aver', i,j,l, aver
    			vari=np.average((diffe[l,:,i,j]-aver)**2,weights=ediffe[l,:,i,j])
    			#print 'vari', i,j,l,vari
    			aver=np.median(diffe[l,:,i,j])			
    			stdratio[l,i,j]=sqrt(vari)/aver	#std of the difference divided by mean for normalization
    			averatio[l,i,j]=aver	#to keep track of the zero point.
    			#blah=std(diffe[l,:,i,j])/mean(diffe[l,:,i,j])	#std of the difference divided by mean for normalization
    			#print 'compare the means', aver, mean(diffe[l,:,i,j])
    			#print 'compare the stds', blah, stdratio[l,i,j]
    			
    
#     g=open('badline_av_jcw.log', 'w')
    
#     #Which line is the culprit of the mess?
#     for i in range(0, size(T)):	#for the temperature
#     	for j in range(0, size(N)):	#for the density
#     		for l in range(0, numberobs):	
#     				badline=argmax(abs(diffe[l,:,i,j]-stdratio[l,i,j]))
#     				g.write("Badline %s %.3f for T=%.0f log10N=%.2f Howbad=%f\n" % (sp[badline], wl[badline], T[i], log10(N[j]), diffe[l,badline,i,j]/stdratio[l,i,j]))				
#     				g.flush()				
#     				#print 'Badline', wl[badline], sp[badline], 'for obs, T ,N', l,T[i],log10(N[j]), diffe[l,:,i,j]/stdratio[l,i,j]
    
    
    #What about removing the bad line? Tried, but at the end I don't have so many lines and it gets funny.
    
    
    
    #Now plot:
    ns=20 #20
    nl= 3	#level for the contour
    fig=figure(figsize=USH.fig_size_s)
    ax1=fig.add_subplot(1,1,1)
    corre=pcolor(log10(N),T,stdratio[0,:,:],cmap=cm.jet, vmin=np.min(stdratio[0,:,:]), vmax=np.min(stdratio[0,:,:])*ns)
    contour(log10(N),T,stdratio[0,:,:],levels=[np.min(stdratio[0,:,:]),nl*np.min(stdratio[0,:,:])],colors='w')#, linewidth=8, smooth=3)
    #axis('tight')
    #ax1.axes.xaxis.set_ticklabels([])
    xlabel(r'log$_{10}$(N$_e$) (cm$^{-3}$)')
    ylabel('T (K)')
    # t=text(8,11700,'2009-10-16', backgroundcolor='w')
    # t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='w'))
    #xlabel(r'log$_{10}$(N$_e$) (cm$^{-3}$)')
    cbaxes = inset_axes(ax1, width="3%", height="40%", loc=2) 
    fig.colorbar(corre,cax=cbaxes, orientation='vertical')
    #fig.colorbar(corre, shrink=0.8, aspect=30)
    #axis('tight')    
    subplots_adjust(left=0.2, bottom=0.15, right=0.97, top=0.97, wspace=0.1, hspace=0.05)

        
    
    # ns=20
    # nl=3	#level for the contour
    # ax1=subplot(1,5,1)
    # corre=pcolor(log10(N),T,stdratio[0,:,:],cmap=cm.jet, vmin=np.min(stdratio[0,:,:]), vmax=np.min(stdratio[0,:,:])*ns)
    # contour(log10(N),T,stdratio[0,:,:],levels=[np.min(stdratio[0,:,:]),nl*np.min(stdratio[0,:,:])],colors='w', linewidth=8, smooth=3)
    # axis('tight')
    # #ax1.axes.xaxis.set_ticklabels([])
    # xlabel(r'log$_{10}$(N$_e$) (cm$^{-3}$)')
    # ylabel('T (K)')
    # t=text(8,11700,'2009-10-16', backgroundcolor='w')
    # t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='w'))
    # #xlabel(r'log$_{10}$(N$_e$) (cm$^{-3}$)')
    # cbaxes = inset_axes(ax1, width="3%", height="40%", loc=2) 
    # fig.colorbar(corre,cax=cbaxes, orientation='vertical')
    # #fig.colorbar(corre, shrink=0.8, aspect=30)
    # axis('tight')
    
    # ax1=subplot(1,5,2)
    # corre=pcolor(log10(N),T,stdratio[1,:,:],cmap=cm.jet, vmin=np.min(stdratio[1,:,:]), vmax=np.min(stdratio[1,:,:])*ns)
    # contour(log10(N),T,stdratio[1,:,:],levels=[np.min(stdratio[1,:,:]),nl*np.min(stdratio[1,:,:])],colors='w', linewidth=8, smooth=3)
    # t=text(8,11700,'2009-10-18', backgroundcolor='w')
    # t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='w'))
    # xlabel(r'log$_{10}$(N$_e$) (cm$^{-3}$)')
    # #ax1.axes.xaxis.set_ticklabels([])
    # ax1.axes.yaxis.set_ticklabels([])
    # cbaxes = inset_axes(ax1, width="3%", height="40%", loc=2) 
    # fig.colorbar(corre,cax=cbaxes, orientation='vertical')
    # axis('tight')
    
    # ax1=subplot(1,5,3)
    # corre=pcolor(log10(N),T,stdratio[2,:,:],cmap=cm.jet, vmin=np.min(stdratio[2,:,:]), vmax=np.min(stdratio[2,:,:])*ns)
    # contour(log10(N),T,stdratio[2,:,:],levels=[np.min(stdratio[2,:,:]),nl*np.min(stdratio[2,:,:])],colors='w', linewidth=8, smooth=3)
    # t=text(8,11700,'2009-10-19', backgroundcolor='w')
    # t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='w'))
    # xlabel(r'log$_{10}$(N$_e$) (cm$^{-3}$)')
    # #ylabel('T (K)')
    # #xlabel(r'log$_{10}$(N$_e$) (cm$^{-3}$)')
    # cbaxes = inset_axes(ax1, width="3%", height="40%", loc=2) 
    # ax1.axes.yaxis.set_ticklabels([])
    # #ax1.axes.xaxis.set_ticklabels([])
    # fig.colorbar(corre,cax=cbaxes, orientation='vertical')
    # axis('tight')
    
    
    # ax1=subplot(1,3,3) 
    # corre=pcolor(log10(N),T,stdratio_average[:,:],cmap=cm.jet, vmin=np.min(stdratio_average[:,:]), vmax=np.min(stdratio_average[:,:])*ns)
    # contour(log10(N),T,stdratio_average[:,:],levels=[np.min(stdratio_average[:,:]),nl*np.min(stdratio_average[:,:])],colors='w', linewidth=8, smooth=3)
    # xlabel(r'log$_{10}$(N$_e$) (cm$^{-3}$)')
    # ylabel('T (K)')
    # t=text(8.5,11700,'Quiescence Average', backgroundcolor='w')
    # t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='w'))
    # #ax1.axes.yaxis.set_ticklabels([])
    # cbaxes = inset_axes(ax1, width="3%", height="40%", loc=2) 
    # fig.colorbar(corre,cax=cbaxes, orientation='vertical')
    # axis('tight')
    
    
    
    ###
    
    #Now find the minimum value (best fit)
    
    Nbest=zeros([numberobs,size(T)])
    Indexbest=zeros([numberobs,size(T)])
    Stdbest=zeros([numberobs,size(T)])
    aveNbest=zeros([size(T)])
    aveIndexbest=zeros([size(T)])
    aveStdbest=zeros([size(T)])
    
    b_averatio=zeros([numberobs,size(T)])
    
    for i in range(0,size(T)):
    	#For the average values:
    	aveNbest[i]=N[argmin(stdratio_average[i,:])]
    	aveIndexbest[i]=argmin(stdratio_average[i,:])
    	nn=argmin(stdratio_average[i,:])
    	aveStdbest[i]=stdratio_average[i,nn]
    	for k in range(0,numberobs):		
    		Nbest[k,i]=N[argmin(stdratio[k,i,:])]
    		Indexbest[k,i]=argmin(stdratio[k,i,:])
    		Stdbest[k,i]=stdratio[k,i,argmin(stdratio[k,i,:])]
    		b_averatio[k,i]=averatio[k,i,argmin(stdratio[k,i,:])]
    		#print 'Observation number',k, 'Temperature', i, T[i], Nbest[k,i], 'Difference', stdratio[k,i,argmin(stdratio[k,i,:])]
    
    
    #Now find the absolute minimum for each spectrum:
    
    Tfin=zeros([numberobs])
    Nfin=zeros([numberobs])
    Dfin=zeros([numberobs])
    
    
    for k in range(0,numberobs):
    	Tfin[k]=T[argmin(Stdbest[k,:])]
    	Nfin[k]=Nbest[k,argmin(Stdbest[k,:])]
    	Dfin[k]=min(Stdbest[k,:])
    	print('Best fit for dataset', dates[k], 'T, log10(N)', Tfin[k], log10(Nfin[k]), 'Diff', Dfin[k], 'Baseline', b_averatio[k,argmin(Stdbest[k,:])])
    	#subplot(1,1,k+1)
    	ax1.plot(log10(Nfin[k]), Tfin[k],'w*',markersize=18)
    	
    #for the average data:
    if numberobs > 1:
        nbest=argmin(aveStdbest)
        Tave=T[nbest]
        Nave=aveNbest[nbest]

        # subplot(1,3,3)
        ax1.plot(log10(Nave),Tave,'g*',markersize=18)
        # 	
        print('Best fit for average spectrum T, log10(N)', T[nbest], log10(aveNbest[nbest]), 'Diff', aveStdbest[nbest])

    ###
    
    
    
    # There are two options: to get the very best fit, or
    # to get the value as average of all the ones within the limits, 
    # rather than to be "locked" to the grid I set.
    #The main issue is that the "very best fit" may be so by "chance".
    #If I consider all the good ones, then it is probably more representative.
    
    #Now I need to get the uncertainties.
    #Some people get 10% of the best values, but this has little justification.
    #There are few for which the std is over 2.5xminimum.
    #Minimum is ~1, which means that at least 1 line is off.
    
    ntotal=size(stdratio[0,:,:])	#This is the total number of points per dataset.
    n10=0.25*ntotal			#This is 10% of the total number of points.
    
    #First trial: get all the points for which the std is up to 1.5 of the minimum.
    
    nlim=3.	#times the stdratio to be considered as "good"
    
    stdratio0=zeros([numberobs,size(T),size(N)])
    totaltemp=[]
    totalN=[]
    stdT=zeros([numberobs])
    stdN=zeros([numberobs])
    
    std_T_ave=[]
    std_N_ave=[]
    
    
    for i in range(0,size(T)):
    	for j in range(0, size(N)):
    		#First sort out the results for the mean value that contains all observations.	
    		if stdratio_average[i,j]<nlim*np.min(stdratio_average):
    			std_T_ave.append(T[i])
    			std_N_ave.append(N[j])
    #Now go for the individual observations
    
    # print('Best fit for average spectrum, T= (best, median)', T[nbest], median(std_T_ave), 'pm', std(std_T_ave), 'log10(N)=', log10(aveNbest[nbest]), log10(average(std_N_ave)), 'pm', log10(std(std_N_ave)))
    
    # for k in range(0,numberobs):
    # 	for i in range(0,size(T)):
    # 		for j in range(0, size(N)):
    # 			if stdratio[k,i,j]<nlim*np.min(stdratio[k,:,:]):
    # 				#stdratio0[k,i,j]=stdratio[k,i,j]	#nothing changes
    # 				totaltemp.append(T[i])
    # 				totalN.append(N[j])
    # 				#once the loop in k is done, I have totaltemp and totalN that contain all the values of T, N within the
    # 				#required limits. I need then to go out of the loop and check the best results.
    # 	#Note that I need to go out of the i,j loop too, as I am adding N[j] and T[i] to the list.
    # 	stdT[k]=std(totaltemp)
    # 	stdN[k]=log10(std(totalN))
    # 	#Tfin[k]=average(totaltemp)
    # 	#Nfin[k]=average(totalN)
    # 	print('Best fit for dataset number', k, 'T= (best, median)', Tfin[k], median(totaltemp), 'pm',stdT[k],'log10(N)=', log10(Nfin[k]), log10(average(totalN)),'pm', stdN[k])
    # 	#reset the counters after each set of observations	
    # 	totaltemp=[]
    # 	totalN=[]
    
    #This error is symmetric although it likely isn't. Rather get "confidence intervals" for a better view.
    
    # totaltemp=[]
    # totalN=[]
    # T1=zeros([numberobs])
    # T2=zeros([numberobs])
    # N1=zeros([numberobs])
    # N2=zeros([numberobs])
    # nval=0
    
    #Now calculate the confidence ranges instead of plain symmetric errors.
    
    #f=open('saha_fitter_std_3lim_quiescence_2020.txt','w')
    #f.write('#0.Observation 1.Goodness 2.T(K) 3.N(cm-3)\n')
    
    
    # for k in range(0,numberobs):
    # 	for i in range(0,size(T)):
    # 		for j in range(0, size(N)):
    # 			if stdratio[k,i,j]<nlim*np.min(stdratio[k,:,:]):
    # 				#stdratio0[k,i,j]=stdratio[k,i,j]	#nothing changes
    # 				totaltemp.append(T[i])
    # 				totalN.append(N[j])
    # 				nval=nval+1
    # 				f.write('%d %f %f %f\n' % (k, stdratio[k,i,j], T[i], N[j]))
    # 				f.flush()
    # 	T1[k]=min(totaltemp)
    # 	T2[k]=max(totaltemp)
    # 	N1[k]=log10(min(totalN))
    # 	N2[k]=log10(max(totalN))
    # 	print('Best fit for obs=', k, ':   T=', Tfin[k], median(totaltemp), average(totaltemp), '(best, median, average); range',T1[k],'-',T2[k],'log10(N)=', log10(Nfin[k]), log10(median(totalN)), log10(average(totalN)), '(best, median, average);  range', N1[k],'-',N2[k], 'Nval=',nval)
    # 	#remember to reset the counters at the end	
    # 	totaltemp=[]
    # 	totalN=[]
    # 	nval=0
    
    # print('Best fit for average spectrum: RANGE, T=', T[nbest], 'range', min(std_T_ave),'-', max(std_T_ave), 'log10(N)=', log10(aveNbest[nbest]),'range', log10(min(std_N_ave)),'-',log10(max(std_N_ave)), 'Nr of points', size(std_N_ave))
    
    
    
    
    #-----------------------------
    
    #Second version forget about colorbars, put them here
    
    # ax1=subplot(1,5,1)
    # corre=pcolor(log10(N),T,stdratio[0,:,:],cmap=cm.jet, vmin=np.min(stdratio[0,:,:]), vmax=np.min(stdratio[0,:,:])*ns)
    # cbaxes = inset_axes(ax1, width="3%", height="40%", loc=2) 
    # fig.colorbar(corre,cax=cbaxes, orientation='vertical')
    
    # ax1=subplot(1,5,2)
    # corre=pcolor(log10(N),T,stdratio[1,:,:],cmap=cm.jet, vmin=np.min(stdratio[1,:,:]), vmax=np.min(stdratio[1,:,:])*ns)
    # cbaxes = inset_axes(ax1, width="3%", height="40%", loc=2) 
    # fig.colorbar(corre,cax=cbaxes, orientation='vertical')
    
    # ax1=subplot(1,5,3)
    # corre=pcolor(log10(N),T,stdratio[2,:,:],cmap=cm.jet, vmin=np.min(stdratio[2,:,:]), vmax=np.min(stdratio[2,:,:])*ns)
    # cbaxes = inset_axes(ax1, width="3%", height="40%", loc=2) 
    # fig.colorbar(corre,cax=cbaxes, orientation='vertical')
    # axis('tight')
    
    
    # ax1=subplot(1,3,3) 
    # corre=pcolor(log10(N),T,stdratio_average[:,:],cmap=cm.jet, vmin=np.min(stdratio_average[:,:]), vmax=np.min(stdratio_average[:,:])*ns)
    # contour(log10(N),T,stdratio_average[:,:],levels=[np.min(stdratio_average[:,:]),nl*np.min(stdratio_average[:,:])],colors='w', linewidth=8, smooth=3)
    # cbaxes = inset_axes(ax1, width="3%", height="40%", loc=2) 
    # fig.colorbar(corre,cax=cbaxes, orientation='vertical')
    
    # #---------------------
    
    # savefig('saha_fitter_std_quiescence_2020.png')
    
    show()


#####


#Final check:

"""figure(2)

n,m=shape(ts) #n will be the nr of wavelengths, m the nr of observations

for i in range(0,size(T)):
	for j in range(0, size(N)):
		for l in range(0,m):
			subplot(2,4,l+1)
			if stdratio[l,i,j]<nlim*np.min(stdratio[k,:,:]):
				#plot(wl, diffe[l,:,i,j]/average(diffe[l,:,i,j])) #AQUI??
				#plot(wl, ratiolines[:,i,j]/average(ratiolines[:,i,j], weights=(qualo[:]*ratiolines[:,i,j]))
				#plot(wl, ratiolines[:,i,j]/median(ratiolines[:,i,j]))
				plot(wl, ts[:,l]/ratiolines[:,i,j])				
				plot(wl, ts[:,l], 'ko')
			

savefig('saha_std_observed_vs_models.png')

show()"""


