import numpy as np
import sys

def spt_coding(spt_in):
	# give a number corresponding to the input SpT
	# the scale is 0 at M0, -1 at K7, -8 at K0 (K8 is counted as M0),  -18 at G0
	if np.size(spt_in) == 1:
		if spt_in[0] == 'M':
			spt_num = float(spt_in[1:])
		elif spt_in[0] == 'K':
			spt_num = float(spt_in[1:])-8.
		elif spt_in[0] == 'G':
			spt_num = float(spt_in[1:])-18.
		elif spt_in[0] == 'F':
			spt_num = float(spt_in[1:])-28.
		elif spt_in[0] == 'A':
			spt_num = float(spt_in[1:])-38.
		elif spt_in[0] == 'B':
			spt_num = float(spt_in[1:])-48.
		elif spt_in[0] == 'L':
			spt_num = float(spt_in[1:])+10.
		elif spt_in[0] == '.':
			spt_num = -99.
		else:
			sys.exit('what?')
		return spt_num
	else:
		spt_num = np.empty(len(spt_in))
		for i,s in enumerate(spt_in):
			if s[0] == 'M':
				spt_num[i] = float(s[1:])
			elif s[0] == 'K':
				spt_num[i] = float(s[1:])-8.
			elif s[0] == 'G':
				spt_num[i] = float(s[1:])-18.
			elif s[0] == 'F':
				spt_num[i] = float(s[1:])-28.
			elif s[0] == 'A':
				spt_num[i] = float(s[1:])-38.
			elif s[0] == 'B':
				spt_num[i] = float(s[1:])-48.
			elif s[0] == 'L':
				spt_num[i] = float(s[1:])+10.
			elif s[0] == '.':
				spt_num[i] = -99.
			else:
				sys.exit('what?')
		return spt_num


def convScodToSpTstring(scod):
	if np.size(scod) == 1:
		if scod<-18 or scod >10:
			print('out of bound')
			return None
		elif scod>=0:
			return 'M'+str(scod)
		elif scod<0 and scod>=-8:
			scodRet = 8+scod
			return 'K'+str(scodRet)
		elif scod<-8 and scod>-18:
			scodRed = 18+scod
			return 'G'+str(scodRed)
		return None
	else:
		spt_out = np.empty(len(scod),dtype = 'U64')
		for s,i in enumerate(scod):

			if i<-18 or i >10:
				print('out of bound')
				spt_out[s] = 'NaN'
				#return None
			elif i>=0:
				#return 'M'+str(i)
				spt_out[s] = 'M'+"%.1f" % (i)
			elif i<0 and i>=-8:
				iRet = 8+i
				spt_out[s] = 'K'+"%.1f" % (iRet)
			elif i<-8 and i>-18:
				iRed = 18+i
				spt_out[s] = 'G'+"%.1f" % (iRed)
			else: spt_out[s] ='NaN'
		return spt_out

"""
# FOR PLOTTING
ticks =  np.arange(-22,10,1)
ticklabels = np.array(['','','F8','','G0','','','','','G5','','','','','K0','','','','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
pl.xticks(ticks,ticklabels)
"""
