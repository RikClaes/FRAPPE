#!/usr/bin/env python

			############################################
			#			  readcol.py 	 			   #
			#	function to read a regular ASCII table #
			############################################

# ---------------------------------------------------------------------------------------
# Purpose:
#	This function will read all the (selected) lines with the format provided
#	and skip all the lines with a different format.
# 	TO be used only when the file is made by columns separated by spaces and
#	all with the same format.
#	Currently, reads up to 20 variables.
#
# OUTPUT:
# 	Result(s) = numpy array with the values from the given column
#
# INPUT:
# 	filename = entire filename with path of the ASCII file to be read
# 	nvar = number of variables to be read (should be the same as output arrays)
#
# OPTIONAL INPUT:
# 	format = format in IDL like style ('I' = integer...)
# 	skipline = lines to be skipped at the beginning of the file
# 	numline = number of lines to be read (if not, read all lines in the file)
#
# Example:
#	At the beginning of the script:
#	from readcol import *
#
#	in the script:
#	v1 [,v2,v3...] = readcol(filename,nvar [,format='FMT',skipline=skipline,numline=numline])
#
#
# Created by:
#		CFM, on Dec 27-29 2012 @Il Ciocco
#Coverted to Py3 by:
#		RC june 2023 @ESO
#
# ---------------------------------------------------------------------------------------

import numpy as np
import string

def readcol_py3(file,nvar,format='FMT',skipline=0,numline=1e5,verbose='YES'):

	#get the total number of lines in the file
	nlines = file_lines(file,verbose=verbose)
	ngood = 0	#here I will put the number of good lines read
	nbad = 0	#here I will put the number of bad lines that I'll skip
	#get the number of lines to be read
	if skipline > 0:
		nlines = nlines - skipline
		#check that there are lines to be read
		if nlines <= 0:
			print('No data to be read!')
			return False
	if numline < 1e5:
		#check that the user asked to read a reasonable number of lines
		if numline <= nlines:
			nlines = numline
		else:
			print('Hey! You asked to read more lines than available!!!')
			nlines = nlines
	#NOW I know that I want to read nlines lines from the file, and
	#that there are enough (i.e.nlines_tot>=nlines) lines in the file

	#read the format string (if any) to decide which variables should
	#be decleared
	if format != 'FMT':
		format = ''.join(format.split(' '))
		format = ''.join(format.split(','))
	else:
		format = 'F'*nvar	#if not specified, assume all variables to be float
		print(format)

	#decleare the variables to be read
	ncol = 0	#total number of colums
	nskip = 0	#number of skipped columns
	dcol = {}	#here I store the number of column for each variable
	if nvar >=1:
		v1,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v1'] = ncol-1
	if nvar >=2:
		v2,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v2'] = ncol-1
	if nvar >=3:
		v3,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v3'] = ncol-1
	if nvar >=4:
		v4,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v4'] = ncol-1
	if nvar >=5:
		v5,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v5'] = ncol-1
	if nvar >=6:
		v6,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v6'] = ncol-1
	if nvar >=7:
		v7,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v7'] = ncol-1
	if nvar >=8:
		v8,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v8'] = ncol-1
	if nvar >=9:
		v9,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v9'] = ncol-1
	if nvar >=10:
		v10,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v10'] = ncol-1
	if nvar >=11:
		v11,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v11'] = ncol-1
	if nvar >=12:
		v12,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v12'] = ncol-1
	if nvar >=13:
		v13,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v13'] = ncol-1
	if nvar >=14:
		v14,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v14'] = ncol-1
	if nvar >=15:
		v15,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v15'] = ncol-1
	if nvar >=16:
		v16,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v16'] = ncol-1
	if nvar >=17:
		v17,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v17'] = ncol-1
	if nvar >=18:
		v18,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v18'] = ncol-1
	if nvar >=19:
		v19,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v19'] = ncol-1
	if nvar >=20:
		v20,ncol,nskip = define_variable(format,ncol,nskip,nlines)
		dcol['v20'] = ncol-1
	if nvar >=21:
		print('MORE THAN 20 variables is NOT YET SUPPORTED! PLEASE UPGRADE ME!')
		return False
	#if there are 'X' at the end, count them as skipped columns
	if len(format) > ncol:
		# print(len(format),ncol)
		if format[ncol] == 'X':
			while (ncol < len(format)) and (format[ncol] == 'X'):
				ncol+=1
				nskip+=1

	#check that the length of format and the number of variables are the same
	if (ncol-nskip) != nvar:
		print('Hey! The FORMAT you specified is wrong!')
		return False

	nread = 0 #number of already read lines in the file
	#read the file without opening it
	for line in open(file):
		if nread < skipline:	#skip the lines at the beginning
			if verbose == 'YES':
				print('skip')
			line.split()
		elif ngood >= nlines:	#skip the lines at the end
			print('final')
			line.split()
		else:
			temp_line = line.split()
			if (len(temp_line) < ncol) or (len(temp_line) > ncol):
			#	print(len(temp_line),ncol)
				nbad+=1 #this line was different then expected, do not use it!
			else:
				if nvar >=1:
					v1[ngood] = temp_line[dcol['v1']]
				if nvar >=2:
					v2[ngood] = temp_line[dcol['v2']]
				if nvar >=3:
					v3[ngood] = temp_line[dcol['v3']]
				if nvar >=4:
					v4[ngood] = temp_line[dcol['v4']]
				if nvar >=5:
					v5[ngood] = temp_line[dcol['v5']]
				if nvar >=6:
					v6[ngood] = temp_line[dcol['v6']]
				if nvar >=7:
					v7[ngood] = temp_line[dcol['v7']]
				if nvar >=8:
					v8[ngood] = temp_line[dcol['v8']]
				if nvar >=9:
					v9[ngood] = temp_line[dcol['v9']]
				if nvar >=10:
					v10[ngood] = temp_line[dcol['v10']]
				if nvar >=11:
					v11[ngood] = temp_line[dcol['v11']]
				if nvar >=12:
					v12[ngood] = temp_line[dcol['v12']]
				if nvar >=13:
					v13[ngood] = temp_line[dcol['v13']]
				if nvar >=14:
					v14[ngood] = temp_line[dcol['v14']]
				if nvar >=15:
					v15[ngood] = temp_line[dcol['v15']]
				if nvar >=16:
					v16[ngood] = temp_line[dcol['v16']]
				if nvar >=17:
					v17[ngood] = temp_line[dcol['v17']]
				if nvar >=18:
					v18[ngood] = temp_line[dcol['v18']]
				if nvar >=19:
					v19[ngood] = temp_line[dcol['v19']]
				if nvar >=20:
					v20[ngood] = temp_line[dcol['v20']]
				ngood+=1 #this line was good, count it!
		nread+=1	 #we read another line

	if verbose == 'YES':
		print('BAD LINES:')
		print(nbad)
		print('GOOD LINES:')
		print(ngood)

	#if the number of good lines was less than the total number of lines to read, then
	#the variable could be bigger, therefore it's better to use only the useful part
	if ngood != nlines:
		if nvar >=1:
			v1 = v1[:ngood]
		if nvar >=2:
			v2 = v2[:ngood]
		if nvar >=3:
			v3 = v3[:ngood]
		if nvar >=4:
			v4 = v4[:ngood]
		if nvar >=5:
			v5 = v5[:ngood]
		if nvar >=6:
			v6 = v6[:ngood]
		if nvar >=7:
			v7 = v7[:ngood]
		if nvar >=8:
			v8 = v8[:ngood]
		if nvar >=9:
			v9 = v9[:ngood]
		if nvar >=10:
			v10 = v10[:ngood]
		if nvar >=11:
			v11 = v11[:ngood]
		if nvar >=12:
			v12 = v12[:ngood]
		if nvar >=13:
			v13 = v13[:ngood]
		if nvar >=14:
			v14 = v14[:ngood]
		if nvar >=15:
			v15 = v15[:ngood]
		if nvar >=16:
			v16 = v16[:ngood]
		if nvar >=17:
			v17 = v17[:ngood]
		if nvar >=18:
			v18 = v18[:ngood]
		if nvar >=19:
			v19 = v19[:ngood]
		if nvar >=20:
			v20 = v20[:ngood]

	#return the number of variables requested
	if nvar == 20:
		return v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20
	elif nvar == 19:
		return v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19
	elif nvar == 18:
		return v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18
	elif nvar == 17:
		return v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17
	elif nvar == 16:
		return v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16
	elif nvar == 15:
		return v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15
	elif nvar == 14:
		return v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14
	elif nvar == 13:
		return v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13
	elif nvar == 12:
		return v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12
	elif nvar == 11:
		return v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11
	elif nvar == 10:
		return v1,v2,v3,v4,v5,v6,v7,v8,v9,v10
	elif nvar == 9:
		return v1,v2,v3,v4,v5,v6,v7,v8,v9
	elif nvar == 8:
		return v1,v2,v3,v4,v5,v6,v7,v8
	elif nvar == 7:
		return v1,v2,v3,v4,v5,v6,v7
	elif nvar == 6:
		return v1,v2,v3,v4,v5,v6
	elif nvar == 5:
		return v1,v2,v3,v4,v5
	elif nvar == 4:
		return v1,v2,v3,v4
	elif nvar == 3:
		return v1,v2,v3
	elif nvar == 2:
		return v1,v2
	elif nvar == 1:
		return v1



def file_lines(file,verbose):
#this functions returns the total number of lines in a file
#by opening, reading and counting the number of lines
	nlines=0
	for line in open(file):
		nlines+=1
	# f = open(file,'r')
	# temp_file = f.read()
	# nlines = len(temp_file.split('\n'))
	# f.close()
	if verbose == 'YES':
		print('NUMBER OF LINES IN THE FILE: ',nlines)
	return nlines



def define_variable(format,ncol,nskip,nlines):
#defines the variable according to its format and checking if the
#column should be skipped ('X')
	if format[ncol] == 'X':
		while format[ncol] == 'X':
			ncol+=1
			nskip+=1
		vec = np.zeros(nlines,variable_fmt(format[ncol]))
		ncol+=1
	else:
		vec = np.zeros(nlines,variable_fmt(format[ncol]))
		ncol+=1
	return vec,ncol,nskip


def variable_fmt(FORMAT):
#translates the format from string IDL style to numpy argument
	if FORMAT == 'I':
		return np.int16
	elif FORMAT == 'L':
		return np.int32
	elif FORMAT == 'F':
		return np.float32
	elif FORMAT == 'D':
		return np.float64
	elif FORMAT == 'E':	# read exponentials as floats
		return np.float64
	elif FORMAT == 'A':
		return np.str_,200	#reads single string elements up to 200 digits
	else:
		print('FORMAT NOT KNOWN!')
		return
