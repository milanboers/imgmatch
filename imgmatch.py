#! /usr/bin/env python
# -*- coding: utf-8 -*-

#	Copyright 2012, Milan Boers
#
#	This program is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#
#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.
#
#	You should have received a copy of the GNU General Public License
#	along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cv2
import os
import numpy as np

"""
Class used to extract descriptors for keypoints from an image. Make one to define the settings you'd like to use.
Uses SURF for keypoint extraction.
"""
class ImageDescriptor(object):
	# Extensions of file types supported by opencv
	SUPPORTED_EXTS = ['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.tiff', '.tif']
	"""
	Constructs an ImageDescriptor.
	
	Keyword arguments:
	size -- the size images should be resized to (longest side). Smaller value will speed up the process, larger value will give more accurate results (default 300)
	surfExtractor -- SURF extractor (instance of class SURF from opencv)
	"""
	def __init__(self, surfExtractor, size=300):
		self.size = size
		self.surfExtractor = surfExtractor
	
	"""
	Returns a numpy array of the descriptors of the image in imgpath.
	
	Arguments:
	imgpath -- Path of the image to get the descriptors of
	"""
	def getDescriptors(self, imgpath):
		img = self._load(imgpath)
		
		keypoints = self.surfExtractor.detect(img)
		
		surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")
		(keypoints, descriptors) = surfDescriptorExtractor.compute(img,keypoints)
		return descriptors
	
	def _load(self, path):
		img = cv2.imread(path)
		
		# calculate new size
		factor = self.size / float(max(img.shape[0], img.shape[1]))
		newsize = (int(round(img.shape[0] * factor)), int(round(img.shape[1] * factor)))
		
		img = cv2.resize(img, newsize)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		return img

"""
Class used to create a database of all image files in a folder.
"""
class DatabaseBuilder(object):
	"""
	Constructs a DatabaseBuilder.
	
	Arguments:
	inputFolder -- name of folder the image files you want to create a database of are in. Don't use subfolders (files need a unique name).
	outputFolder -- name of folder the resulting database files will be in.
	imageDescriptor -- instance of ImageDescriptor
	"""
	def __init__(self, inputFolder, outputFolder, imageDescriptor):
		self.inputFolder = inputFolder
		self.outputFolder = outputFolder
		self.imageDescriptor = imageDescriptor
	
	"""
	Builds the database. Saves all resulting database files to the outputFolder.
	"""
	def build(self):
		for root, dirs, files in os.walk(self.inputFolder):
			for file in files:
				ext = os.path.splitext(file)[1]
				if file[0] != '_' and ext in self.imageDescriptor.SUPPORTED_EXTS:
					inPath = os.path.join(self.inputFolder, file)
					descriptors = self.imageDescriptor.getDescriptors(path)
					
					outPath = os.path.join(self.outputFolder, file)
					np.save(path, descriptors)

"""
Class used to find the best matching image from a database for another image.
"""
class Matcher(object):
	"""
	Constructs a matcher.
	
	Arguments:
	dbfolder -- folder the database files are in (should be what you used as outputFolder when building the database)
	imageDescriptor -- instance of ImageDescriptor
	
	Keyword arguments:
	distanceThreshold -- don't take the distance score into account if the value is higher than this value. A lower value results in more differing scores, but might also result in more false positives. (default .1)
	"""
	def __init__(self, dbfolder, imageDescriptor, distanceThreshold=.1):
		self.dbfolder = dbfolder
		self.imageDescriptor = imageDescriptor
		self.distanceThreshold = distanceThreshold
	
	"""
	Matches an image against the database. Returns a generator for (filename, score) pairs.
	
	Arguments:
	imgpath -- path of the image that needs to be matched against the database
	"""
	def match(self, imgpath):
		descriptors = self.imageDescriptor.getDescriptors(imgpath)
		knn = self._getKnn(descriptors)
		
		for root, dirs, files in os.walk(self.dbfolder):
			for file in files:
				ext = os.path.splitext(file)[1]
				if ext == '.npy':
					dbfile = np.load(os.path.join(self.dbfolder, file))
					yield (file, self._match(knn, dbfile))
	
	def _getKnn(self, descriptors):
		# Setting up samples and responses for kNN
		samples = np.array(descriptors)
		responses = np.arange(len(descriptors), dtype = np.float32)
		
		# kNN training
		knn = cv2.KNearest()
		knn.train(samples, responses)
		
		return knn
	
	def _match(self, knn, descriptors):
		score = 0
		
		for h,des in enumerate(descriptors):
			des = np.array(des, np.float32).reshape((1,128))
			retval, results, neigh_resp, dists = knn.find_nearest(des,1)
			#res,dist =  int(results[0][0]), dists[0][0]
			dist = dists[0][0]
			
			if dist < self.distanceThreshold:
				score += self.distanceThreshold - dist
		
		return score