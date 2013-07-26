#!/usr/bin/python
import os
import sys


outside_images_dir = 'images/outside'
inside_images_dir = 'images/inside'

outside_classification = '1'
inside_classification = '0'

if __name__ == "__main__":

	training_examples 	= open('training_examples.txt', 'w')
	training_labels 	= open('training_labels.txt', 'w')

	outside_images = [os.path.join(outside_images_dir, image_filename) 	for image_filename in os.listdir (outside_images_dir) if image_filename[0] != '.']
	outside_labels = [outside_classification 							for image_filename in os.listdir(outside_images_dir) if image_filename[0] != '.']
	training_examples.write ('\n'.join (outside_images))
	training_labels.write 	('\n'.join (outside_labels))

	training_examples.write ('\n')
	training_labels.write ('\n')

	inside_images = [os.path.join(inside_images_dir, image_filename) 	for image_filename in os.listdir (inside_images_dir) if image_filename[0] != '.']
	inside_labels = [inside_classification 								for image_filename in os.listdir(inside_images_dir) if image_filename[0] != '.']
	training_examples.write ('\n'.join (inside_images))
	training_labels.write 	('\n'.join (inside_labels))
