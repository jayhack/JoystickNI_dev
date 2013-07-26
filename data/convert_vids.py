#!/usr/bin/python
import os
import sys

	
if __name__ == '__main__':

	data_dir = "."
	videos_dir = os.path.join (data_dir, "video")
	images_dir = os.path.join (data_dir, "images")

	inside_vids_dir = os.path.join (videos_dir, "inside")
	outside_vids_dir = os.path.join (videos_dir, "outside")

	inside_images_dir = os.path.join (images_dir, "inside")
	outside_images_dir = os.path.join (images_dir, "outside")

	outside_vids = os.listdir (outside_vids_dir)
	for vid in outside_vids:
		if vid[0] != '.':
			command_string = './get_stills ' + os.path.join(outside_vids_dir, vid) + " " + os.path.join(outside_images_dir, vid.split('.')[0] + '.jpg') 
			os.system(command_string)

	inside_vids = os.listdir (inside_vids_dir)
	for vid in inside_vids:
		if vid[0] != '.':
			command_string = './get_stills ' + os.path.join(inside_vids_dir, vid) + " " + os.path.join(inside_images_dir, vid.split('.')[0] + '.jpg') 
			os.system(command_string)
	
