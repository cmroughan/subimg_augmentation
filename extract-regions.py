#!/usr/bin/env python3

from shapely import union_all, is_valid, make_valid
from shapely.geometry.polygon import orient
from shapely.geometry import Polygon, box
from PIL import Image, ImageDraw
from lxml import etree
import PIL.ImageOps   
import numpy as np
import itertools
import argparse
import json
import os 

ALTO_NS = "{http://www.loc.gov/standards/alto/ns-v4#}"

# BASIC FUNCTIONS ######################################################
def average_downsample(arr, n):
	end =  n * int(len(arr)/n)
	return np.mean(arr[:end].reshape(-1, n), 1)

# ----------------------------------------------------------------------

def to_int(s):
	return int(s.split('.')[0])

# ----------------------------------------------------------------------

def create_polygon(coordinates):
	return orient(Polygon([(int(coordinates[i]), int(coordinates[i + 1])) for i in range(0, len(coordinates), 2)]), sign=1.0)

# ----------------------------------------------------------------------
# CREATE SUBIMAGE ######################################################
def calculate_threshold(subimage):
	# Convert image to NumPy array
	img_array = np.array(subimage)
	# Calculate luminosity for each colored pixel
	luminosity_array = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
	# Combine pixel data with coordinates and luminosity
	pixels_with_coordinates_and_luminosity = [(y, x, img_array[y, x], luminosity_array[y, x]) for x in range(subimage.width) for y in range(subimage.height)]
	# Sort by luminosity
	sorted_pixels = sorted(pixels_with_coordinates_and_luminosity, key=lambda x: x[3])

	# Get greyscale array and do an averaging downsample
	# Averaging downsample assumes that the pixel area of the subimage > 500.
	# If it is less, then skip the downsample.
	greyscale = np.array([luminosity for _, _, _, luminosity in sorted_pixels])
	if subimage.width * subimage.height > 500:
		divisor = int(len(greyscale) / 500)
		greyscale_downsample = average_downsample(greyscale, divisor)
	else:
		greyscale_downsample = greyscale
	
	# Find index at which the greyscale > the median value
	y1, y2 = greyscale[0], greyscale[-1]
	median = (y2 - y1) / 2 + y1
	overMedian = list(greyscale_downsample).index(next(x for x in greyscale_downsample if x > median))
	
	# Find differences between consecutive values
	differences = np.diff(greyscale_downsample)
	slope_overall = (greyscale_downsample[-1] - greyscale_downsample[0]) / len(greyscale_downsample)

	candidate_value = np.argmax(differences[overMedian:] < 0.75*slope_overall)
	candidate_value = candidate_value + overMedian
	candidate_percentile = 100 - (candidate_value / len(greyscale_downsample) * 100)

	return [candidate_percentile, (y1, y2), greyscale_downsample]

# ----------------------------------------------------------------------

def create_subimage(subimage, candidate_percentile):
	# Subtract X percentile pixel value from the subimage, to remove folio background
	inv_img = PIL.ImageOps.invert(subimage)
	X = candidate_percentile-1
	# if X goes up to 90, get some faded ink approximations
	pX_pixel_value = np.percentile(np.array(inv_img), X, axis=(0, 1))
	subtracted_arr = np.array(inv_img) - np.array(pX_pixel_value)
	subtracted_arr = np.clip(subtracted_arr, [0, 0, 0], [255, 255, 255])

	return Image.fromarray(subtracted_arr.astype('uint8'))

# ----------------------------------------------------------------------

def fill_polygon(image, polygon_coords):
	# Create a black canvas with the same size as the original image
	mask = Image.new('RGBA', image.size, (0, 0, 0, 255))

	# Draw a filled polygon on the mask
	draw = ImageDraw.Draw(mask)
	draw.polygon(polygon_coords, fill=(255, 255, 255, 0))

	# Composite the filled polygon on the original image
	result = Image.alpha_composite(image.convert('RGBA'), mask)

	# Convert the result back to 'RGB'
	result = result.convert('RGB')

	return result

# ----------------------------------------------------------------------
# XML HANDLING #########################################################
def geom_from_textBlock(folio_img, textBlock, region, regionsOnly=False):
	global ALTO_NS, geomDict, zoneDict
	regionGeom = {'polygon_union': None, 'polygon_region': None,
				 'source_img_width': None, 'source_img_height': None,
				 'lines': []}
	
	# Get source width, height
	x0 = geomDict[region]['bbox_region'].bounds[0]
	y0 = geomDict[region]['bbox_region'].bounds[1]
	w, h = folio_img.size
	regionGeom['source_img_width'] = w
	regionGeom['source_img_height'] = h
	
	# Get normalized region polygon coords
	XML_coordinates = textBlock.find(f'.//{ALTO_NS}Polygon').attrib['POINTS'].split()
	polygon_coordinates = [(round(to_int(XML_coordinates[i])), round(to_int(XML_coordinates[i + 1]))) for i in range(0, len(XML_coordinates), 2)] 
	cropped_coordinates = [(x-x0, y-y0) for x, y in polygon_coordinates]
	regionGeom['polygon_region'] = cropped_coordinates

	# Get normalized union polygon coords
	polygon_coordinates = list(geomDict[region]['polygon_all'].exterior.coords)
	cropped_coordinates = [(round(x-x0), round(y-y0)) for x, y in polygon_coordinates]
	regionGeom['polygon_union'] = cropped_coordinates

	# Get line data, if lines are present in region
	textLines = [] if regionsOnly else textBlock.findall(f'.//{ALTO_NS}TextLine')
	for textLine in textLines:
		lineData = {'line_type': None, 'baseline': None, 'polygon_line': None, 'text': None}
		if 'TAGREFS' in textLine.attrib:
			lineData['line_type'] = zoneDict[textLine.attrib['TAGREFS']]

		XML_coordinates = textLine.attrib['BASELINE'].split()
		baseline_coordinates = [(round(to_int(XML_coordinates[i])), round(to_int(XML_coordinates[i + 1]))) for i in range(0, len(XML_coordinates), 2)] 
		cropped_coordinates = [(x-x0, y-y0) for x, y in baseline_coordinates]
		lineData['baseline'] = cropped_coordinates

		XML_coordinates = textLine.find(f'.//{ALTO_NS}Polygon').attrib['POINTS'].split()
		polygon_coordinates = [(round(to_int(XML_coordinates[i])), round(to_int(XML_coordinates[i + 1]))) for i in range(0, len(XML_coordinates), 2)] 
		cropped_coordinates = [(x-x0, y-y0) for x, y in polygon_coordinates]
		lineData['polygon_line'] = cropped_coordinates

		string_el = textLine.find(f'.//{ALTO_NS}String')
		if 'CONTENT' in string_el.attrib:
			lineData['text'] = string_el.attrib['CONTENT']

		regionGeom['lines'].append(lineData)
	
	return regionGeom

# ----------------------------------------------------------------------

def subimage_from_region(folio_img, region):
	global geomDict

	x0 = geomDict[region]['bbox_region'].bounds[0]
	y0 = geomDict[region]['bbox_region'].bounds[1]
	x1 = geomDict[region]['bbox_region'].bounds[2]
	y1 = geomDict[region]['bbox_region'].bounds[3]
	
	# Crop folio_img to subimage
	cropped_img = folio_img.crop((x0, y0, x1, y1))
	candidate_percentile, y_values, greyscale = calculate_threshold(cropped_img)
	result = create_subimage(cropped_img, candidate_percentile)

	# Set any pixels outside the region polygon to black
	polygon_coordinates = list(geomDict[region]['polygon_region'].exterior.coords)
	cropped_coordinates = [(round(x-x0), round(y-y0)) for x, y in polygon_coordinates]
	result = fill_polygon(result, cropped_coordinates)

	return result

# ----------------------------------------------------------------------

def build_zoneDict(tree):
	global ALTO_NS
	zoneDict = {}
	tags = tree.findall(f'.//{ALTO_NS}OtherTag')
	for tag in tags:
		zoneDict[tag.attrib['ID']] = tag.attrib['LABEL']

	return zoneDict

# ----------------------------------------------------------------------

def filter_textBlocks(tree, filter=[]):
	global ALTO_NS, zoneDict
	
	# Remove dummyblocks and filtered region types
	textBlocks = tree.findall(f'.//{ALTO_NS}TextBlock')
	for textBlock in textBlocks:
		if 'dummyblock' in textBlock.attrib['ID']:
			textBlock.getparent().remove(textBlock)
		elif 'TAGREFS' not in textBlock.attrib:
			textBlock.getparent().remove(textBlock)
		elif textBlock.attrib['TAGREFS'] in filter:
			textBlock.getparent().remove(textBlock)
		elif zoneDict[textBlock.attrib['TAGREFS']] in filter:
			textBlock.getparent().remove(textBlock)
	textBlocks = tree.findall(f'.//{ALTO_NS}TextBlock')
	
	return textBlocks

# ----------------------------------------------------------------------

def get_geometry(textBlocks, regionsOnly=False):
	global ALTO_NS, zoneDict
	geomDict = {}
	
	# Get geometries
	for textBlock in textBlocks:
		pathID = textBlock.attrib['ID']
		if pathID in geomDict:
			print(f'Warning: {pathID} in XML file multiple times!')
		regionType = textBlock.attrib['TAGREFS']
		
		x0 = to_int(textBlock.attrib['HPOS'])
		y0 = to_int(textBlock.attrib['VPOS'])
		x1 = x0 + to_int(textBlock.attrib['WIDTH'])
		y1 = y0 + to_int(textBlock.attrib['HEIGHT'])
		bbox = box(x0, y0, x1, y1)

		textLines = [] if regionsOnly else textBlock.findall(f'.//{ALTO_NS}TextLine') 
		for textLine in textLines:
			x0 = min(x0, to_int(textLine.attrib['HPOS']))
			y0 = min(y0, to_int(textLine.attrib['VPOS']))
			x1 = max(x1, to_int(textLine.attrib['HPOS'])+to_int(textBlock.attrib['WIDTH']))
			y1 = max(y1, to_int(textLine.attrib['VPOS'])+to_int(textBlock.attrib['HEIGHT']))
		bbox_all = bbox if regionsOnly else box(x0, y0, x1, y1)

		poly = create_polygon(textBlock.find(f'./{ALTO_NS}Shape/{ALTO_NS}Polygon').attrib['POINTS'].split())
		if not is_valid(poly):
			poly = make_valid(poly)

		geomList = [poly]
		for textLine in textLines:
			poly_elem = textLine.find(f'./{ALTO_NS}Shape/{ALTO_NS}Polygon')
			if poly_elem is None:
				continue
			poly_line = create_polygon(poly_elem.attrib['POINTS'].split())
			if not is_valid(poly_line):
				poly = make_valid(poly_line)
			geomList.append(poly_line)
		poly_all = union_all(geomList)
			
		geomDict[pathID] = {
			'bbox_region': bbox,
			'bbox_all': bbox_all,
			'polygon_region': poly,
			'polygon_all': poly_all,
			'region_type': zoneDict[regionType],
			'textBlock': textBlock
		}
	return(geomDict)

# ----------------------------------------------------------------------

def find_non_overlapping():
	global ALTO_NS, geomDict
	# Navigate the XML
	
	# Initialize list of all regions in XML
	# If overlap is found, the regions will be removed from this list
	candidateIDs = [r for r in geomDict]
	
	# Search TextBlocks and Polygons for overlapping material
	for regionA, regionB in itertools.combinations(geomDict.keys(), 2):
		try:
			# Get rid of MultiPolygons
			polygonA = geomDict[regionA]['polygon_all']
			polygonB = geomDict[regionB]['polygon_all']
			if 'MultiPolygon' in str(type(polygonA)):
				if regionA in candidateIDs:
					candidateIDs.remove(regionA)
			if 'MultiPolygon' in str(type(geomDict[regionA]['polygon_region'])):
				if regionA in candidateIDs:
					candidateIDs.remove(regionA)
			if 'MultiPolygon' in str(type(polygonB)):
				if regionB in candidateIDs:
					candidateIDs.remove(regionB)
			if 'MultiPolygon' in str(type(geomDict[regionB]['polygon_region'])):
				if regionB in candidateIDs:
					candidateIDs.remove(regionB)
					
			# Check bbox intersection first, as the simpler case
			boxA = geomDict[regionA]['bbox_all']
			boxB = geomDict[regionB]['bbox_all']
			if not boxA.intersects(boxB):
				continue
				
			# If bboxes intersect, check respective polygons
			# If polygons intersect, remove from candidateIDs
			if polygonA.intersects(polygonB):
				if regionA in candidateIDs:
					candidateIDs.remove(regionA)
				if regionB in candidateIDs:
					candidateIDs.remove(regionB)
					
		except Exception as e:
			print(regionA)
			print(regionB)
			print(e)
				
	# Return remaining list, which now contains only non-overlapping regions
	return candidateIDs

# ----------------------------------------------------------------------
# ITERATE THROUGH INPUT FILES ##########################################
def process_file(inputDir, f, dir=None, regionsOnly=False):
	global output, zoneDict, geomDict
	print(f' Processing {f}...')

	tree = etree.parse(f'{inputDir}/{f}')
	
	fileName = tree.find(f'.//{ALTO_NS}fileName').text
	folio_img = Image.open(f'{inputDir}/{fileName}')
	folio_img = folio_img.convert('RGB')

	textBlocks = filter_textBlocks(tree)
	geomDict = get_geometry(textBlocks, regionsOnly=regionsOnly)
	noOverlap = find_non_overlapping()

	textBlocks = tree.findall(f'.//{ALTO_NS}TextBlock')
	textBlockDict = {}
	for t in textBlocks:
		if t.attrib['ID'] in noOverlap:
			textBlockDict[t.attrib['ID']] = t
			
	for r in noOverlap:
		zone = geomDict[r]['region_type']
		if not os.path.exists(f'{output}/{zone}'):
			os.makedirs(f'{output}/{zone}')

		f_stem = ".".join(f.split(".")[:-1])
		if dir is not None:
			f_stem = f'{dir}_{f_stem}'
		outFile = f'{f_stem}_{r}'
		subimg = subimage_from_region(folio_img, r)
		subimg.save(f"{output}/{zone}/{outFile}.webp", lossless=True, quality=100)

		textBlock = textBlockDict[r]
		regionGeom = geom_from_textBlock(folio_img, textBlock, r, regionsOnly=regionsOnly)
		with open(f"{output}/{zone}/{outFile}.json", 'w', encoding='utf-8') as out:
			json.dump(regionGeom, out)

# ----------------------------------------------------------------------

def main():
	global zoneDict, output
	inputDir = 'input'
	output = 'imgs_extracted'

	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--types', help='"rl" for regions+lines, "r" for regions only', default='rl', choices = ['rl', 'r'])
	args = parser.parse_args()

	if args.types == 'rl':
		regionsOnly = False
	elif args.types == 'r':
		regionsOnly = True
	else:
		print('-t / --types argument must take either "rl" or "r". Defaulting to regions+lines...')
		regionsOnly = False

	xmlFiles = [f for f in os.listdir(inputDir) if '.xml' in f]
	xmlFiles.sort()
	subdirs = [d for d in os.listdir(inputDir) if os.path.isdir(f'{inputDir}/{d}')]

	for f in xmlFiles:
		path = inputDir
		tree = etree.parse(f'{path}/{f}')
		zoneDict = build_zoneDict(tree)
		process_file(path, f, regionsOnly=regionsOnly)

	for d in subdirs:
		print(f'Checking directory {d}...')
		path = f'{inputDir}/{d}'
		xmlFiles = [f for f in os.listdir(path) if '.xml' in f]
		xmlFiles.sort()
		for f in xmlFiles:
			tree = etree.parse(f'{path}/{f}')
			zoneDict = build_zoneDict(tree)
			process_file(path, f, dir=d, regionsOnly=regionsOnly)

# ----------------------------------------------------------------------

if __name__ == '__main__':
	main()