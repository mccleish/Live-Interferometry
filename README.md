# Live-Interferometry

Zoomtest:
broken test program that takes in webcam input (AFN_Cap) and allows output window to be moved to identify and center an interferogram. (Controlled with WASD)

Live_Testing:
Replaces zoomtest. No WASD control needed.


Filtering_Processing is a functional program that analyzes a still interferogram using Hough filtering with opencv, numpy and scipy. This will be integrated with live interferometry soon. Must configure with filepath to image, laser length (default = HeNe laser) and aperture (default = 4 inches like in the Zygo MarkII) as well as webcam output.

Features: Outputs the fringe length, displays frequency diagram, filtered image, and other info (pixel scale, calculated beam angle, etc).

Todo: Calculate PV, RMS, Strehl Ratio, incorporate something from https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-22-36754&id=540901
