# Live-Interferometry

Zoomtest:
broken test program that takes in webcam input (AFN_Cap) and allows output window to be moved to identify and center an interferogram. (Controlled with WASD)
Placeholder (copilot) code has been written to take the FFT, display it, and calcuate the Strehl ratio (currently displays the wrong value). 
Next steps are to display the FFT phase to the user, show area of the transform used for calculations, and properly calculate Strehl ratio like DFTFringe or to instead use algorithms from (https://opg.optica.org/ao/fulltext.cfm?uri=ao-26-9-1668) for interferogram analysis.

Filtering_Processing is a functional program that analyzes a still interferogram using Hough filtering with opencv, numpy and scipy. This will be integrated with live interferometry soon. Must configure with filepath to image, laser length (default = HeNe laser) and aperture (default = 4 inches like in the Zygo MarkII)
Features: Outputs the fringe length, displays frequency diagram, filtered image, and other info.
