
# Load image (interferogramtest.png)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy


def calculate_beam_angle(fringe_spacing_mm, wavelength_nm):
    """Calculate the angle between interfering beams using fringe spacing and wavelength"""
    # Convert wavelength from nm to mm
    wavelength_mm = wavelength_nm / 1_000_000
    
    # Use the formula: fringe_spacing = wavelength / (2 * sin(θ/2))
    # Solving for θ: θ = 2 * arcsin(wavelength / (2 * fringe_spacing))
    
    # Check to avoid division by zero or invalid arcsin input
    if fringe_spacing_mm <= 0 or wavelength_mm / (2 * fringe_spacing_mm) > 1:
        return None
        
    angle_radians = 2 * np.arcsin(wavelength_mm / (2 * fringe_spacing_mm))
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def analyze_interferogram(results):
    fringe_spacing_pixels=results['fft_spacing']           # From FFT analysis
    aperture_diameter_pixels=400      # 2 * radius from circle detection
    aperture_diameter_real_mm= (4* 25.4)   # 4* 1 inch aperture (Zygo MKII)
    wavelength_nm=632.8                 # HeNe laser wavelength (Zygo MKII)
    # Calculate pixel scale
    pixel_scale_mm = aperture_diameter_real_mm / aperture_diameter_pixels
    print(f"Pixel scale: {pixel_scale_mm:.6f} mm/pixel")
    
    # Convert fringe spacing to physical units
    fringe_spacing_mm = fringe_spacing_pixels * pixel_scale_mm
    fringe_spacing_microns = fringe_spacing_mm * 1000
    print(f"Fringe spacing: {fringe_spacing_microns:.2f} microns ({fringe_spacing_pixels:.2f} pixels)")
    
    # Calculate beam angle
    beam_angle_degrees = calculate_beam_angle(fringe_spacing_mm, wavelength_nm)
    
    if beam_angle_degrees is not None:
        print(f"Calculated beam angle: {beam_angle_degrees:.4f} degrees")
        
        # Verify calculation by working backwards
        theoretical_spacing_mm = wavelength_nm / (1_000_000 * 2 * np.sin(np.radians(beam_angle_degrees/2)))
        print(f"Verification - theoretical spacing: {theoretical_spacing_mm*1000:.2f} microns")
    else:
        print("Could not calculate beam angle - check measurements")
    
    #return {
    #    'pixel_scale_mm': pixel_scale_mm,
    #    'fringe_spacing_mm': fringe_spacing_mm,
    #    'fringe_spacing_microns': fringe_spacing_microns,
    #    'beam_angle_degrees': beam_angle_degrees
    #}
    
def measure_fringe_distance(img, center=None, angle=None):

    # Step 1: Extract a line profile across the fringes
    # For diagonal fringes, we need to sample along a line perpendicular to the fringes
    height, width = img.shape
    print('hello')
    # If center not provided, assume it's the middle of the image
    if center is None:
        center = (width // 2, height // 2)
    
    # If angle not provided, we need to detect the fringe orientation
    if angle is None:
        # Use Hough Line Transform to detect the orientation of fringes
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Find the dominant angle from the Hough lines
            angles = [line[0][1] for line in lines]
            dominant_angle = np.median(angles)
            # The sampling line should be perpendicular to the fringes
            sampling_angle = dominant_angle + np.pi/2
        else:
            # Default angle if lines detection fails
            sampling_angle = np.pi/4  # 45 degrees
    else:
        # Convert input angle to radians if needed
        sampling_angle = angle if angle > 0.1 else angle * np.pi / 180
    
    # Calculate vector for the sampling line (perpendicular to fringes)
    # Length should be about the radius of the circle
    line_length = min(width, height) // 2
    dx = int(np.cos(sampling_angle) * line_length)
    dy = int(np.sin(sampling_angle) * line_length)
    
    # Define start and end points for the sampling line
    start_x = center[0] - dx
    start_y = center[1] - dy
    end_x = center[0] + dx
    end_y = center[1] + dy
    
    # Ensure points are within image boundaries
    start_x = max(0, min(width-1, start_x))
    start_y = max(0, min(height-1, start_y))
    end_x = max(0, min(width-1, end_x))
    end_y = max(0, min(height-1, end_y))
    
    # Extract intensity profile along the line
    num_points = int(np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2))
    x_points = np.linspace(start_x, end_x, num_points).astype(np.int32)
    y_points = np.linspace(start_y, end_y, num_points).astype(np.int32)
    intensity_profile = img[y_points, x_points]
    
    # Visualize the sampling line
    line_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    cv2.line(line_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    # Method 1: Use peak detection to find fringe centers
    # Apply smoothing to reduce noise
    smoothed_profile = scipy.ndimage.gaussian_filter1d(intensity_profile, sigma=2)
    
    # Detect peaks (bright fringes)
    peaks, _ = scipy.signal.find_peaks(smoothed_profile, height=np.mean(smoothed_profile), distance=5)
    
    # Detect valleys (dark fringes)
    valleys, _ = scipy.signal.find_peaks(-smoothed_profile, height=-np.mean(smoothed_profile), distance=5)
    
    # Calculate average distance between consecutive peaks
    peak_distances = np.diff(peaks)
    valley_distances = np.diff(valleys)
    
    # Average fringe spacing from peaks
    if len(peak_distances) > 0:
        avg_peak_spacing = np.mean(peak_distances)
    else:
        avg_peak_spacing = 0
        
    # Average fringe spacing from valleys
    if len(valley_distances) > 0:
        avg_valley_spacing = np.mean(valley_distances)
    else:
        avg_valley_spacing = 0
    
    # Average of both methods
    if avg_peak_spacing > 0 and avg_valley_spacing > 0:
        avg_fringe_spacing = (avg_peak_spacing + avg_valley_spacing) / 2
    else:
        avg_fringe_spacing = avg_peak_spacing if avg_peak_spacing > 0 else avg_valley_spacing
    
    # Method 2: Use FFT to find frequency (more robust for regular patterns)
    # Compute the FFT of the intensity profile
    # Improved FFT analysis section
    # Compute the FFT of the intensity profile
    fft = np.fft.fft(smoothed_profile - np.mean(smoothed_profile))
    freqs = np.fft.fftfreq(len(smoothed_profile))

    # Find the dominant frequency (excluding DC component and very low frequencies)
    magnitudes = np.abs(fft)
    # Zero out the DC component and very low frequencies (e.g., first few indices)
    magnitudes[:3] = 0  # Ignore DC and very low frequencies
    dominant_idx = np.argmax(magnitudes[:len(magnitudes)//2])  # Only look at first half (positive frequencies)

    # Calculate fringe spacing from frequency
    if freqs[dominant_idx] != 0:
        fft_fringe_spacing = 1 / abs(freqs[dominant_idx])
    else:
        fft_fringe_spacing = 0
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Original image with sampling line
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    plt.title('Sampling Line Direction')
    
    # Intensity profile
    plt.subplot(222)
    plt.plot(smoothed_profile)
    plt.plot(peaks, smoothed_profile[peaks], 'x', color='red', label='Peaks')
    plt.plot(valleys, smoothed_profile[valleys], 'o', color='green', label='Valleys')
    plt.title('Intensity Profile')
    plt.xlabel('Pixel')
    plt.ylabel('RGB value')
    plt.legend()
    
    # FFT magnitude
    plt.subplot(223)
    plt.plot(freqs[1:len(freqs)//2], magnitudes[1:len(freqs)//2])
    plt.axvline(freqs[dominant_idx], color='r', linestyle='--')
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (cycles/fringe per pixel)')
    #ex. 0.05 = 20 pixels per fringe (1/0.05)
    plt.ylabel('Magnitude')
    #magnitude has no physical meaning here
    
    results = {
        'peak_detection_spacing': avg_fringe_spacing,
        'fft_spacing': fft_fringe_spacing,
        'sampling_angle_degrees': sampling_angle * 180 / np.pi,
        'peaks': peaks,
        'valleys': valleys
    }
    plt.subplot(224)  # Bottom half for text
    plt.axis('off')   # No axes
    plt.text(0.5, 0.5, 
        f"MEASUREMENT RESULTS:\n\n"
        f"Peak detection method: {results['peak_detection_spacing']:.2f} pixels between fringes\n"
        f"FFT method: {results['fft_spacing']:.2f} pixels between fringes\n" 
        f"Sampling angle: {results['sampling_angle_degrees']:.2f} degrees",
        fontsize=14, 
        ha='center',  # Horizontal alignment
        va='center',  # Vertical alignment
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))



    plt.tight_layout()
    plt.ion()
    plt.show(block=False)

    

    #print(f"Peak detection method: {results['peak_detection_spacing']:.2f} pixels between fringes")
    #print(f"FFT method: {results['fft_spacing']:.2f} pixels between fringes")
    #print(f"Sampling line angle: {results['sampling_angle_degrees']:.2f} degrees")

    return results



def main():

    # Load the interferogram image
    img = cv2.imread('C:\\Users\\cmccl\\OneDrive\\Pictures\\interferogramtest.png', cv2.IMREAD_GRAYSCALE)

    # Apply preprocessing to enhance the circular edge
    # Step 1: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Step 2: Apply edge detection
    # Canny edge detector works well for finding boundaries
    edges = cv2.Canny(blurred, 50, 150)

    # Step 3: Optional - Dilate edges to connect potential gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Step 4: Use Hough Circle Transform
    # Parameters need tuning based on specific images
    circles = cv2.HoughCircles(
        dilated_edges,
        cv2.HOUGH_GRADIENT,
        dp=1,           # Resolution ratio
        minDist=5,    # Minimum distance between detected centers
        param1=5,      # Upper threshold for Canny edge detector
        param2=5,      # Threshold for center detection
        minRadius=5,   # Minimum radius to be detected
        maxRadius=0     # Maximum radius (0 means auto-maximum)
    )

    # Process results
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Get the most prominent circle (usually the first one)
        center_x, center_y, radius = circles[0, 0]
        print(f"Detected circle: center=({center_x}, {center_y}), radius={radius}")
        
        # Draw the detected circle for visualization
        result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.circle(result_img, (center_x, center_y), radius, (0, 255, 0), 2)
        cv2.circle(result_img, (center_x, center_y), 2, (0, 0, 255), 3)
        
        # Create and apply the mask
        mask = np.zeros_like(img)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        filtered_img = cv2.bitwise_and(img, mask)
        
        # Display results
        plt.figure(figsize=(15, 5))
        plt.subplot(141), plt.imshow(img, cmap='gray'), plt.title('Original')
        plt.subplot(142), plt.imshow(dilated_edges, cmap='gray'), plt.title('Edges')
        plt.subplot(143), plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)), plt.title('Detected Circle')
        plt.subplot(144), plt.imshow(filtered_img, cmap='gray'), plt.title('Filtered Result')
        plt.tight_layout()
        plt.ion()
        plt.show(block=False)
        results = measure_fringe_distance(filtered_img)
        analyze_interferogram(results)
        print('fringe measured)')
    else:
        print("No circles detected. Try adjusting the parameters.")

    #cv2.imwrite('filtered_interferogram.png', filtered_img)

    while True:
        plt.pause(1000)


if __name__ == main():
    main()
