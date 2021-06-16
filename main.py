import spectrum_calibration_restoration
import numpy as np
import os


def main():
    spectrum = spectrum_calibration_restoration.SpectrumCalibrationRestoration()
    # spectrum.calibration_to_spectrum(spectrum.calibration, spectrum.channel_error_coefficients)
    spectrum.calibration = np.loadtxt(os.path.join('calibration', 'calibration_from_min_max_checker.txt'))
    spectrum.kisa_iterations_spectrum_restoration(10)


if __name__ == '__main__':
    main()
