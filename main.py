import spectrum_calibration_restoration
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    spectrum = spectrum_calibration_restoration.SpectrumCalibrationRestoration()
    # spectrum.calibration_to_spectrum(spectrum.calibration, spectrum.channel_error_coefficients)
    spectrum.kisa_iterations(10)


if __name__ == '__main__':
    main()
