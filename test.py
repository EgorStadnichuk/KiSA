import matplotlib.pyplot as plt
import numpy as np
import os


def test():
    calibration = np.loadtxt(os.path.join('calibration', 'calibration_from_min_max_lower.txt'))
    # spectrum.calibration_to_spectrum(spectrum.calibration, spectrum.channel_error_coefficients)
    # spectrum.kisa_iterations(10)
    plt.bar(range(1, 10 + 1), calibration)
    plt.xlabel('Number of channel')
    plt.ylabel('Calibration coefficient')
    plt.show()


if __name__ == '__main__':
    test()
