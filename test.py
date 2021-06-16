import matplotlib.pyplot as plt
import numpy as np
import os
import spectrum
import satelite_read_hdf
import spectrum_calibration_restoration


def test():
    # calibration = np.loadtxt(os.path.join('calibration', 'calibration_from_min_max_checker.txt'))
    # # geant4_simulation_energies, geant4_simulation_data = satelite_read_hdf.obtain_geant4_data()
    # # detector_processed_data = np.load(os.path.join('accelerator_data', 'processed_data_10pucks.npy'))
    # plt.bar(range(1, 10 + 1), calibration)
    # plt.xlabel('Number of channel')
    # plt.ylabel('Calibration coefficient')
    # plt.show()
    # proton_bunch, errors = spectrum.optimiser(calibration)
    # print(proton_bunch)
    # print(errors)
    spectra = spectrum_calibration_restoration.SpectrumCalibrationRestoration()
    spectra.calibration = np.loadtxt(os.path.join('calibration', 'calibration_from_min_max_checker.txt'))
    spectra.kisa_iterations_individual_restoration(10)


if __name__ == '__main__':
    test()
