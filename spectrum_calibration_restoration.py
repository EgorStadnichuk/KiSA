import numpy as np
import scipy.integrate
from scipy.optimize import minimize
import satelite_read_hdf
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class SpectrumCalibrationRestoration:
    geant4_simulation_energies, geant4_simulation_data = satelite_read_hdf.obtain_geant4_data()
    # geant4_simulation_energies - a 1d array with simulated energies, measured in MeV
    # geant4_simulation_data - a 2d array with simulated mean energy deposit. Calls as geant4_simulation_data[i][j],
    # i - energy (in accordance with geant4_simulation_energies), j - number of detector channel
    number_of_channels = 10
    # number of channels used in data analysis, channels start from the first one (entrance window for accelerator beam)
    detector_processed_data = np.load(os.path.join('accelerator_data', 'processed_data_10pucks.npy'))
    # detector_processed_data - a 2d array with not calibrated experimental energy deposit. Calls as
    # detector_processed_data[i, j], i - number of channel, j - number of event (particle hit)
    calibration = np.loadtxt(os.path.join('calibration', 'calibration_from_min_max_lower.txt'))
    # calibration - a 1d array with calibration coefficients. Calls as calibration[i], i - number of channel
    # calibration defines as: proton energy [MeV] = calibration * processed_data
    channel_error_coefficients = np.loadtxt(os.path.join('calibration', 'errors_from_proton_spectrum_first.txt'))
    # channel_error_coefficients - 1d array. Calls as channel_error_coefficients[i], i - number of channel
    # channel error coefficient defines as: energy deposit error [MeV] = error coefficient * sqrt(energy deposit [MeV])
    # physical meaning: channel signal gauss distribution width for 1 MeV energy deposit detection
    geant4_simulation_data_proton_energy_bin = 0.05  # MeV
    # geant4_simulation_data_proton_energy_bin - an energy step of Geant4 proton simulations
    proton_energies = 0

    # proton_energies - a 1d array of restored energies of accelerator protons. Calls as proton_energies[i],
    # i - number of event in accordance with detector_processed_data

    def geant4_simulation_data_return_energy_index_lower(self, energy):  # energy in MeV, returns lower index from
        # geant4_simulation_energies array for particular energy, lower means the index before the particular energy
        # geant4_simulation_energies[index] <= energy < geant4_simulation_energies[index + 1]
        index = 0
        while self.geant4_simulation_energies[index] < energy:
            index = index + 1
            if index == len(self.geant4_simulation_energies):
                return int(index - 1)
        return int(index - 1)

    def geant4_simulation_data_mean_energy_deposit_from_energy(self, channel_number, energy: object) -> object:
        # returns interpolated energy deposit, in other words, makes geant4_simulation_data continuous via
        # quadratic interpolation
        index = self.geant4_simulation_data_return_energy_index_lower(energy)
        if index < 3:  # to prevent this method from array index out of bounds error
            return interp1d(self.geant4_simulation_energies[0: 3], np.array(
                [self.geant4_simulation_data[0][channel_number], self.geant4_simulation_data[1][channel_number],
                 self.geant4_simulation_data[2][channel_number]]), kind='quadratic', fill_value='extrapolate')(
                np.array([energy]))
        elif index > len(self.geant4_simulation_energies) - 3:  # to prevent this method from array index out of bounds
            # error
            return interp1d(self.geant4_simulation_energies[len(self.geant4_simulation_energies) - 3: len(
                self.geant4_simulation_energies)], np.array([self.geant4_simulation_data[
                            len(self.geant4_simulation_energies) - 3][channel_number], self.geant4_simulation_data[
                    len(self.geant4_simulation_energies) - 2][channel_number], self.geant4_simulation_data[len(
                        self.geant4_simulation_energies) - 1][channel_number]]), kind='quadratic',
                            fill_value='extrapolate')(np.array([energy]))
        else:
            return interp1d(self.geant4_simulation_energies[index - 1: index + 2], np.array(
                [self.geant4_simulation_data[index - 1][channel_number],
                 self.geant4_simulation_data[index][channel_number],
                 self.geant4_simulation_data[index + 1][channel_number]]), kind='quadratic', fill_value='extrapolate')(
                np.array([energy]))

    def spectrum_to_calibration(self, spectrum, number_of_particles):  # returns detector calibration from
        # the given spectrum
        # spectrum - continuous normalized energy spectrum function
        calibration = np.zeros(self.number_of_channels)
        simulations_energy_deposit = np.zeros(self.number_of_channels)
        processed_data_energy_deposit = np.zeros(self.number_of_channels)
        for i in range(self.number_of_channels):
            simulations_energy_deposit[i] = number_of_particles * scipy.integrate.simps(
                lambda energy: spectrum(energy) * self.geant4_simulation_data_mean_energy_deposit_from_energy(energy),
                self.geant4_simulation_energies[0],
                self.geant4_simulation_energies[len(self.geant4_simulation_energies)]) / np.sum(
                self.detector_processed_data[i, :])  # here spectrum convolves with simulation energy deposit to obtain
            # total theoretical signal on individual channels
            for j in range(len(self.detector_processed_data[i, :])):
                processed_data_energy_deposit[i] = processed_data_energy_deposit[i] + self.detector_processed_data[i, j]
                # obtains experimental total channel signal
            calibration[i] = simulations_energy_deposit[i] / processed_data_energy_deposit[i]  # according to definition
        return calibration

    def proton_energies_to_calibration(self, proton_energies):  # returns detector calibration from
        # given energies of detected protons (in MeV)
        calibration = np.zeros(self.number_of_channels)
        simulations_energy_deposit = np.zeros(self.number_of_channels)
        processed_data_energy_deposit = np.zeros(self.number_of_channels)
        for i in range(self.number_of_channels):
            for j in range(len(proton_energies)):
                simulations_energy_deposit[i] = simulations_energy_deposit[i] + \
                                                self.geant4_simulation_data_mean_energy_deposit_from_energy(
                                                    i, proton_energies[j])
                # obtains simulation total channel signal in MeV
                processed_data_energy_deposit[i] = processed_data_energy_deposit[i] + self.detector_processed_data[i, j]
                # obtains experimental total channel signal
            calibration[i] = simulations_energy_deposit[i] / processed_data_energy_deposit[i]  # according to definition
        plt.bar(range(1, self.number_of_channels + 1), calibration)
        plt.xlabel('Number of channel')
        plt.ylabel('Calibration coefficient')
        plt.show()
        plt.close()
        return calibration

    def monochromatic_minimize_function(self, number_of_event, energy, calibration, channel_error_coefficients):
        #  chi square functional for individual accelerator proton hit event
        result = 0
        for n in range(self.number_of_channels):
            result = result + (self.detector_processed_data[n, number_of_event] * calibration[n] -
                               self.geant4_simulation_data_mean_energy_deposit_from_energy(n, energy)) ** 2 / \
                     channel_error_coefficients[n] / np.sqrt(self.detector_processed_data[n, number_of_event] *
                                                             calibration[n])
        return result

    def monochromatic_fit(self, number_of_event, calibration, channel_error_coefficients, x0=np.array([70.0])):
        # minimizes monochromatic_minimize_function to obtain optimal proton energy for individual accelerator
        # proton hit event
        return minimize(
            lambda energy: self.monochromatic_minimize_function(number_of_event, energy, calibration,
                                                                channel_error_coefficients),
            x0=x0, method='Nelder-Mead', options={'maxiter': 100000, 'disp': False})

    def calibration_to_spectrum(self, calibration, channel_error_coefficients):
        # uses calibrated processed data to obtain experimental proton energies array
        # analyses each accelerator proton hit event to return individual proton optimal energy in MeV
        energies = np.zeros(len(self.detector_processed_data[0, :]))
        for event in range(len(self.detector_processed_data[0, :])):
            energies[event] = self.monochromatic_fit(event, calibration, channel_error_coefficients).x[0]
        plt.hist(energies, bins=50)
        plt.xlabel('Proton energy, MeV')
        plt.ylabel('Distribution')
        plt.show()
        plt.close()
        return energies

    def kisa_iterations(self, number_of_iterations):
        # iteratively obtains optimal detector auto calibration and experimental proton spectrum
        # calibration_to_spectrum method returns detected accelerator proton energies array from calibration,
        # then proton_energies_to_calibration returns calibration from protons energies
        # procedures repeat iteratively to obtain optimal spectrum and calibration
        for i in range(number_of_iterations):
            print(i)
            self.proton_energies = self.calibration_to_spectrum(self.calibration, self.channel_error_coefficients)
            self.calibration = self.proton_energies_to_calibration(self.proton_energies)
