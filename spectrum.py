import numpy as np
from scipy import optimize
import os
from tables import open_file
import matplotlib.pyplot as plt


path = os.path.join(os.getcwd())
data_experiment = np.load(os.path.join('accelerator_data', 'processed_data_10pucks.npy'))
resukt = []
energies = []
with open_file(os.path.join(path, 'Geant4_data', 'result_detail.hdf5')) as h5file:
    for group in h5file.root:
        table = h5file.get_node(group, "deposit")
        energy = table.attrs["values_macros_energy"]
        number = table.attrs["values_macros_number"]
        data = table.read()
        data = data["event"]  # .sum(axis=0) / number
        resukt.append(data)
        energies.append(energy)
data = np.array(resukt)  # сразу разбиты на 20 шайб
energies = np.array(energies)  # энергии протонов, от 60 до 80 МэВ с шагом 0.05 МэВ
number_of_spectrum_picture = 0  # для построения картинок с разными номерами итераций


def B_spline(x, k, i, t):
    """
    B-spline
    t : numpy array
      knots
    k, i : int
      spline number
    return: B_ki spline at point x
    """
    if k == 0:
        return 1.0 if t[i] <= x < t[i + 1] else 0.0
    if t[i + k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i]) / (t[i + k] - t[i]) * B_spline(x, k - 1, i, t)
    if t[i + k + 1] == t[i + 1]:
        c2 = 0.0
    else:
        c2 = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * B_spline(x, k - 1, i + 1, t)
    return c1 + c2


def distribution(x, x_knots, b):
    """
    proton distribution
    x_knots : numpy array
    b : numpy array
      weights
    return: time at point x
    """
    t = x_knots
    T = 1e-19
    for i in range(1, len(t) - 5):
        T += b[i - 1] * B_spline(x, 3, i, t)

    return T


def norm(x, mu, sigma):
    """
    normal distribution
    """
    n = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi) / sigma
    return n / np.sum(n)


def KL(P, Q):
    """
    KL divergence
    """
    e = 10e-9
    P_new = P[((P > e) & (Q > e))]
    Q_new = Q[((P > e) & (Q > e))]

    return np.sum(P_new * np.log(P_new / Q_new))


def puck_distribution_sim(proton_bunch, shared_bins, data_simulated, puck=9):
    """
    This function calculates distribution in disk via simulated data using proton energy distribution
    :param proton_bunch: numpy.array, proton energy distribution
    :param shared_bins: bins from function puck_distribution_exp
    :param data_simulated: numpy.array, simulated data
    :param puck: int, number of disk
    :return:
    """
    y = np.zeros(len(shared_bins) - 1)
    for i in range(147, 225):
        hist = np.histogram(data_simulated[i, :, puck][(data_simulated[i, :, puck] > 0)], bins=shared_bins)
        y_new = hist[0] / np.sum(hist[0])
        y = y + y_new * proton_bunch[i]
    return y


def puck_distribution_exp(data_experiment, puck, bins=35):
    """
    This function calculate experimental distribution in disk
    :param data_experiment: numpy.array, calibrated experimental data
    :param puck: int, number of disk
    :param bins: int, number of bins to build histogram
    :return:
    """
    hist_exp, bins_exp = np.histogram(data_experiment[:, puck][(data_experiment[:, puck] > 0)], bins=bins,
                                      range=(0.03, 20))
    hist_exp = hist_exp / np.sum(hist_exp)
    return bins_exp, hist_exp


def optimize_spectrum(weights_errors, basis_function, data_experiment, data_simulated,
                      x_knots, energies):
    """
    This is objective function for optimisation
    :param weights_errors: numpy.array, basis weights and errors of disks
    :param basis_function: func, basis function
    :param data_experiment: numpy.array, calibrated experimental data
    :param data_simulated: numpy.array, simulated data
    :param x_knots: numpy.array, basis knots
    :param energies: numpy.array, energies from simulation
    :return:
    """
    divergence = 0
    errors = weights_errors[-10:]
    number_of_splines = len(weights_errors) - len(errors)
    weights = weights_errors[:number_of_splines]
    for puck in range(10):
        x_exp, y_exp = puck_distribution_exp(data_experiment=data_experiment, puck=puck)

        proton_bunch = np.array([basis_function(x, x_knots, weights) for x in energies])
        proton_bunch = proton_bunch / np.sum(proton_bunch)
        y_sim = puck_distribution_sim(proton_bunch=proton_bunch, shared_bins=x_exp, data_simulated=data_simulated,
                                      puck=puck)
        x_exp = np.delete(x_exp, -1)

        sigma = errors[puck]
        x_g = np.arange(-x_exp[-1], x_exp[-1], x_exp[1] - x_exp[0])
        g_points = norm(x_g, 0, sigma)
        y_sim_conv = np.convolve(np.concatenate((np.zeros(len(x_exp)), y_sim)), g_points, mode='same')[len(x_exp):]
        y_sim_conv = y_sim_conv / np.sum(y_sim_conv)

        divergence = divergence + KL(y_sim_conv, y_exp)
    return divergence


def optimiser(calibration, data_experiment_optimizer=data_experiment, data_simulated=data, energies_optimizer=energies,
              n=15):
    """
    This function optimises weights of spline basis and disk errors
    :param calibration: numpy.array, calibration of disks
    :param data_experiment_optimizer: numpy.array, uncalibrated experiment data
    :param data_simulated: numpy.array, data from simulation
    :param energies_optimizer: numpy.array, energies from simulation
    :param n: int, number of splines
    :return: proton energy spectrum, errors of disks
    """
    a0 = energies_optimizer[147]  # 67.4 MeV - minimum possible proton energy
    a1 = energies_optimizer[225]  # 71.3 maximum possible proton energy
    x_knots = np.concatenate(
        [np.linspace(a0 - 0.3, a0 - 0.1, 3), np.linspace(a0, a1, n), np.linspace(a1 + 0.1, a1 + 0.3, 3)])
    b0 = 0.1 * np.ones(len(x_knots) - 6)  # initial weights
    errors = 2 * np.ones(10)  # initial errors
    weights_errors0 = np.concatenate((b0, errors))
    bl_weights = np.zeros(len(b0))
    bl_errors = 0.1 * np.ones(len(errors))
    bl = tuple(np.concatenate((bl_weights, bl_errors)))  # lower bound
    bw = tuple(np.ones(len(weights_errors0) * 10))  # upper bound
    bnds = optimize.Bounds(bl, bw)  # bounds

    m = len(data_experiment_optimizer.T[:][:])
    data_calibrated = np.zeros((m, 10))
    for i in range(m):
        data_calibrated[i] = data_experiment_optimizer.T[:][i] * calibration

    res = optimize.minimize(optimize_spectrum, args=(distribution, data_calibrated, data_simulated, x_knots, energies_optimizer),
                            x0=weights_errors0, method='L-BFGS-B', bounds=bnds, options={'maxiter': 90})
    weights = res.x[:n]
    errors = res.x[n:]
    proton_bunch = np.array([distribution(x, x_knots, weights) for x in energies_optimizer])
    proton_bunch = proton_bunch / np.sum(proton_bunch)
    plt.bar(energies_optimizer, proton_bunch)
    plt.xlabel('Proton energy, MeV')
    plt.ylabel('Distribution')
    global number_of_spectrum_picture
    print('Global check ' + str(number_of_spectrum_picture))
    plt.savefig(os.path.join('pictures', 'spectrum_' + str(number_of_spectrum_picture) + '.png'))
    number_of_spectrum_picture = number_of_spectrum_picture + 1
    plt.close()

    return proton_bunch, errors
