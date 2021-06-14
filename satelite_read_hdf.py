import numpy as np
import matplotlib.pyplot as plt
import tables
import matplotlib as mpl
import os

# PATH = "/home/zelenyy/data/satellite/wrapper/result_wrapper.hdf5"
# PATH = "/home/zelenyy/data/satellite/double_wrapper/result_double_wrapper.hd?f5"
PATH = os.path.join(os.path.abspath(os.curdir), 'Geant4_data', 'result_detail.hdf5')


def print_polystyrene_tavyk(path):
    # result = []
    energies = []
    with tables.open_file(path) as h5file:
        for group in h5file.root:
            table = h5file.get_node(group, "deposit")
            # data = table.read()
            # wrapper_data = h5file.get_node(group, "wrapper_deposit").read()
            energy = table.attrs["values_macros_energy"]
            energies.append(energy)
            # result.append((energy, data["event"].sum(axis=1).mean(), wrapper_data["event"].sum(axis=1).mean()))

    # for it in result:
    #     sum_ = it[1] + it[2]
    #     # print(it[0], 100 * it[1] / sum_, 100 * it[2] / sum_)

    return energies


def get_data_by_energy(path, my_energy=75.0):
    with tables.open_file(path) as h5file:
        for group in h5file.root:
            table = h5file.get_node(group, "deposit")
            energy = table.attrs["values_macros_energy"]
            number = table.attrs["values_macros_number"]
            if energy == my_energy:
                data = table.read()
                data = data["event"].sum(axis=0) / number
    return data


def obtain_geant4_data():
    energy = print_polystyrene_tavyk(PATH)
    data = []
    for i in range(len(energy)):
        data.append(get_data_by_energy(PATH, energy[i]))
    return energy, data


def main():
    energy = print_polystyrene_tavyk(PATH)
    data = []
    for i in range(len(energy)):
        data.append(get_data_by_energy(PATH, energy[i]))
    # print(data)
    plt.figure(figsize=(7, 7))
    plt.title("Протон, " + str(energy[0]) + " МэВ")
    n = 15
    plt.bar(range(1, n + 1), data[0][:n])
    plt.xlabel("Номер шайбы")
    plt.ylabel("Энергия, МэВ")
    plt.savefig(os.path.join("Geant4_data", "proton_75_MeV.png"))
    return 0


if __name__ == '__main__':
    main()
