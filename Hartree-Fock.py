# This code executes Restricted Hartree-Fock computations
#
# Author: Georgii N. Sizov
#
# Notes: Input file format is completely equivalent to
#        Gaussian input format with the following first line:
#        '# HF GEN Units = AU'

import numpy as np
import math as mt
import scipy as sp   # only need for taking a square root of a matrix and computing gamma functions


def input_reading(file):

    def atom_label(name):
        # returns a nuclei charge from atom label
        match name:
            case 'H':
                return 1
            case 'He':
                return 2
            case 'Li':
                return 3
            case 'Be':
                return 4
            case 'B':
                return 5
            case 'C':
                return 6
            case 'N':
                return 7
            case 'O':
                return 8
            case 'F':
                return 9
            case 'Ne':
                return 10

    def geometry(inpfile):  # return a matrix of nucleus charges and coordinates
        g = [] # geometry is written here
        cur_line = 5
        while inpfile[cur_line] != '\n':
            tmp = np.zeros((1, 4))
            line = str.split(inpfile[cur_line])
            k = 1 / 0.52917721054482  # coefficient to recalculate angstrom into bohr
            #k = 1
            tmp[0, 0] = float(atom_label(line[0]))  # atomic charge
            tmp[0, 1] = float(line[1]) * k  # x coordinate
            tmp[0, 2] = float(line[2]) * k  # y coordinate
            tmp[0, 3] = float(line[3]) * k  # z coordinate
            g.append(tmp)
            cur_line += 1
        g = np.concatenate(g)
        return g

    def electron_number(geom, charge):
        atom_number = int(geom.size / 4)  # a total number of nuclei
        positive_charge = 0
        for i in range(atom_number):
            positive_charge += geom[i, 0]
        return positive_charge - charge

    def xyz_coordinates(geom, charge):
        # We may have several nuclei of the same charge.
        # Each nucleus has its own set of basis functions
        # this functions is needed for executing these basis functions
        number = 0  # a number of nuclei with a given charge
        k = int(geom.size / 4)
        coordinates = []
        for i in range(k):
            xyz = np.zeros((1, 3))
            if geom[i, 0] == charge:
                xyz[0, :] = geom[i, 1:4]
                coordinates.append(xyz)
                number += 1
        return coordinates, number

    def basis_function(inpfile, current_line, xyz):

        def s_type(k_prim, inpfile, cline, xyz_s):
            exp_tmp = np.zeros((k_prim, 8))
            for i in range(k_prim):
                tmp = [float(s) for s in str.split(inpfile[cline + i])]
                exp_tmp[i, 0] = tmp[0]  # exponent
                exp_tmp[i, 1] = mt.pow(2 * tmp[0] / mt.pi, 0.75) * tmp[1]
                exp_tmp[i, 5:8] = xyz_s[0, 0:3]
            return exp_tmp, current_line + k_prim + 1, 1, k_prim

        def p_type(k_prim, inpfile, cline, xyz_p):
            exp_tmp_x = np.zeros((k_prim, 8))
            exp_tmp_y = np.zeros((k_prim, 8))
            exp_tmp_z = np.zeros((k_prim, 8))
            exp_tmp = []
            for i in range(k_prim):
                tmp = [float(s) for s in str.split(inpfile[cline + i])]

                exp_tmp_x[i, 0] = tmp[0]  # exponent
                exp_tmp_x[i, 1] = mt.pow(2 * tmp[0] / mt.pi, 0.75) * mt.pow(4.0 * tmp[0], 0.5) * tmp[1]
                exp_tmp_x[i, 2] = 1
                exp_tmp_x[i, 5:8] = xyz_p[0, 0:3]

                exp_tmp_y[i, 0] = tmp[0]  # exponent
                exp_tmp_y[i, 1] = mt.pow(2 * tmp[0] / mt.pi, 0.75) * mt.pow(4.0 * tmp[0], 0.5) * tmp[1]
                exp_tmp_y[i, 3] = 1
                exp_tmp_y[i, 5:8] = xyz_p[0, 0:3]

                exp_tmp_z[i, 0] = tmp[0]  # exponent
                exp_tmp_z[i, 1] = mt.pow(2 * tmp[0] / mt.pi, 0.75) * mt.pow(4.0 * tmp[0], 0.5) * tmp[1]
                exp_tmp_z[i, 4] = 1
                exp_tmp_z[i, 5:8] = xyz_p[0, 0:3]

            exp_tmp.append(exp_tmp_x)
            exp_tmp.append(exp_tmp_y)
            exp_tmp.append(exp_tmp_z)
            return exp_tmp, current_line + k_prim + 1, 3, k_prim

        def d_type(k_prim, inpfile, cline, xyz_d):
            exp_tmp_z2 = np.zeros((3 * k_prim, 8))
            exp_tmp_xz = np.zeros((k_prim, 8))
            exp_tmp_yz = np.zeros((k_prim, 8))
            exp_tmp_x2_y2 = np.zeros((2 * k_prim, 8))
            exp_tmp_xy = np.zeros((k_prim, 8))
            exp_tmp = []
            for i in range(k_prim):
                tmp = [float(s) for s in str.split(inpfile[cline + i])]
                # tmp[0] is an exponent; tmp[1] is a contraction coef
                exp_tmp_z2[3 * i, 0] = tmp[0]  # exponent
                coef1 = (2 ** (7 / 4)) / mt.sqrt(3) * (tmp[0] ** (7 / 4)) / (mt.pi ** (3 / 4))
                exp_tmp_z2[3 * i, 1] = 2 * tmp[1] * coef1
                exp_tmp_z2[3 * i, 4] = 2 # 2z ** 2
                exp_tmp_z2[3 * i, 5:8] = xyz_d[0, 0:3]
                exp_tmp_z2[3 * i + 1, 0] = tmp[0]  # exponent
                exp_tmp_z2[3 * i + 1, 1] = - tmp[1] * coef1
                exp_tmp_z2[3 * i + 1, 2] = 2  # - x ** 2
                exp_tmp_z2[3 * i + 1, 5:8] = xyz_d[0, 0:3]
                exp_tmp_z2[3 * i + 2, 0] = tmp[0]  # exponent
                exp_tmp_z2[3 * i + 2, 1] = - tmp[1] * coef1
                exp_tmp_z2[3 * i + 2, 3] = 2  # - y ** 2
                exp_tmp_z2[3 * i + 2, 5:8] = xyz_d[0, 0:3]

                exp_tmp_xz[i, 0] = tmp[0]  # exponent
                exp_tmp_xz[i, 1] = mt.pow(2 * tmp[0] / mt.pi, 0.75) * (4.0 * tmp[0]) * tmp[1]
                exp_tmp_xz[i, 2] = 1
                exp_tmp_xz[i, 4] = 1
                exp_tmp_xz[i, 5:8] = xyz_d[0, 0:3]

                exp_tmp_yz[i, 0] = tmp[0]  # exponent
                exp_tmp_yz[i, 1] = mt.pow(2 * tmp[0] / mt.pi, 0.75) * (4.0 * tmp[0]) * tmp[1]
                exp_tmp_yz[i, 3] = 1
                exp_tmp_yz[i, 4] = 1
                exp_tmp_yz[i, 5:8] = xyz_d[0, 0:3]

                exp_tmp_x2_y2[2 * i, 0] = tmp[0]  # exponent
                coef2 = (2 ** (7 / 4)) * (tmp[0] ** (7 / 4)) / (mt.pi ** (3 / 4))
                exp_tmp_x2_y2[2 * i, 1] = tmp[1] * coef2
                exp_tmp_x2_y2[2 * i, 2] = 2
                exp_tmp_x2_y2[2 * i, 5:8] = xyz_d[0, 0:3]
                exp_tmp_x2_y2[2 * i + 1, 0] = tmp[0]  # exponent
                exp_tmp_x2_y2[2 * i + 1, 1] = - tmp[1] * coef2
                exp_tmp_x2_y2[2 * i + 1, 3] = 2
                exp_tmp_x2_y2[2 * i + 1, 5:8] = xyz_d[0, 0:3]

                exp_tmp_xy[i, 0] = tmp[0]  # exponent
                exp_tmp_xy[i, 1] = mt.pow(2 * tmp[0] / mt.pi, 0.75) * (4.0 * tmp[0]) * tmp[1]
                exp_tmp_xy[i, 2] = 1
                exp_tmp_xy[i, 3] = 1
                exp_tmp_xy[i, 5:8] = xyz_d[0, 0:3]

            exp_tmp.append(exp_tmp_z2)
            exp_tmp.append(exp_tmp_xz)
            exp_tmp.append(exp_tmp_yz)
            exp_tmp.append(exp_tmp_x2_y2)
            exp_tmp.append(exp_tmp_xy)
            return exp_tmp, current_line + k_prim + 1, 5, k_prim

        def f_type(k_prim, inpfile, cline, xyz_f):
            exp_tmp_z3 = np.zeros((3 * k_prim, 8))  # 2 z^3 - 3 x^2 z - 3 y^2 z
            exp_tmp_xz2 = np.zeros((3 * k_prim, 8))  # 4 z^2 x - x^3 - x y^2
            exp_tmp_yz2 = np.zeros((3 * k_prim, 8))  # 4 y z^2 - x^2 y - y^3
            exp_tmp_zx2_zy2 = np.zeros((2 * k_prim, 8))  # z x^2 - z y^2
            exp_tmp_xyz = np.zeros((k_prim, 8))  # xyz
            exp_tmp_x3_3xy2 = np.zeros((2 * k_prim, 8))  # x^3 - 3 y^2 x
            exp_tmp_3yx2_y3 = np.zeros((2 * k_prim, 8))  # 3 x^2 y - y^3
            exp_tmp = []
            for i in range(k_prim):
                tmp = [float(s) for s in str.split(inpfile[cline + i])]
                # tmp[0] is an exponent; tmp[1] is a contraction coef
                exp_tmp_z3[3*i : 3*i+3, 0] = tmp[0] # exponent
                # coefficients
                coef1 = (2 ** (11 / 4)) * (tmp[0] ** (9 / 4)) / (15 ** (1 / 2)) / (mt.pi ** (3 / 4))
                exp_tmp_z3[3 * i, 1] = 2 * tmp[1] * coef1
                exp_tmp_z3[3 * i + 1, 1] = - 3 * tmp[1] * coef1
                exp_tmp_z3[3 * i + 2, 1] = - 3 * tmp[1] * coef1
                # powers
                exp_tmp_z3[3 * i, 4] = 3 # 2 z^3
                exp_tmp_z3[3 * i + 1, 2] = 2  # - 3 x^2 z
                exp_tmp_z3[3 * i + 1, 4] = 1  # - 3 x^2 z
                exp_tmp_z3[3 * i + 2, 3] = 2  # - 3 y^2 z
                exp_tmp_z3[3 * i + 2, 4] = 1  # - 3 y^2 z
                #geometry
                exp_tmp_z3[3 * i:3 * i + 3, 5:8] = xyz_f[0, :3]


                exp_tmp_xz2[3 * i: 3 * i + 3, 0] = tmp[0]  # exponent
                # coefficients
                coef1 = (2 ** (7 / 4)) * (10 ** (1 / 2)) * (tmp[0] ** (9 / 4)) / 5 / (mt.pi ** (3 / 4))
                exp_tmp_xz2[3 * i, 1] = 4 * tmp[1] * coef1  # 4 z^2 x
                exp_tmp_xz2[3 * i + 1, 1] = - tmp[1] * coef1  # - x^3
                exp_tmp_xz2[3 * i + 2, 1] = - tmp[1] * coef1  # - x y^2
                # powers
                exp_tmp_xz2[3 * i, 4] = 2  # 4 z^2 x
                exp_tmp_xz2[3 * i, 2] = 1  # 4 z^2 x
                exp_tmp_xz2[3 * i + 1, 2] = 3  # - x^3
                exp_tmp_xz2[3 * i + 2, 2] = 1  # - x y^2
                exp_tmp_xz2[3 * i + 2, 3] = 2  # - x y^2
                # geometry
                exp_tmp_xz2[3 * i:3 * i + 3, 5:8] = xyz_f[0, :3]


                exp_tmp_yz2[3 * i: 3 * i + 3, 0] = tmp[0]  # exponent
                # coefficients
                coef1 = (2 ** (7 / 4)) * (10 ** (1 / 2)) * (tmp[0] ** (9 / 4)) / 5 / (mt.pi ** (3 / 4))
                exp_tmp_yz2[3 * i, 1] = 4 * tmp[1] * coef1  # 4 y z^2
                exp_tmp_yz2[3 * i + 1, 1] = - tmp[1] * coef1  # - x^2 y
                exp_tmp_yz2[3 * i + 2, 1] = - tmp[1] * coef1  # - y^3
                # powers
                exp_tmp_yz2[3 * i, 3] = 1  # 4 y z^2
                exp_tmp_yz2[3 * i, 4] = 2  # 4 y z^2
                exp_tmp_yz2[3 * i + 1, 2] = 2  # - x^2 y
                exp_tmp_yz2[3 * i + 1, 3] = 1  # - x^2 y
                exp_tmp_yz2[3 * i + 2, 3] = 3  # - y^3
                # geometry
                exp_tmp_yz2[3 * i:3 * i + 3, 5:8] = xyz_f[0, :3]


                exp_tmp_zx2_zy2[3 * i: 3 * i + 2, 0] = tmp[0]  # exponent
                # coefficients
                coef1 = (2 ** (11 / 4)) * (tmp[0] ** (9 / 4)) / (mt.pi ** (3 / 4))
                exp_tmp_zx2_zy2[3 * i, 1] = tmp[1] * coef1  # z x^2
                exp_tmp_zx2_zy2[3 * i + 1, 1] = - tmp[1] * coef1  # - z y^2
                # powers
                exp_tmp_zx2_zy2[3 * i, 2] = 2  # z x^2
                exp_tmp_zx2_zy2[3 * i, 4] = 1  # z x^2
                exp_tmp_zx2_zy2[3 * i + 1, 3] = 2  # - z y^2
                exp_tmp_zx2_zy2[3 * i + 1, 4] = 1  # - z y^2
                # geometry
                exp_tmp_zx2_zy2[3 * i:3 * i + 2, 5:8] = xyz_f[0, :3]


                exp_tmp_xyz[3 * i: 3 * i + 1, 0] = tmp[0]  # exponent
                # coefficients
                coef1 = (2 ** (15 / 4)) * (tmp[0] ** (9 / 4)) / (mt.pi ** (3 / 4))
                exp_tmp_xyz[3 * i, 1] = tmp[1] * coef1  # x y z
                # powers
                exp_tmp_xyz[3 * i, 2] = 1  # x y z
                exp_tmp_xyz[3 * i, 3] = 1  # x y z
                exp_tmp_xyz[3 * i, 4] = 1  # x y z
                # geometry
                exp_tmp_xyz[3 * i:3 * i + 2, 5:8] = xyz_f[0, :3]


                exp_tmp_x3_3xy2[3 * i: 3 * i + 2, 0] = tmp[0]  # exponent
                # coefficients
                coef1 = (2 ** (7 / 4)) * (6 ** (1 / 4)) * (tmp[0] ** (9 / 4)) / 3 / (mt.pi ** (3 / 4))
                exp_tmp_x3_3xy2[3 * i, 1] = tmp[1] * coef1  # x^3
                exp_tmp_x3_3xy2[3 * i + 1, 1] = - 3 * tmp[1] * coef1  # - 3 y^2 x
                # powers
                exp_tmp_x3_3xy2[3 * i, 2] = 3  # x^3
                exp_tmp_x3_3xy2[3 * i + 1, 2] = 1  # - 3 y^2 x
                exp_tmp_x3_3xy2[3 * i + 1, 3] = 2  # - 3 y^2 x
                # geometry
                exp_tmp_x3_3xy2[3 * i:3 * i + 2, 5:8] = xyz_f[0, :3]


                exp_tmp_3yx2_y3[3 * i: 3 * i + 2, 0] = tmp[0]  # exponent
                # coefficients
                coef1 = (2 ** (7 / 4)) * (6 ** (1 / 4)) * (tmp[0] ** (9 / 4)) / 3 / (mt.pi ** (3 / 4))
                exp_tmp_3yx2_y3[3 * i, 1] = 3 * tmp[1] * coef1  # x^3
                exp_tmp_3yx2_y3[3 * i + 1, 1] = - tmp[1] * coef1  # - 3 y^2 x
                # powers
                exp_tmp_3yx2_y3[3 * i, 2] = 2  # 3 x^2 y
                exp_tmp_3yx2_y3[3 * i, 3] = 1  # 3 x^2 y
                exp_tmp_3yx2_y3[3 * i + 1, 3] = 3  # - y^3
                # geometry
                exp_tmp_3yx2_y3[3 * i:3 * i + 2, 5:8] = xyz_f[0, :3]


            exp_tmp.append(exp_tmp_z3)
            exp_tmp.append(exp_tmp_xz2)
            exp_tmp.append(exp_tmp_yz2)
            exp_tmp.append(exp_tmp_zx2_zy2)
            exp_tmp.append(exp_tmp_xyz)
            exp_tmp.append(exp_tmp_x3_3xy2)
            exp_tmp.append(exp_tmp_3yx2_y3)
            return exp_tmp, current_line + k_prim + 1, 7, k_prim

        head_line = str.split(inpfile[current_line])
        gen_tmp = np.zeros((1, 2))
        gen_tmp[0, 1] = float(head_line[1])
        match head_line[0]:
            case 'S':
                return s_type(int(gen_tmp[0, 1]), inpfile, current_line + 1, xyz)
            case 'P':
                return p_type(int(gen_tmp[0, 1]), inpfile, current_line + 1, xyz)
            case 'D':
                return d_type(int(gen_tmp[0, 1]), inpfile, current_line + 1, xyz)
            case 'F':
                return f_type(int(gen_tmp[0, 1]), inpfile, current_line + 1, xyz)

    file_lines = file.readlines()
    current_line = 4  # current executed line
    # following Gaussian format 1s line is '# HF GEN Units = AU',
    # 2nd and 4th lines are blank, 3rd line is a job name
    charge, multiplicity = [int(s) for s in str.split(file_lines[current_line])]
    geom = geometry(file_lines)  # geometry matrix
    n = int(electron_number(geom, charge))  # counts a number of electrons
    current_line += int(geom.size / 4) + 2
    gen = []  # encodes an order of basis functions and a number of primitives in each basis function
    exp = []  # encodes primitives
    current_basis_func = 1
    while file_lines[current_line] != '\n':
        title_line = str.split(file_lines[current_line])
        nuclei_charge = atom_label(title_line[0])
        xyz, nuclei_number = xyz_coordinates(geom, nuclei_charge)  # coordinates of specific type of atoms
        # nuclei_number is a number of atoms of a specific type
        current_line += 1
        initial_line = current_line
        considered_atom = 1
        while nuclei_number >= considered_atom:
            current_line = initial_line
            while file_lines[current_line] != '****\n':
                exp_tmp, current_line_change, numb, prim = basis_function(file_lines, current_line,
                                                                          xyz[considered_atom - 1])
                if numb == 1:  # adding s-type
                    gen_tmp = np.zeros((1, 2))
                    gen_tmp[0, 0] = current_basis_func
                    gen_tmp[0, 1] = int(prim)
                    current_basis_func += 1
                    gen.append(gen_tmp)
                    exp.append(exp_tmp)
                elif numb == 3:  # adding p-type
                    gen_tmp = np.zeros((3, 2))
                    gen_tmp[0, 0] = current_basis_func
                    gen_tmp[0, 1] = int(prim)
                    gen_tmp[1, 0] = current_basis_func + 1
                    gen_tmp[1, 1] = int(prim)
                    gen_tmp[2, 0] = current_basis_func + 2
                    gen_tmp[2, 1] = int(prim)
                    gen.append(gen_tmp)
                    for j in range(numb):
                        exp.append(exp_tmp[j])
                    current_basis_func = current_basis_func + 3
                elif numb == 5:  # adding d-type
                    gen_tmp = np.zeros((5, 2))
                    gen_tmp[0, 0] = current_basis_func
                    gen_tmp[0, 1] = 3 * int(prim)
                    gen_tmp[1, 0] = current_basis_func + 1
                    gen_tmp[1, 1] = int(prim)
                    gen_tmp[2, 0] = current_basis_func + 2
                    gen_tmp[2, 1] = int(prim)
                    gen_tmp[3, 0] = current_basis_func + 3
                    gen_tmp[3, 1] = 2 * int(prim)
                    gen_tmp[4, 0] = current_basis_func + 4
                    gen_tmp[4, 1] = int(prim)
                    gen.append(gen_tmp)
                    for j in range(numb):
                        exp.append(exp_tmp[j])
                    current_basis_func = current_basis_func + 5
                elif numb == 7:  # adding d-type
                    gen_tmp = np.zeros((7, 2))
                    gen_tmp[0, 0] = current_basis_func
                    gen_tmp[0, 1] = 3 * int(prim)
                    gen_tmp[1, 0] = current_basis_func + 1
                    gen_tmp[1, 1] = 3 * int(prim)
                    gen_tmp[2, 0] = current_basis_func + 2
                    gen_tmp[2, 1] = 3 * int(prim)
                    gen_tmp[3, 0] = current_basis_func + 3
                    gen_tmp[3, 1] = 2 * int(prim)
                    gen_tmp[4, 0] = current_basis_func + 4
                    gen_tmp[4, 1] = int(prim)
                    gen_tmp[5, 0] = current_basis_func + 5
                    gen_tmp[5, 1] = 2 * int(prim)
                    gen_tmp[6, 0] = current_basis_func + 6
                    gen_tmp[6, 1] = 2 * int(prim)
                    gen.append(gen_tmp)
                    for j in range(numb):
                        exp.append(exp_tmp[j])
                    current_basis_func = current_basis_func + 7
                current_line = current_line_change
            considered_atom += 1
        current_line += 1
    gen = np.concatenate(gen)
    k = int(gen.size)
    return n, int(k / 2), geom, gen, exp


# *********************************************#
#                                              #
#          MOLECULAR INTEGRALS BLOCK           #
#                                              #
# *********************************************#


def E(i, j, t, xpa, xpb, p, k):
    # gives coefficients in the expansion of 1-D gaussian products in terms of hermitian polynomials
    if t > 0:
        return 1 / (2 * p * t) * (i * E(i - 1, j, t - 1, xpa, xpb, p, k) + j * E(i, j - 1, t - 1, xpa, xpb, p, k))
    elif i > 0:

        return xpa * E(i - 1, j, 0, xpa, xpb, p, k) + E(i - 1, j, 1, xpa, xpb, p, k)
    elif j > 0:
        return xpb * E(i, j - 1, 0, xpa, xpb, p, k) + E(i, j - 1, 1, xpa, xpb, p, k)
    else:
        return k


def one_dim_overlap(f1, f2, q):
    # one-dimensional overlap
    # if q = 1 then x
    # if q = 2 then y
    # if q = 3 then z

    i = int(f1[1 + q])
    j = int(f2[1 + q])  # i in gaussians
    # expansion coefficients
    p = f1[0] + f2[0]
    mu = f1[0] * f2[0] / (f1[0] + f2[0])
    # x coefs
    pq = (f1[0] * f1[4 + q] + f2[0] * f2[4 + q]) / p
    qpa = pq - f1[4 + q]
    qpb = pq - f2[4 + q]
    qab = f1[4 + q] - f2[4 + q]
    kq = mt.exp(- mu * qab ** 2)
    return E(i, j, 0, qpa, qpb, p, kq) * mt.sqrt(mt.pi / p)


def overlap_prim(f1, f2):
    # compute three-dimensional overlaps
    sx = one_dim_overlap(f1, f2, 1)
    sy = one_dim_overlap(f1, f2, 2)
    sz = one_dim_overlap(f1, f2, 3)
    return f1[1] * f2[1] * sx * sy * sz


def kinetic_prim(f1, f2):
    # computes kinetic energy matrix elements between primitives

    def one_kin(f1, f2, q):
        # computes one-dimensional kinetic matrix elements
        # q = 1 is along x axis
        # q = 2 (y axis), q = 3 (z axis)
        i = int(f1[1 + q])
        j = int(f2[1 + q])  # i in gaussians
        # expansion coefficients
        p = f1[0] + f2[0]
        mu = f1[0] * f2[0] / (f1[0] + f2[0])
        # q coefs
        pq = (f1[0] * f1[4 + q] + f2[0] * f2[4 + q]) / p
        qpa = pq - f1[4 + q]
        qpb = pq - f2[4 + q]
        qab = f1[4 + q] - f2[4 + q]
        kq = mt.exp(- mu * qab ** 2)
        a = - 2 * f2[0] ** 2 * E(i, j + 2, 0, qpa, qpb, p, kq)
        b = f2[0] * (2 * j + 1) * E(i, j, 0, qpa, qpb, p, kq)
        c = - j * (j - 1) / 2 * E(i, j - 2, 0, qpa, qpb, p, kq)
        return (a + b + c) * mt.sqrt(mt.pi / p)

    term1 = one_kin(f1, f2, 1) * one_dim_overlap(f1, f2, 2) * one_dim_overlap(f1, f2, 3)
    term2 = one_dim_overlap(f1, f2, 1) * one_kin(f1, f2, 2) * one_dim_overlap(f1, f2, 3)
    term3 = one_dim_overlap(f1, f2, 1) * one_dim_overlap(f1, f2, 2) * one_kin(f1, f2, 3)
    return f1[1] * f2[1] * (term1 + term2 + term3)


def boys_function(n, x):
    # it computes a value of the boys function F_n(x)
    if x != 0 :
        p = sp.special.gammainc(n + 0.5, x)  # incomplete gamma functions
        g = sp.special.gamma(n + 0.5)
        return g * p / (2 * x **(n + 0.5))
    else:
        return 1 / (2 * n + 1)


def R(t, u, v, n, p, x, y, z, pR2):
    # computes R_ijk^n integrals
    if t >= 2:
        return (t - 1) * R(t - 2, u, v, n + 1, p, x, y, z, pR2) + x * R(t - 1, u, v, n + 1, p, x, y, z, pR2)
    elif u >= 2:
        return (u - 1) * R(t, u - 2, v, n + 1, p, x, y, z, pR2) + y * R(t, u - 1, v, n + 1, p, x, y, z, pR2)
    elif v >= 2:
        return (v - 1) * R(t, u, v - 2, n + 1, p, x, y, z, pR2) + z * R(t, u, v - 1, n + 1, p, x, y, z, pR2)
    else:
        return x ** t * y ** u * z ** v * (-2 * p) ** (n + t + u + v) * boys_function(n + t + u + v, pR2)


def Etuv(f1, f2):
    # compute three-dimensional arrow Etuv = Et * Eu * Ev
    i = int(f1[2])
    j = int(f2[2])  # i in gaussians
    k = int(f1[3])
    l = int(f2[3])  # j in gaussians
    m = int(f1[4])
    n = int(f2[4])  # k in gaussians
    # expansion coefficients
    p = f1[0] + f2[0]
    mu = f1[0] * f2[0] / (f1[0] + f2[0])
    # x coefs
    px = (f1[0] * f1[5] + f2[0] * f2[5]) / p
    xpa = px - f1[5]
    xpb = px - f2[5]
    xab = f1[5] - f2[5]
    kx = mt.exp(- mu * xab ** 2)
    Eij = np.zeros(i + j + 1)
    for q in range(i + j + 1):
        Eij[q] = E(i, j, q, xpa, xpb, p, kx)
    # y coefs
    py = (f1[0] * f1[6] + f2[0] * f2[6]) / p
    ypa = py - f1[6]
    ypb = py - f2[6]
    yab = f1[6] - f2[6]
    ky = mt.exp(- mu * yab ** 2)
    Ekl = np.zeros(k + l + 1)
    for q in range(k + l + 1):
        Ekl[q] = E(k, l, q, ypa, ypb, p, ky)
    # z coefs
    pz = (f1[0] * f1[7] + f2[0] * f2[7]) / p
    zpa = pz - f1[7]
    zpb = pz - f2[7]
    zab = f1[7] - f2[7]
    kz = mt.exp(- mu * zab ** 2)
    Emn = np.zeros(m + n + 1)
    for q in range(m + n + 1):
        Emn[q] = E(m, n, q, zpa, zpb, p, kz)
    E3 = np.zeros((i + j + 1, k + l + 1, m + n + 1))
    for q in range(i + j + 1):
        for a in range(k + l + 1):
            for s in range(m + n + 1):
                E3[q, a, s] = Eij[q] * Ekl[a] * Emn[s]
    return E3, i + j + 1, k + l + 1, m + n + 1, p, px, py, pz  # COMPLETE


def elnucl_prim(f1, f2, geom):
    # it takes two gaussian primitives and nucleus coordinates
    # it computes electron-nucleus matrix elements
    # Omega12 = E^(ij)_t * E^(kl)_u * E^(mn)_v * L_tuv

    def elnucl_hermite(t, u, v, p, px, py, pz, cx, cy, cz, Z):
        # electron-nucleus interaction for Hermite Gaussians
        # t, u, v are hermite polynomial powers
        # p is a gaussian exponent
        # px, py, pz are coordinates of a basis function center
        # cx, cy, cz are nucleus coordinates
        # Z is a nucleus charge
        xpc = px - cx
        ypc = py - cy
        zpc = pz - cz
        pR2 = p * (xpc ** 2 + ypc ** 2 + zpc ** 2)
        return - Z * 2 * mt.pi / p * R(t, u, v, 0, p, xpc, ypc, zpc, pR2)

    K = int(geom.size / 4)
    E3, ij, kl, mn, p, px, py, pz = Etuv(f1, f2)
    result = 0
    for q in range(K):
        for t in range(ij):
            for u in range(kl):
                for v in range(mn):
                    result += E3[t, u, v] * elnucl_hermite(t, u, v, p, px, py, pz, geom[q, 1], geom[q, 2], geom[q, 3], geom[q, 0])
    return result * f1[1] * f2[1]


def matrix_element(prim1, f1, prim2, f2, geom, type):
    # computes matrix elements between contracted basis functions
    # type = 1 overlap
    # type = 2 kinetic
    # type = 3 electron-nuclei
    M12 = 0  # a total matrix element
    match type:
        case 1:
            for i in range(int(prim1)):
                for j in range(int(prim2)):
                    M12 += overlap_prim(f1[i, :], f2[j, :])
        case 2:
            for i in range(int(prim1)):
                for j in range(int(prim2)):
                    M12 += kinetic_prim(f1[i, :], f2[j, :])
        case 3:
            for i in range(int(prim1)):
                for j in range(int(prim2)):
                    M12 += elnucl_prim(f1[i, :], f2[j, :], geom)
    return M12


def matrix(gen, exp, geom, type):
    # computes matrix elements between contracted basis functions
    # type = 1 overlap
    # type = 2 kinetic
    # type = 3 electron-nuclei
    # returns a kinetic energy matrix

    K = int(gen.size / 2)  # a matrix size
    T = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            T[i, j] = matrix_element(gen[i, 1], exp[i], gen[j, 1], exp[j], geom, type)
    return T


def elel_tensor(gen, exp):


    def elel_bf(prim1, f1, prim2, f2, prim3, f3, prim4, f4):
        # electron-electron interactions between contracted basis functions

        def elel_prim(f1, f2, f3, f4):
            # electron-electron interactions between primitives

            def elel_hermite(t1, u1, v1, p1, px1, py1, pz1, t2, u2, v2, p2, px2, py2, pz2):
                # electron-electron interaction for hermite gaussians
                # t, u, v are hermite polynomial parameters
                # p is an exponent
                # px, py, pz are coordinates of a basis function center
                x = px1 - px2
                y = py1 - py2
                z = pz1 - pz2
                a = p1 * p2 / (p1 + p2)
                pR2 = a * (x ** 2 + y ** 2 + z ** 2)
                return (-1) ** (t2 + u2 + v2) * 2 * mt.pi ** (5 / 2) / (p1 * p2 * mt.sqrt(p1 + p2)) * R(t1 + t2,
                                                                                                        u1 + u2,
                                                                                                        v1 + v2, 0, a,
                                                                                                        x, y, z, pR2)

            E31, ij1, kl1, mn1, p1, px1, py1, pz1 = Etuv(f1, f2)
            E32, ij2, kl2, mn2, p2, px2, py2, pz2 = Etuv(f3, f4)
            result = 0
            for t1 in range(ij1):
                for u1 in range(kl1):
                    for v1 in range(mn1):
                        for t2 in range(ij2):
                            for u2 in range(kl2):
                                for v2 in range(mn2):
                                    result += E31[t1, u1, v1] * E32[t2, u2, v2] * elel_hermite(t1, u1, v1, p1, px1, py1,
                                                                                               pz1, t2, u2, v2, p2, px2,
                                                                                               py2, pz2)
            return result * f1[1] * f2[1] * f3[1] * f4[1]

        M = 0
        for i in range(int(prim1)):
            for j in range(int(prim2)):
                for k in range(int(prim3)):
                    for l in range(int(prim4)):
                        M += elel_prim(f1[i, :], f2[j, :], f3[k, :], f4[l, :])
        return M


    # computes a tensor of electron-electron interactions
    K = int(gen.size / 2)  # a matrix size
    T = np.zeros((K, K, K, K))
    for i in range(K):
        print(i/K * 100, "% of tensor are computed ")
        for j in range(K):
            for k in range(K):
                for l in range(K):
                    T[i, j, k, l] = elel_bf(gen[i, 1], exp[i], gen[j, 1], exp[j], gen[k, 1], exp[k], gen[l, 1], exp[l])
    return T


def normalization(gen, exp):
    # normalizes basis functions if they are not
    k = int(gen.size / 2)
    for i in range(k):
        C2_inv = matrix_element(gen[i, 1], exp[i], gen[i, 1], exp[i], 0, 1)
        C = 1 / mt.sqrt(C2_inv)  # normalization factor
        for j in range(int(gen[i, 1])):
            exp[i][j, 1] = exp[i][j, 1] * C
    return exp


def orthogonalization(ovrlp, kin, elnuc):
    #  recomputes matrices in an orthonormal basis set
    C = np.linalg.inv(sp.linalg.sqrtm(ovrlp))  # matrix makes a basis orthogonal
    kin1 = np.matmul(C.T, np.matmul(kin, C))
    elnuc1 = np.matmul(C.T, np.matmul(elnuc, C))
    return C, kin1, elnuc1


def tensor_orthogonalization(C, elel):
    # C is an orthogonalalizing matrix
    K = int(elel.size ** 0.25)
    elel1 = np.zeros((K, K, K, K))
    for n in range(K):
        print(" n = ", n)
        for m in range(K):
            for p in range(K):
                for t in range(K):
                    for i in range(K):
                        for j in range(K):
                            for k in range(K):
                                for l in range(K):
                                    elel1[n, m, p, t] += C[i, n] * C[j, m] * C[k, p] * C[l, t] * elel[i, j, k, l]
    return elel1


def matrix_computation(file):
    # Data for 1-RDM construction
    n, k, geom, gen, exp = input_reading(file)
    exp = normalization(gen, exp)  # normalization of basis functions if they are not
    s = matrix(gen,exp, geom, 1)  # overlap matrix
    c = np.linalg.inv(sp.linalg.sqrtm(s))
    kin = matrix(gen, exp, geom, 2)  # kinetic energy matrix
    elnucl = matrix(gen, exp, geom, 3)  # electron-nuclei interaction matrix
    #c, kin1, elnucl1 = orthogonalization(s, kin, elnucl)  # recomputing matrices within orthogonal basis set
    ten = elel_tensor(gen, exp)
    #ten1 = tensor_orthogonalization(c, ten)  # recomputing the tensor
    return n, k, geom, gen, exp, c, kin, elnucl, kin + elnucl, ten


# *********************************************#
#                                              #
#   Fock matrix and energy evaluation block    #
#                                              #
# *********************************************#

def density_matrix(MOs, N, K):
    # computes a density matrix for a given set of MO coefficients
    P = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            for t in range(int(N / 2)):
                P[i, j] += 2 * MOs[i, t] * MOs[j, t]
    return P


def two_electron_part(Ten, P, K):
    # computes two electron part of the Fock matrix
    # for some selected MO coefficients from a matrix 'MOs'.
    # N / 2 is a number of orbitals. So, the code takes only
    # first N / 2 columns of MOs matrix
    G = np.zeros((K, K))
    for k in range(K):
        for l in range(K):
            for i in range(K):
                for j in range(K):
                    G[k, l] += P[i, j] * (Ten[k, l, j, i] - Ten[k, i, j, l] / 2)
    return G


def Fock_matrix(MOs, H, Ten, N, K):
    # computes the Fock matrix for given MO coefficients
    P = density_matrix(MOs, N, K)
    F = H + two_electron_part(Ten, P, K)
    return F


def nuclei_energy(geom):

    def coll_ind(n):
        # collective index
        # it takes its number n and return ij
        n += 1
        a = (mt.sqrt(1 + 8 * n) - 1) / 2
        j = mt.ceil(a)
        lb = (j - 1) * j / 2
        i = int(n - lb)
        return i - 1, j - 1

    N = int(geom.size / 4)  # a number of nuclei
    M = int(N * (N + 1) / 2)  # atomic pair including repeating
    E = 0  # nuclei energy
    for k in range(M):
        i, j = coll_ind(k)
        if i != j :
            x = geom[i, 1] - geom[j, 1]
            y = geom[i, 2] - geom[j, 2]
            z = geom[i, 3] - geom[j, 3]
            r = mt.sqrt(x ** 2 + y ** 2 + z ** 2)
            E += geom[i, 0] * geom[j, 0] / r
    return E


def total_energy(MOs, H, F, N, K, geom):
    # computes energy at each itteration for a given Fock matrix
    E = 0
    P = density_matrix(MOs, N, K)
    for i in range(K):
        for j in range(K):
            E += P[i, j] * (H[i, j] + F[i, j])
    return E / 2 + nuclei_energy(geom)


# *********************************************#
#                                              #
#                    SCF BLOCK                 #
#                                              #
# *********************************************#


def generalized_eig(H, C):
    # solves the generalized eigenvalue problem
    H_ort = np.matmul(C.T, np.matmul(H, C))
    E_orb = np.linalg.eig(H_ort)[0]
    MOs_ort = np.linalg.eig(H_ort)[1]
    idx = E_orb.argsort()[::1]
    E_orb = E_orb[idx]
    MOs_ort = MOs_ort[:, idx]
    return C @ MOs_ort, E_orb


def SCF_cycles(Hcore, Ten, N, K, Geom, C, eps):
    # performs SCF calculations
    MOs, E_orb = generalized_eig(Hcore, C)  # initial guess
    print("MOs", MOs)
    F = Fock_matrix(MOs, Hcore, Ten, N, K)
    E0 = total_energy(MOs, Hcore, F, N, K, Geom)
    delta_E = np.absolute(E0)  # initial energy
    i = 0  # step
    while delta_E > eps:
        F = Fock_matrix(MOs, Hcore, Ten, N, K)
        MOs, E_orb = generalized_eig(F, C)  # MOs[:, k] is a k-th eigenvalue
        E1 = total_energy(MOs, Hcore, F, N, K, Geom)
        print("step i = ", i, " Energy is ", E1, "delta_E ", delta_E)
        delta_E = np.absolute(E1 - E0)
        E0 = E1; i += 1
    print("Total energy is",  E0)
    print("orbital energies")
    print(E_orb)
    print("MO coefficients")
    print(MOs)
    return F

file = open(r"C:\Users\georg\Hartree-Fock\Input.txt", 'r')
N, K, Geom, Gen, Exp, C, Kin, Vext, Hcore, Ten = matrix_computation(file)
file.close()
F = SCF_cycles(Hcore, Ten, N, K, Geom, C, 10 ** (-8))





