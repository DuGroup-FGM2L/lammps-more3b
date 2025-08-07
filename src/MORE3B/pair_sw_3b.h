/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(sw/3b,PairSW3B);
// clang-format on
#else

#ifndef LMP_PAIR_SW_3B_H
#define LMP_PAIR_SW_3B_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSW3B : public Pair {
 public:
  PairSW3B(class LAMMPS *);
  ~PairSW3B() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;

  static constexpr int MAX_LINE_PARAMS = 12;
  static constexpr int MIN_LINE_PARAMS = 9;

  struct Param {
    double epsilon, lambda, costheta;
    double a_ij, a_ik, gamma_ij, gamma_ik, sigma_ij, sigma_ik;
    int ielement, jelement, kelement;
    int is_artificial, is_zero;
  };

  struct MaxParamLengths {
    int epsilon, decepsilon;
    int lambda, declambda;
    int costheta, deccostheta;
    int a_ij, deca_ij;
    int a_ik, deca_ik;
    int gamma_ij, decgamma_ij;
    int gamma_ik, decgamma_ik;
    int sigma_ij, decsigma_ij;
    int sigma_ik, decsigma_ik;
    int iname, jname, kname;
  };

 protected:
  double **cutmax;            // array of maximum cutoffs for each pair
  Param *params;              // parameter set for an I-J-K interaction
  int maxshort;               // size of short neighbor list array
  int *neighshort;            // short neighbor list array
  int params_mapped;          // whether parameters have been read and mapped to elements
  int use_symmetry;           // treat ABC and ACB triplets the same
  MaxParamLengths mparam;     // Single struct that will keep maximum string lengths


  void settings(int, char **) override;
  virtual void allocate();
  virtual void read_file(char *);
  virtual void setup_params();

  int count_decimal_digits(const std::string&);

  void threebody(Param *, double, double, double *, double *,
                       double *, double *, double *, int, double &);
};

}    // namespace LAMMPS_NS

#endif
#endif
