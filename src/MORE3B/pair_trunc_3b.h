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
PairStyle(trunc/3b,PairTrunc3B);
// clang-format on
#else

#ifndef LMP_PAIR_TRUNC_3B_H
#define LMP_PAIR_TRUNC_3B_H

#include "pair.h"

namespace LAMMPS_NS {

class PairTrunc3B : public Pair {
 public:
  PairTrunc3B(class LAMMPS *);
  ~PairTrunc3B() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;

  static constexpr int LINE_PARAMS = 6;

  struct Param {
    int ielement, jelement, kelement;
    double k, costheta0, rho;
  };

 protected:
  double cutmax;              // three-body term cutoff
  Param *params;              // parameter set for an I-J-K interaction
  int maxshort;               // size of short neighbor list array
  int *neighshort;            // short neighbor list array
  int params_mapped;          // whether parameters have been read and mapped to elements
  int use_symmetry;           // treat ABC and ACB triplets the same

  void settings(int, char **) override;
  virtual void allocate();
  virtual void read_file(char *);
  virtual void setup_params();

  void threebody(Param *, double, double, double *, double *,
                       double *, double *, double *, int, double &);
};

}    // namespace LAMMPS_NS

#endif
#endif
