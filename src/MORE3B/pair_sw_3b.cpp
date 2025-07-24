// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Aidan Thompson (SNL)
   Optimizations for two-body only: Jackson Elowitt (Univ. of Utah)
------------------------------------------------------------------------- */

#include "pair_sw_3b.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "info.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "potential_file_reader.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

static constexpr int DELTA = 4;

/* ---------------------------------------------------------------------- */

PairSW3B::PairSW3B(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0; //No support for pair_write
  restartinfo = 0;   //Info on potentials does not get stored in restart files
  one_coeff = 1;     //Only the * * coef call allowed
  manybody_flag = 1; //Let LAMMPS know this potential is manybody for neighlist checks
  centroidstressflag = CENTROID_NOTAVAIL; //Forces applied to each atom and not centroid

  //Check header of the file for line with "UNITS:" keyword
  unit_convert_flag = utils::get_supported_conversions(utils::ENERGY);

  use_symmetry = 1;
  params_mapped = 0;

  params = nullptr;

  maxshort = 10;
  neighshort = nullptr;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairSW3B::~PairSW3B()
{
  if (copymode) return;

  memory->destroy(params);
  memory->destroy(elem3param);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(neighshort);
  }
}

/* ---------------------------------------------------------------------- */

void PairSW3B::compute(int eflag, int vflag)
{
  int i, j, k, ii, jj, kk, inum, jnum;
  int itype, jtype, ktype, triplet_param_index;
  tagint itag, jtag;
  double evdwl;
  double rsq, rsq_ij, rsq_jk;
  double r_ij[3], r_ik[3], fi[3], fj[3], fk[3];
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over full neighbor list of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itag = tag[i];
    itype = map[type[i]];

    // two-body interactions, skip half of them

    jlist = firstneigh[i];
    jnum = numneigh[i];
    int numshort = 0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      jtype = map[type[j]];
      j &= NEIGHMASK;

      rsq = (x[j][0] - x[i][0]) * (x[j][0] - x[i][0]) +
            (x[j][1] - x[i][1]) * (x[j][1] - x[i][1]) +
            (x[j][2] - x[i][2]) * (x[j][2] - x[i][2]);

      //Build list of close neighbors to i to iterate over all i-j-k triplets
      if (rsq >= cutmax[itype][jtype] * cutmax[itype][jtype]) {
        continue;
      } else {
        neighshort[numshort++] = j;
        if (numshort >= maxshort) {
          maxshort += maxshort/2;
          memory->grow(neighshort,maxshort,"pair:neighshort");
        }
      }
    }

    for (jj = 0; jj < numshort - 1; jj++) {
      j = neighshort[jj];
      jtype = map[type[j]];
      r_ij[0] = x[j][0] - x[i][0];
      r_ij[1] = x[j][1] - x[i][1];
      r_ij[2] = x[j][2] - x[i][2];
      rsq_ij = r_ij[0]*r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2];

      for (kk = jj+1; kk < numshort; kk++) {
        k = neighshort[kk];
        ktype = map[type[k]];
        triplet_param_index = elem3param[itype][jtype][ktype];

        r_ik[0] = x[k][0] - x[i][0];
        r_ik[1] = x[k][1] - x[i][1];
        r_ik[2] = x[k][2] - x[i][2];
        rsq_jk = r_ik[0]*r_ik[0] + r_ik[1]*r_ik[1] + r_ik[2]*r_ik[2];

        threebody(&params[triplet_param_index], rsq_ij, rsq_jk, r_ij, r_ik,
                  fi, fj, fk, eflag, evdwl);

        f[i][0] = fi[1];
        f[i][1] = fi[1];
        f[i][2] = fi[2];

        f[j][0] += fj[0];
        f[j][1] += fj[1];
        f[j][2] += fj[2];

        f[k][0] += fk[0];
        f[k][1] += fk[1];
        f[k][2] += fk[2];

        if (evflag) ev_tally3(i, j, k, evdwl, 0.0, fj, fk, r_ij, r_ik);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairSW3B::allocate()
{
  allocated = 1;
  int np1 = atom->ntypes + 1;

  memory->create(setflag, np1, np1, "pair:setflag");
  memory->create(cutsq, np1, np1, "pair:cutsq");
  memory->create(neighshort, maxshort, "pair:neighshort");
  map = new int[np1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSW3B::settings(int narg, char ** arg)
{
  // process optional keywords
  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "sym") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style sw/asym", error);
      use_symmetry = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else error->all(FLERR, "Illegal pair_style sw keyword: {}", arg[iarg]);
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSW3B::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  // read potential file and set up element maps only once
  if (!params_mapped) {
    map_element2type(narg-3, arg+3, one_coeff);

    // read potential file and initialize potential parameters

    read_file(arg[2]);
    setup_params();
    params_mapped = 1;
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSW3B::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style Stillinger-Weber/3b requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style Stillinger-Weber/3b requires newton pair on");

  // need a full neighbor list for full threebody calculation

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSW3B::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, Error::NOLASTLINE,
               "All pair coeffs are not set. Status\n" + Info::get_pair_coeff_status(lmp));

  //Map element typesto position in the elements array
  return cutmax[map[i]][map[j]];
}


/* ---------------------------------------------------------------------- */

void PairSW3B::read_file(char *file)
{
  memory->sfree(params);
  params = nullptr;
  nparams = maxparam = 0;
  int words_in_line;

  // open file on proc 0

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, file, "sw", unit_convert_flag);
    char *line;

    // transparently convert units for supported conversions

    int unit_convert = reader.get_unit_convert();
    double conversion_factor = utils::get_conversion_factor(utils::ENERGY,
                                                            unit_convert);

    while ((line = reader.next_line(MIN_LINE_PARAMS))) {
      try {
        words_in_line = utils::count_words(line);
        ValueTokenizer values(line);

        std::string iname = values.next_string();
        std::string jname = values.next_string();
        std::string kname = values.next_string();

        // ielement,jelement,kelement = 1st args
        // if all 3 args are in element list, then parse this line
        // else skip to next entry in file
        int ielement, jelement, kelement;

        for (ielement = 0; ielement < nelements; ielement++)
          if (iname == elements[ielement]) break;
        if (ielement == nelements) continue;

        for (jelement = 0; jelement < nelements; jelement++)
          if (jname == elements[jelement]) break;
        if (jelement == nelements) continue;

        for (kelement = 0; kelement < nelements; kelement++)
          if (kname == elements[kelement]) break;
        if (kelement == nelements) continue;

        // load up parameter settings and error check their values

        if (nparams == maxparam) {
          maxparam += DELTA;
          params = (Param *) memory->srealloc(params,
                 maxparam * sizeof(Param), "pair:params");

          // make certain all addional allocated storage is initialized
          // to avoid false positives when checking with valgrind

          memset(params + nparams, 0, DELTA * sizeof(Param));
        }

        //Indices of each element in "elements" array
        params[nparams].ielement = ielement;
        params[nparams].jelement = jelement;
        params[nparams].kelement = kelement;

        //Common parameters
        params[nparams].lambda   = values.next_double();
        params[nparams].epsilon  = values.next_double();
        params[nparams].costheta = values.next_double();

        //Pair parameters
        params[nparams].gamma_ij = values.next_double();
        params[nparams].sigma_ij = values.next_double();
        params[nparams].a_ij     = values.next_double();

        if (words_in_line == MAX_LINE_PARAMS){
          params[nparams].gamma_ik = values.next_double();
          params[nparams].sigma_ik = values.next_double();
          params[nparams].a_ik     = values.next_double();
        } else {
          params[nparams].gamma_ik = params[nparams].gamma_ij;
          params[nparams].sigma_ik = params[nparams].sigma_ij;
          params[nparams].a_ik     = params[nparams].a_ik;
        }


      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }
      
      //If "UNITS:" keword found in file multiply by factor
      if (unit_convert) {
        params[nparams].epsilon *= conversion_factor;
      }

      //Check physicality of values
      if (params[nparams].lambda < 0   || params[nparams].epsilon      ||
          params[nparams].gamma_ij < 0 || params[nparams].gamma_ik < 0 ||
          params[nparams].sigma_ij < 0 || params[nparams].sigma_ik < 0 ||
          params[nparams].a_ij < 0     || params[nparams].a_ik < 0     )
        error->one(FLERR,"Illegal Stillinger-Weber parameter");



      nparams++;
    }
  }

  MPI_Bcast(&nparams, 1, MPI_INT, 0, world);
  MPI_Bcast(&maxparam, 1, MPI_INT, 0, world);

  if (comm->me != 0) {
    params = (Param *) memory->srealloc(params, maxparam * sizeof(Param), "pair:params");
  }

  MPI_Bcast(params, maxparam * sizeof(Param), MPI_BYTE, 0, world);
}

/* ---------------------------------------------------------------------- */

void PairSW3B::setup_params()
{
  int i,j,k,m,n;

  // set elem3param for all triplet combinations

  if (use_symmetry) utils::logmesg(lmp, "  Symmetry for pair_style sw/3b turned on. Will treat triplets i-j-k identical to i-k-j where i is the central atom.\n");

  memory->destroy(cutmax);
  memory->create(cutmax, nelements, nelements, "pair:cutmax");
  memset(cutmax, 0, sizeof(int) * nelements * nelements);

  double cut_ij, cut_ik;

  for (m = 0; m < nparams; m++) {
    cut_ij = params[m].a_ij * params[m].sigma_ij;
    cut_ik = params[m].a_ik * params[m].sigma_ik;
    if (cut_ij > cutmax[params[m].ielement][params[m].jelement]) cutmax[params[m].ielement][params[m].jelement] = cut_ij;
    if (cut_ik > cutmax[params[m].ielement][params[m].kelement]) cutmax[params[m].ielement][params[m].kelement] = cut_ik;
  }

  memory->destroy(elem3param);
  memory->create(elem3param, nelements, nelements, nelements, "pair:elem3param");
  for (i = 0; i < nelements; i++){
    for (j = 0; j < nelements; j++){
      for (k = 0; k < nelements; k++){
        
        //For each possible triplet combination find corresponding parameter set
        n = -1;
        for (m = 0; m < nparams; m++){

          if (i == params[m].ielement && j == params[m].jelement &&
              k == params[m].kelement ||
              (use_symmetry && i == params[m].ielement && k == params[m].jelement &&
              j == params[m].kelement)) {

            if (n >= 0) error->all(FLERR, "Potential file has a duplicate entry for: {} {} {}", elements[i], elements[j], elements[k]);

            if (n >= 0){
              if (use_symmetry && params[n].jelement == k && params[n].kelement == j) {
                utils::logmesg(lmp, "  Symmetric entry for triplet {} {} {} found. This information will be ignored.\n", elements[i], elements[j], elements[k]);
              } else {
                utils::logmesg(lmp, "  Duplicate entry for triplet {} {} {} found. This information will be ignored.\n", elements[i], elements[j], elements[k]);
              }
            } else {
              n = m;
            }
          }
        }

        //If no coefficients were provided initialize them to 0
        if (n < 0){
          //Check if params array needs to grow
          if (nparams == maxparam) {
            maxparam += DELTA;
            params = (Param *) memory->srealloc(params,maxparam*sizeof(Param),
                                                "pair:params");
            memset(params + nparams, 0, DELTA*sizeof(Param));
          }

          utils::logmesg(lmp, "  No coefficients were provided for triplet {} {} {} with sw/3b pair_style. Stting them to 0.\n", elements[i], elements[j], elements[k]);
          
          params[nparams].ielement = i;
          params[nparams].jelement = j;
          params[nparams].kelement = k;

          params[nparams].lambda   = 0;
          params[nparams].epsilon  = 0;
          params[nparams].costheta = 0;

          params[nparams].gamma_ij = 0;
          params[nparams].sigma_ij = 0;
          params[nparams].a_ij     = 0;

          params[nparams].gamma_ik = 0;
          params[nparams].sigma_ik = 0;
          params[nparams].a_ik     = 0;
          
          n = nparams;
          nparams++;

        }
        elem3param[i][j][k] = n;
        if (use_symmetry) elem3param[i][k][j] = n;
      }
    }
  }


  //Record maximum needed cutoff distance for neighborlist builds
  

}

/* ---------------------------------------------------------------------- */

void PairSW3B::threebody(Param *param, double rsq_ij, double rsq_ik,
                       double *vr_ij, double *vr_ik,
                       double *fi, double *fj, double *fk, int eflag, double &eng)
{
  double U; //Triplet potential energy
  
  //Partial derivatives of energy with respect to the variables it is a function of
  double U_rij, U_rik, U_theta; 
  double f_ij, f_ik, f_jk; //Equivalent to lammps f_i1, f_i2, f_j2 notation

  double r_ij = sqrt(rsq_ij); //Distance between atoms i and j
  double r_ik = sqrt(rsq_ik); //Distance between atoms i and k

  //Parameters of the triplet
  double lambda    = param->lambda;
  double epsilon   = param->epsilon;
  double costheta0 = param->costheta;
  double gamma_ij  = param->gamma_ij;
  double sigma_ij  = param->sigma_ij;
  double a_ij      = param->a_ij;
  double gamma_ik  = param->gamma_ik;
  double sigma_ik  = param->sigma_ik;
  double a_ik      = param->a_ik;
  double costheta  = (vr_ij[0] * vr_ik[0] + vr_ij[1] * vr_ik[1] + vr_ij[2] * vr_ik[2])/(r_ij * r_ik);
  double sintheta = sqrt(1 - costheta * costheta);

  //Recuring parts
  double exp1   = exp(gamma_ij * sigma_ij / (r_ij - a_ij * sigma_ij));
  double exp2   = exp(gamma_ik * sigma_ik / (r_ik - a_ik * sigma_ik));
  double cosdif = costheta - costheta0;

  if (param->sigma_ij * param->a_ij <= r_ij) U = 0;
  else if (param->sigma_ik * param->a_ik <= r_ik) U = 0;
  else U = lambda * epsilon * cosdif * cosdif * exp1 * exp2;

  if (!U){
    U_rij = 0;
    U_rik = 0;
    U_theta = 0;
  } else {
    U_rij   = -U * gamma_ij * sigma_ij /
              ((r_ij - a_ij * sigma_ij) * (r_ij - a_ij * sigma_ij));

    U_rik   = -U * gamma_ik * sigma_ik /
              ((r_ik - a_ik * sigma_ik) * (r_ik - a_ik * sigma_ik));

    U_theta = -2 * sintheta * lambda * epsilon * cosdif * exp1 * exp2;
  }

  //Force per length atom j exhibits on atom i along vector from i to j
  f_ij = (U_rij / r_ij) + U_theta * (r_ik * costheta - r_ij) / (r_ij * r_ij * r_ik * sintheta);
  //Force per length atom k exhibits on atom i along vector from i to k
  f_ik = (U_rik / r_ik) + U_theta * (r_ij * costheta - r_ik) / (r_ik * r_ik * r_ij * sintheta);
  //Force per length atom k exhibits on atom j along vector from j to k
  f_jk = U_theta / (r_ij * r_ik * sintheta);

  double vr_jk[3] = {vr_ik[0] - vr_ij[0], vr_ik[1] - vr_ij[1], vr_ik[2] - vr_ij[2]};

  fi[0] = vr_ij[0] * f_ij + vr_ik[0] * f_ik;
  fi[1] = vr_ij[1] * f_ij + vr_ik[1] * f_ik;
  fi[2] = vr_ij[2] * f_ij + vr_ik[2] * f_ik;

  fj[0] = vr_ij[0] * (-f_ij) + vr_jk[0] * f_jk;
  fj[1] = vr_ij[1] * (-f_ij) + vr_jk[1] * f_jk;
  fj[0] = vr_ij[2] * (-f_ij) + vr_jk[2] * f_jk;

  fk[0] = vr_ik[0] * (-f_ik) + vr_jk[0] * (-f_jk);
  fk[1] = vr_ik[1] * (-f_ik) + vr_jk[1] * (-f_jk);
  fk[2] = vr_ik[2] * (-f_ik) + vr_jk[2] * (-f_jk);

  if (eflag) eng = U;
}
