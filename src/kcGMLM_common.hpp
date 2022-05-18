#ifndef GMLM_GMLM_Common_H
#define GMLM_GMLM_Common_H
#include "kcShared.hpp"

//common datastuctures for the population and individual recording GMLMs

namespace kCUDA { 
/* CLASSES GPUGMLM_params, GPUGMLM_group_params
 *  The argument structures for the GMLM parameters
 *
 *  GPUGMLM_params
 *      W [dim_P x 1] (FPTYPE): baseline rate terms
 *      Groups [dim_J] (GPUGMLM_group_params): each group's parameters
 *  
 *  GPUGMLM_group_params
 *      dim_R (scalar) [size_t]: rank of the tensor group
 *      dim_T [dim_S x 1]: sizes of each component
 *
 *      V [dim_P x dim_R] (FPTYPE): the neuron weighting components
 *      T [dim_S x 1]
 *          T[ss] [dim_T[ss] x dim_R]: the components for dim ss
 */
template <class FPTYPE>
class GPUGMLM_group_params {
public:
    std::vector<GLData<FPTYPE> *> T; //length dim_S: each T[ss] is dim_T[ss] x dim_R
    GLData<FPTYPE> * V; // dim_P x dim_R
    
    virtual ~GPUGMLM_group_params() {};
    
    inline size_t dim_R(std::shared_ptr<GPUGL_msg> msg) const {
        size_t dim_R_c = V->getSize(1);
        for(int ss = 0; ss < T.size(); ss++) {
            if(T[ss]->getSize(1) != dim_R_c) {
                std::ostringstream output_stream;
                output_stream << "GPUGMLM_group_params errors: inconsistent ranks!";
                msg->callErrMsgTxt(output_stream);
            }
        }
        return dim_R_c;
    }
    inline int dim_R() const {
        int dim_R_c = V->getSize(1);
        for(int ss = 0; ss < T.size(); ss++) {
            if(T[ss]->getSize(1) != dim_R_c) {
                return -1;
            }
        }
        return dim_R_c;
    }
    inline size_t dim_S() const {
        return T.size();
    }
    inline size_t dim_T(int dim, std::shared_ptr<GPUGL_msg> msg) const  {
        if(dim < 0 || dim > dim_S() || T[dim] == NULL) {
            std::ostringstream output_stream;
            output_stream << "GPUGMLM_group_params errors: invalid dim!";
            msg->callErrMsgTxt(output_stream);
        }
        return T[dim]->getSize(0);
    }
    inline size_t dim_P() const {
        return V->getSize(0);
    }
};

template <class FPTYPE>
class GPUGMLM_params {
public:
    GLData<FPTYPE> * W = NULL; // length dim_P
    GLData<FPTYPE> * B = NULL; // dim_B x dim_P (lead dimension is ld_B)
    
    std::vector<GPUGMLM_group_params<FPTYPE> *> Groups; //length dim_J
    
    virtual ~GPUGMLM_params() {};
    
    inline size_t dim_P(std::shared_ptr<GPUGL_msg> msg) const {
        if(W == NULL) {
            return 0;
        }

        if(W->size() != B->getSize(1) && B->size() > 0) {
            std::ostringstream output_stream;
            output_stream << "GPUGMLM_params errors: inconsistent dim_P between W and B!";
            msg->callErrMsgTxt(output_stream);
        }
        return W->size();
    }
    inline size_t dim_P() const {
        if(W == NULL) {
            return 0;
        }
        if(W->size() != B->getSize(1) && B->size() > 0) {
            return 0;
        }
        return W->size();
    }
    inline size_t dim_B() const {
        if(B == NULL) {
            return 0;
        }
        return B->getSize(0);
    }
    inline unsigned int dim_J() const {
        return Groups.size();
    }
};


/* CLASSES GPUGMLM_computeOptions, GPUGMLM_group_computeOptions
 *  The option structures for what to compute.
 *  Note that if the derivative computation is false, the corresponding second derivative will never be computed.
 *
 *  GPUGMLM_params
 *      compute_dW [scalar]: options to compute gradient of W
 *      compute_trialLL [scalar]: option to compute the trial-specific log likelihoods
 *      Groups [dim_J] (GPUGMLM_group_computeOptions): each group's options
 *  
 *  GPUGMLM_group_computeOptions
 *      compute_dV [scalar]: options to compute gradient
 *      compute_dT [dim_S x 1]:  options to compute gradient of each T
 */
class GPUGMLM_group_computeOptions {
public:
    bool compute_dV;
    std::vector<bool> compute_dT;  //length dim_S
    
    virtual ~GPUGMLM_group_computeOptions() {};
};

template <class FPTYPE>
class GPUGMLM_computeOptions {
public:
    bool compute_dB;
    bool compute_dW;
    bool compute_trialLL; // if you don't want to copy all the individual trial likelihoods back to host
    
    bool update_weights = true;
    GLData<FPTYPE> * trial_weights; // EMPTY or (max_trials x 1) or (max_trials x dim_P). Weighting for each trial (if 0, doesn't compute trial at all). Can be used for SGD
                                    // if EMPTY: all trial weights 1
                                    // if (max_trials x 1): weights for each trial
                                    // if (max_trials x dim_P): weights for each neuron in each trial (POPULATION MODEL ONLY)
    
    std::vector<GPUGMLM_group_computeOptions *> Groups; //length dim_J
    
    virtual ~GPUGMLM_computeOptions() {};
};

/* CLASSES GPUGMLM_results, GPUGMLM_group_results
 *  The results structures for the GMLM (not similarity to the arg struct).
 *  WARNING: if any field is not specified by the 'opts' argument to the log likelihood call,
 *           the results may not be valid!
 *
 *  GPUGMLM_params
 *      trialLL [dim_M x 1] (FPTYPE): trial-wise log likelihood
 *      dW      [dim_P x 1] (FPTYPE): baseline rate terms
 *      dB      [dim_B x dim_P] (FPTYPE): linear terms (neuron specific)
 *      Groups  [dim_J] (GPUGMLM_group_params): each group's parameters
 *  
 *  GPUGMLM_group_params
 *      dim_R (scalar) [size_t]: rank of the tensor group
 *      dim_T [dim_S x 1]: sizes of each component
 *
 *      dV  [dim_P       x dim_R] (FPTYPE): the neuron weighting components
 *      dT [dim_S x 1]
 *          dT[ss]  [dim_T[ss] x dim_R]: the components for dim ss
 */
template <class FPTYPE>
class GPUGMLM_group_results {
public:
    std::vector<GLData<FPTYPE> *>  dT;  //length dim_S: each T[ss] is dim_T[ss] x dim_R 
    GLData<FPTYPE>  *  dV = NULL; //size dim_P x dim_R
    
    virtual ~GPUGMLM_group_results() {};
    
    inline size_t dim_R(std::shared_ptr<GPUGL_msg> msg) const {
        size_t dim_R = 0;
        if(dV != NULL) {
            dim_R = dV->getSize(1);
        }
        else {
            std::ostringstream output_stream;
            output_stream << "GPUGMLM_group_results errors: inconsistent ranks!";
            msg->callErrMsgTxt(output_stream);
        }
        for(int ss = 0; ss < dT.size(); ss++) {
            if( dT[ss] == NULL  || dT[ss]->getSize(1) != dim_R) {
                std::ostringstream output_stream;
                output_stream << "GPUGMLM_group_results errors: inconsistent ranks!";
                msg->callErrMsgTxt(output_stream);
            }
        }
        return dim_R;
    }
    inline size_t dim_S() const {
        return dT.size();
    }
    inline size_t dim_T(int dim, std::shared_ptr<GPUGL_msg> msg) const {
        if(dim < 0 || dim > dim_S()) {
            std::ostringstream output_stream;
            output_stream << "GPUGMLM_group_results errors: invalid dim!";
            msg->callErrMsgTxt(output_stream);
        }
        if(dT[dim] != NULL) {
            return dT[dim]->getSize(0);
        }
        else {
            return 0;
        }
    }
    inline size_t dim_P() const {
        if(dV != NULL) {
            return dV->getSize(0);
        }
        else {
            return 0;
        }
    }
};


template <class FPTYPE>
class GPUGMLM_results {
public:
    bool isSimultaneousPopulation = true;

    GLData<FPTYPE> * dW = NULL; //length dim_P
    GLData<FPTYPE> * dB = NULL; //size dim_P x dim_K (lead dim is ld_K)
    
    GLData<FPTYPE> * trialLL = NULL; //length dim_M        if !isSimultaneousPopulation
                              // size dim_M x dim_P if  isSimultaneousPopulation
    
    std::vector<GPUGMLM_group_results<FPTYPE> *> Groups; //length dim_J
    virtual ~GPUGMLM_results() {};
    
    inline size_t dim_P(std::shared_ptr<GPUGL_msg> msg) const  {
        size_t dim_P_c = 0;
        if(isSimultaneousPopulation && trialLL != NULL && !trialLL->empty()) {
            dim_P_c = trialLL->getSize(1);
        }
        else if(dW != NULL && !dW->empty()) {
            dim_P_c = dW->getNumElements();
        }
        else if(dB != NULL && !dB->empty()) {
            dim_P_c = dB->getSize(1);
        }
        
        if(dW != NULL && dW->getNumElements() != dim_P_c && !dW->empty()) {
            std::ostringstream output_stream;
            output_stream << "GPUGMLMPop_results errors: inconsistent dim_P for dW!";
            msg->callErrMsgTxt(output_stream);
        }
        if(isSimultaneousPopulation && trialLL != NULL && trialLL->getSize(1) != dim_P_c && !trialLL->empty()) {
            std::ostringstream output_stream;
            output_stream << "GPUGMLMPop_results errors: inconsistent dim_P for trialLL!";
            msg->callErrMsgTxt(output_stream);
        }
        if(dB->getSize(1) != dim_P_c && dB != NULL && !dB->empty()) {
            std::ostringstream output_stream;
            output_stream << "GPUGMLMPop_results errors: inconsistent dim_P for dB!";
            msg->callErrMsgTxt(output_stream);
        }
        return dim_P_c;
    }
    inline size_t dim_B() const {
        return dB->getSize(0);
    }
    inline size_t dim_M() const {
        if(isSimultaneousPopulation) {
            return trialLL->getSize(0);
        }
        else {
            return trialLL->getNumElements();
        }
    }
};



/* CLASSES GPUGMLM_structure_args, GPUGMLM_structure_Group_args, GPUGMLM_GPU_block_args
 *  complete specification of inputs to create a GMLM: contains all dimension information and regressors
 *  
 *  Trials are placed into a vector of GPUGMLM_GPU_block_args for each GPU.
 *  A single GPUGMLM_structure_args is created which described how each trial should look.
 *
 *  Callers to this GMLM should construct one of these can make sure the dimensions and entries make sense
 *  GPUGMLM should do a deep copy of everything (all the big elements are copied to the GPUs).
 *
 *  Note: we require some exposed pointers just to make sure that everything plays nicely with CUDA.
 *        If the pointers and dimensions aren't correct, this can lead to seg faults
 */
template <class FPTYPE>
class GPUGMLM_trial_Group_args {
    public:
        //all vectors are length dim_S
        std::vector<GLData<FPTYPE> *> X; //each X is size dim_N x dim_F[dd] x (dim_A or 1) (or empty if the universal regressors are used)
        
        std::vector<GLData<int> *> iX; // for each non-empty X, iX is a dim_N x dim_A matrix of indices into X_shared (the universal regressors)

        virtual ~GPUGMLM_trial_Group_args() {};

        size_t dim_N(std::shared_ptr<GPUGL_msg> msg) const {
            if(X.size() != iX.size()) {
                std::ostringstream output_stream;
                output_stream << "GPUGMLM_trial_Group_args errors: inconsistent sizes of X and iX!";
                msg->callErrMsgTxt(output_stream);
            }

            size_t dim_N_c = 0;
            for(int ii = 0; ii < X.size(); ii++) {
                if(X[ii] != NULL && !(X[ii]->empty()) && iX[ii] != NULL && !(iX[ii]->empty())) {
                    std::ostringstream output_stream;
                    output_stream << "GPUGMLM_trial_Group_args errors: mode " << ii << " has both local and shared regressors.";
                    msg->callErrMsgTxt(output_stream);
                }

                size_t dim_N_ci = 0;
                if(X[ii] != NULL && !(X[ii]->empty())) {
                    dim_N_ci = X[ii]->getSize(0);
                }
                else if(iX[ii] != NULL && !(iX[ii]->empty())) {
                    dim_N_ci = iX[ii]->getSize(0);
                }
                else {
                    std::ostringstream output_stream;
                    output_stream << "GPUGMLM_trial_Group_args errors: mode " << ii << " has no regressors.";
                    msg->callErrMsgTxt(output_stream);
                }
                if(ii == 0) {
                    dim_N_c = dim_N_ci;
                }

                if(dim_N_c != dim_N_ci) {
                    std::ostringstream output_stream;
                    output_stream << "GPUGMLM_trial_Group_args errors: inconsistent dim_N!";
                    msg->callErrMsgTxt(output_stream);
                }
            }
            return dim_N_c;

        }
        size_t dim_N() const {
            if(X.size() != iX.size()) {
                return 0;
            }

            size_t dim_N_c = 0;
            for(int ii = 0; ii < X.size(); ii++) {
                if(X[ii] != NULL && !(X[ii]->empty()) && iX[ii] != NULL && !(iX[ii]->empty())) {
                    return 0;
                }

                size_t dim_N_ci = 0;
                if(X[ii] != NULL && !(X[ii]->empty())) {
                    dim_N_ci = X[ii]->getSize(0);
                }
                else if(iX[ii] != NULL && !(iX[ii]->empty())) {
                    dim_N_ci = iX[ii]->getSize(0);
                }
                else {
                    return 0;
                }
                if(ii == 0) {
                    dim_N_c = dim_N_ci;
                }

                if(dim_N_c != dim_N_ci) {
                    return 0;
                }
            }
            return dim_N_c;
        }
        inline size_t dim_D() const {
            if(X.size() != iX.size()) {
                return 0;
            }
            else {
                return X.size();
            }
        }
        inline size_t dim_F(unsigned int mode) const {
            if(mode < dim_D()) {
                if(X[mode] != NULL && !(X[mode]->empty()) && iX[mode] != NULL && !(iX[mode]->empty())) {
                    return 0; // bad setup: both local and global regressors
                }

                if(X[mode] != NULL && !(X[mode]->empty())) {
                    return X[mode]->getSize(1);
                }
                else {
                    return 0;
                }
            }
            else {
                return 0;
            }
        }
        inline int isFactorShared(unsigned int mode) const {
            if(mode < X.size() && mode < iX.size()) {
                if(X[mode] != NULL && !(X[mode]->empty()) && iX[mode] != NULL && !(iX[mode]->empty())) {
                    return -1; // bad setup: both local and global regressors
                }

                if(X[mode] != NULL && !(X[mode]->empty())) {
                    return 0;
                }
                else {
                    return 1;
                }
            }
            return -1;
        }
        inline bool validDimA(unsigned int dim_A_check) const {
            for(int mode = 0; mode < dim_D(); mode++) {
                if(X[mode] != NULL && !(X[mode]->empty()) && iX[mode] != NULL && !(iX[mode]->empty())) {
                    return false; // bad setup: both local and global regressors
                }

                if(X[mode] != NULL && !(X[mode]->empty())) {
                    if(X[mode]->getSize(2) > 1 && X[mode]->getSize(2) != dim_A_check) {
                        return false;
                    }
                }

                else if(iX[mode] != NULL && !(iX[mode]->empty())) {
                    if(iX[mode]->getSize(1) != dim_A_check) {
                        return false;
                    }
                }
            }
            return true;
        }
};

template <class FPTYPE>
class GPUGMLM_trial_args {
    public:
        std::vector<GPUGMLM_trial_Group_args<FPTYPE> *> Groups; // the regressor tensor groups

        unsigned int neuron;     // which neuron the trial belongs to (if !isSimultaneousPopulation)

        unsigned int trial_idx;  // a trial number index
        
        GLData<FPTYPE> * Y = NULL; // length dim_N, the spike counts per bin if(!isSimultaneousPopulation)
                            // otherwise, dim_N x dim_P
        
        GLData<FPTYPE> * X_lin = NULL; // size dim_N x dim_B, the linear term of the neuron

        virtual ~GPUGMLM_trial_args() {};
        
        size_t dim_N(std::shared_ptr<GPUGL_msg> msg) const {
            size_t dim_N_c = 0;
            if(Y != NULL) {
                dim_N_c = Y->getSize(0);
            }
            else if(X_lin != NULL && !(X_lin->empty())) {
                dim_N_c = X_lin->getSize(0);
            }
            if(X_lin != NULL && !(X_lin->empty()) && X_lin->getSize(0) != dim_N_c) {
                std::ostringstream output_stream;
                output_stream << "GPUGMLM_trial_args errors: inconsistent dim_N!";
                msg->callErrMsgTxt(output_stream);
            }
            return dim_N_c;
        }
        size_t dim_N() const {
            size_t dim_N_c = 0;
            if(Y != NULL) {
                dim_N_c = Y->getSize(0);
            }
            else if(X_lin != NULL && !(X_lin->empty())) {
                dim_N_c = X_lin->getSize(0);
            }
            if(X_lin != NULL && !(X_lin->empty()) && X_lin->getSize(0) != dim_N_c) {
                dim_N_c = 0;
            }
            for(int jj = 0; jj < Groups.size(); jj++) {
                if(Groups[jj] == NULL || Groups[jj]->dim_N() != dim_N_c) {
                    return 0;
                }
            }
            return dim_N_c;
        }
        size_t dim_B() const {
            if(X_lin != NULL) {
                return X_lin->getSize(1);
            }
            else {
                return 0;
            }
        }
        inline size_t dim_P(std::shared_ptr<GPUGL_msg> msg) const {
            size_t dim_P_c = 0;
            if(Y != NULL) {
                dim_P_c = Y->getSize(1);
            }
            else if(X_lin != NULL && X_lin->empty()) {
                dim_P_c = X_lin->getSize(2);
            }
            if(X_lin != NULL && X_lin->getSize(2) != 1 && X_lin->getSize(2) != dim_P_c) {
                std::ostringstream output_stream;
                output_stream << "GPUGMLM_trial_args errors: inconsistent dim_P!";
                msg->callErrMsgTxt(output_stream);
            }
            return dim_P_c;
        }
        inline size_t dim_P() const {
            size_t dim_P_c = 0;
            if(Y != NULL) {
                dim_P_c = Y->getSize(1);
            }
            else if(X_lin != NULL && X_lin->empty()) {
                dim_P_c = X_lin->getSize(2);
            }
            if(X_lin != NULL && X_lin->getSize(2) != 1 && X_lin->getSize(2) != dim_P_c) {
                return 0;
            }
            return dim_P_c;
        }
};


template <class FPTYPE>
class GPUGMLM_GPU_block_args {
    public:
        int dev_num;            //which GPU to use
        int max_trials_for_sparse_run; //max trials that can be run in a minibatch (for pre-allocating space. Can always run the full set)

        std::vector<GPUGMLM_trial_args <FPTYPE> *> trials; // the trials for this block

        virtual ~GPUGMLM_GPU_block_args() {};
};


template <class FPTYPE>
class GPUGMLM_structure_Group_args {
    protected:
    
        inline size_t dim_D() const {
            size_t max_factor = X_shared.size();
            for(int dd = 0; dd < max_factor; dd++) {
                bool found = false;
                for(int ss = 0; ss < factor_idx.size() && !found; ss++) {
                    if(factor_idx[ss] == dd) {
                        found = true;
                    }
                    else if(factor_idx[ss] >= max_factor) {
                        return 0;
                    }
                }
                if(!found) {
                    return 0;
                }
            }
            
            return max_factor;
        }
    public:
        //all vectors are length dim_S
        std::vector<size_t  > dim_T;             // outer tensor dimension for each component
        std::vector<GLData<FPTYPE> *> X_shared; // dim_D, any univerisal regressors (size dim_X_shared[dd] x dim_T[ss]): empty if no shared indices used

        std::vector<unsigned int> factor_idx; // size of dim_T, factors for each dim (how the coeffs are divided into partial CP decompositions)
                                     //(MUST BE 0 to max factor - 1, where max factor gives dim_D!)
        
        size_t dim_A;                 // number of events for this tensor Group
        size_t dim_R_max;             // max rank of group (for pre-allocation)

        virtual ~GPUGMLM_structure_Group_args() {};
        
        inline size_t dim_S() const {
            return dim_T.size();
        }
        inline size_t dim_D(std::shared_ptr<GPUGL_msg> msg) const {
            size_t max_factor = X_shared.size();
            if(dim_T.size() != factor_idx.size()) {
                std::ostringstream output_stream;
                output_stream << "GPUGMLM_structure_Group_args errors: inconsistent dim_S!";
                msg->callErrMsgTxt(output_stream);
            }
            //search for each factor
            for(int dd = 0; dd < max_factor; dd++) {
                bool found = false;
                for(int ss = 0; ss < factor_idx.size() && !found; ss++) {
                    if(factor_idx[ss] == dd) {
                        found = true;
                    }
                    else if(factor_idx[ss] >= max_factor) {
                        std::ostringstream output_stream;
                        output_stream << "GPUGMLM_structure_Group_args errors: invalid CP decomp factorization!";
                        msg->callErrMsgTxt(output_stream);
                    }
                }
                if(!found) {
                    std::ostringstream output_stream;
                    output_stream << "GPUGMLM_structure_Group_args errors: invalid CP decomp factorization!";
                    msg->callErrMsgTxt(output_stream);
                }
            }
            
            return max_factor;
        }
        inline size_t dim_F(unsigned int factor, std::shared_ptr<GPUGL_msg> msg) const {
            size_t ff = 0;
            if(factor < dim_D(msg)) {
                ff = 1;
                for(int ss = 0; ss < factor_idx.size(); ss++) {
                    if(factor_idx[ss] == factor) {
                        ff *= dim_T[ss];
                    }
                }
            }
            else {
                std::ostringstream output_stream;
                output_stream << "GPUGMLM_structure_Group_args errors: invalid factor number!";
                msg->callErrMsgTxt(output_stream);
            }
            return ff;
        }
        inline size_t dim_F(unsigned int factor) const {
            size_t ff = 0;
            if(factor < dim_D()) {
                ff = 1;
                for(int ss = 0; ss < factor_idx.size(); ss++) {
                    if(factor_idx[ss] == factor) {
                        ff *= dim_T[ss];
                    }
                }
            }
            else {
                return 0;
            }
            return ff;
        }
        
        //validates some of the basic GMLM structure of the regressors for a trial
        int validateTrialStructure(const GPUGMLM_trial_Group_args <FPTYPE> * trial, size_t dim_N_target) const {
            
            //checks if local vectors are all correct length
            if(dim_D() == 0) {
                return -1;
            }
            
            //checks if trial vectors are all correct length
            if(trial->dim_D() != dim_D()) {
                return -2;
            }
            if(trial->dim_N() != dim_N_target || dim_N_target == 0) {
                return -3;
            }
            if(!(trial->validDimA(dim_A))) {
                return -4;
            }
            
            //checks if universal regressor indices match
            for(unsigned int ff = 0; ff < dim_D(); ff++) {
                size_t dim_F_c = trial->dim_F(ff);
                int isShared = trial->isFactorShared(ff);
                if(isShared == 1 && (X_shared[ff] == NULL || X_shared[ff]->empty())){
                    return -10 - static_cast<int>(ff); 
                }
                else if(isShared< 1 && (X_shared[ff] != NULL && !(X_shared[ff]->empty()))) {
                    return -20 - static_cast<int>(ff); 
                }
                else if(!isShared && (trial->dim_F(ff) != dim_F(ff) || dim_F_c == 0)) {
                    return -30 - static_cast<int>(ff); //expected a trial-wise set of regressors, not iX
                }
            }
            
            return 1; //passed basic tests (seg faults may still exist if the pointers are bad!)
        }
};

template <class FPTYPE>
class GPUGMLM_structure_args {
    public:
        std::vector<GPUGMLM_structure_Group_args<FPTYPE> *> Groups; // the coefficient tensor groups, length dim_J

        bool isSimultaneousPopulation = true;

        size_t dim_B; // size of the linear term in the model
        size_t dim_P; // number of neurons to expect 
        
        logLikeType logLikeSettings = ll_poissExp;
        std::vector<FPTYPE> logLikeParams;

        FPTYPE binSize;    // bin width (in seconds) if you want the baseline rate parameters to be in terms of sp/s (totally redundant with the exp nonlinear, but fine)

        virtual ~GPUGMLM_structure_args() {};
    
        //validates some of the basic GMLM structure of the regressors for a trial
        // but not thoroughly here
        int validateTrialStructure(const GPUGMLM_trial_args <FPTYPE> * trial) const { 
            
            if(trial->Groups.size() != Groups.size()) {
                return -100000; //invalid number of tensor groups
            }
            if(trial->dim_B() != dim_B) {
                return -100001;
            }
            if(isSimultaneousPopulation && trial->dim_P() != dim_P) {
                return -100002;
            }
            if(!isSimultaneousPopulation && trial->dim_P() != 1) {
                return -100003;
            }
            
            for(int jj = 0; jj < Groups.size(); jj++) {
                int vd = Groups[jj]->validateTrialStructure(trial->Groups[jj], trial->dim_N());
                if(vd != 1) {
                    return vd - jj*100; // group structure was invalid 
                }
            }
            
            return 1; //passed basic tests (seg faults may still exist if the pointers are bad!)
        }
        int validateTrialStructure(const GPUGMLM_GPU_block_args <FPTYPE> * block) const {
            for(int mm = 0; mm < block->trials.size(); mm++) {
                //checks each trial on a block
                int vd = validateTrialStructure(block->trials[mm]);
                if(vd != 1) {
                    return vd;
                }
            }
            return 1;
        }                           
        int validateTrialStructure(const std::vector<GPUGMLM_GPU_block_args <FPTYPE> *> blocks) const {
            for(int bb = 0; bb < blocks.size(); bb++) {
                //checks each block
                int vd = validateTrialStructure(blocks[bb]);
                if(vd != 1) {
                    return vd;
                }
            }
            return 1;
        }
        unsigned int dim_J() {
            return Groups.size();
        }
};


}; //namespace
#endif