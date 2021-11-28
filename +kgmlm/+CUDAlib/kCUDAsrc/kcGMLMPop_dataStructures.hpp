/*
 * kcGMLMPop_dataStructures.hpp
 * Holds all the data - parameters, results, regressors, computation space
 * for a GMLMPop (on one GPU).
 *
 * Package GMLM_dmc for dimensionality reduction of neural data.
 *   Population GMLM and indidivual-cell-recordings GMLM are in separate CUDA files.
 *   I decided to do this in order to make the different optimization requirements a little more clean.
 *   
 *  References
 *   Kenneth Latimer & David Freeedman (2021). Low-dimensional encoding of 
 *   decisions in parietal cortex reflects long-term training history.
 *   bioRxiv
 *
 *  Copyright (c) 2021 Kenneth Latimer
 *
 *   This software is distributed under the GNU General Public
 *   License (version 3 or later); please refer to the file
 *   License.txt, included with the software, for details.
 */
#ifndef GMLMPop_GMLMPop_DATASTRUCTURES_H
#define GMLMPop_GMLMPop_DATASTRUCTURES_H

#include "kcBase.hpp"
#include "kcGMLMPop.hpp"
namespace kCUDA {
    
/*The main datastructures for the GMLMPop on the GPU
 *      parameters
 *      results
 *      dataset (all data + pre-allocated compute space)
 *    
 *      I make overly generous use of friend classes to streamline some computation code.
 */
    
template <class FPTYPE> class GPUGMLMPop_parameters_Group_GPU;
template <class FPTYPE> class GPUGMLMPop_parameters_GPU;
template <class FPTYPE> class GPUGMLMPop_results_Group_GPU;
template <class FPTYPE> class GPUGMLMPop_results_GPU;
template <class FPTYPE> class GPUGMLMPop_dataset_Group_GPU;
template <class FPTYPE> class GPUGMLMPop_dataset_GPU;

/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
 * PARAMETERS on GPU 
 */

template <class FPTYPE>
class GPUGMLMPop_parameters_GPU : GPUGL_base {
private:
    friend class GPUGMLMPop_parameters_Group_GPU<FPTYPE>; //to give groups access to the main vars
    friend class GPUGMLMPop_computeBlock<FPTYPE>; 
    friend class GPUGMLMPop_results_GPU<FPTYPE>;
    
    //each neuron's baseline rate
    GPUData<FPTYPE> * W; //vectory size dim_P
    //each neuron's linear terms
    GPUData<FPTYPE> * B;
    
    GPUData<FPTYPE> * trial_weights;                  // can be NULL or trial_weights_temp (NULL means all weights == 1)
    GPUData<FPTYPE> * trial_weights_0;                // is (dim_M x 0) 
    GPUData<FPTYPE> * trial_weights_temp;             // vector size dim_M x dim_P
    
    GPUData<unsigned int> * trial_included;           // can be NULL or trial_included_temp (NULL means not doing a sparse run)
    GPUData<unsigned int> * trial_included_temp;      // vector size trial_weights_nonzero_cnt (0 to max_trials)
                                                      // list of which trials are currently being computed (non-zero weights) - page-locked host + GPU memory 
    GPUData<unsigned int> * trial_included_0;         // is (dim_M x 0) 
    
    logLikeType logLikeSettings;
    GPUData<FPTYPE> * logLikeParams;

    cudaEvent_t	paramsLoaded_event; 	
    
    //each group
    std::vector<GPUGMLMPop_parameters_Group_GPU<FPTYPE> *> Groups;
    
    
public:
    
    //Primary constructor takes in the GMLMPop setup
    GPUGMLMPop_parameters_GPU(const GPUGMLMPop_structure_args<FPTYPE> * GMLMPopstructure, const size_t dim_M_, const int dev_, std::shared_ptr<GPUGL_msg> msg_);
        
    //destructor
    ~GPUGMLMPop_parameters_GPU();
    
    //sends user-provided parameters to the GPU loading with the given streams
        //opts gives the user-provided trial weights. if opts is NULL, assumes all trial_weights are 1
    void copyToGPU(const GPUGMLMPop_params<FPTYPE> * gmlm_params, GPUGMLMPop_dataset_GPU<FPTYPE> * dataset,  const cudaStream_t stream, const std::vector<cudaStream_t> stream_Groups, const GPUGMLMPop_computeOptions<FPTYPE> * opts = NULL);

    inline size_t getNumberOfNonzeroWeights() const  {
        return trial_included->getSize(0);
    }
    
    inline GPUData<FPTYPE> * getTrialWeights() {
        return trial_weights;
    }
    
    //dimensions
    inline size_t dim_M() const {
        return trial_weights_temp->getSize_max(0);
    }
    inline size_t dim_J() const { //number of coefficient tensor groups
        return Groups.size();
    }
    inline size_t dim_P() const { // number of neurons
        return W->getSize(0);
    }
    inline size_t dim_B() const { // number of linear coefs
        return B->getSize(0);
    }
    inline size_t dim_R(int jj) const { // rank of a Group
        return Groups[jj]->dim_R();
    }
};

template <class FPTYPE>
class GPUGMLMPop_parameters_Group_GPU : GPUGL_base {
private:
    //neuron loading weights
    GPUData<FPTYPE>  * V;  //size dim_P x dim_R_max
    
    GPUData<bool> * compute_dT = NULL;
    GPUData<bool> * compute_dF = NULL;
    
    //params for each tensor dimension
    std::vector<GPUData<FPTYPE> *> T; //T[ss] is size dim_T[s] x dim_R_max
    std::vector<GPUData<FPTYPE> *> F; //F[dd] is size dim_F[dd] x dim_R_max
    
    std::vector<GPUData<FPTYPE> *> dF_dT; //dF_dT[ss] is size dim_T[ss] x dim_F[factor_idx[ss]] x dim_R_max
    
    GPUData<unsigned int> * N_per_factor = NULL;
    GPUData<unsigned int> * factor_idx = NULL; // size dim_S - how each dimensions's coefficients are decomposed in full->partial CP (0:(dim_S-1) for full multilinear, all 0's for a complete tensor)
    
    size_t dim_F_max;
    
    const GPUGMLMPop_parameters_GPU<FPTYPE> * parent;
    
    friend class GPUGMLMPop_computeBlock<FPTYPE>; 
    friend class GPUGMLMPop_dataset_Group_GPU<FPTYPE>; 
    friend class GPUGMLMPop_results_Group_GPU<FPTYPE>;
    
public:
    
    //Primary constructor takes in the GMLMPop setup
    GPUGMLMPop_parameters_Group_GPU(const GPUGMLMPop_structure_Group_args<FPTYPE> * GMLMPopGroupStructure, const GPUGMLMPop_parameters_GPU<FPTYPE> * parent_);
        
    //destructor
    ~GPUGMLMPop_parameters_Group_GPU();
    
    void copyToGPU(const GPUGMLMPop_group_params<FPTYPE> * gmlm_group_params, const cudaStream_t stream, const GPUGMLMPop_group_computeOptions * opts);
    
    void assembleF(const cudaStream_t stream);
    inline GPUData<FPTYPE> * getTrialWeights() const {
        return parent->trial_weights;
    }
    
    //dimensions
    inline size_t dim_D() const { //xx
        return F.size();
    }
    inline size_t dim_P() const  { //xx
        return parent->dim_P();
    }
    inline size_t dim_S() const { //xx
        return T.size();
    }
    inline size_t dim_T(int idx) const  {
        return T[idx]->getSize(0);
    }
    inline size_t dim_F(int idx) const  {
        return F[idx]->getSize(0);
    }
    inline size_t dim_R() const  { //xx
        return V->getSize(1);
    }
    inline size_t dim_R_max() const  { //xx
        return V->getSize_max(1);
    }
    inline cudaError_t set_dim_R(const int dim_R_new, const cudaStream_t stream) {
        if(dim_R_new < 0 || dim_R_new > dim_R_max()) {
            return cudaErrorInvalidValue;
        }
        else {
            cudaError_t ce = V->resize(stream, -1, dim_R_new, -1);
            
            for(int ss = 0; ss < dim_S() && ce == cudaSuccess; ss++) {
                ce = T[ss]->resize(stream, -1, dim_R_new, -1);
            }
            for(int dd = 0; dd < dim_D() && ce == cudaSuccess; dd++) {
                ce = F[dd]->resize(stream, -1, dim_R_new, -1);
            }
            for(int ss = 0; ss < dim_S() && ce == cudaSuccess; ss++) {
                ce = dF_dT[ss]->resize(stream, -1, -1, dim_R_new);
            }
            return ce;
        }
    }
};

/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
 * RESULTS on GPU 
 * 
 * 
 *
 */
template <class FPTYPE>
class GPUGMLMPop_results_GPU : GPUGL_base {
private:
    friend class GPUGMLMPop_results_Group_GPU<FPTYPE>; //to give groups access to the main vars
    friend class GPUGMLMPop_computeBlock<FPTYPE>; 
    
    //each trials's log like
    GPUData<FPTYPE> * trialLL; // max_trials x dim_P
    
    //derivative of each neuron's baseline rate
    GPUData<FPTYPE> * dW;
    
    //derivative of each neuron's linear terms
    GPUData<FPTYPE> * dB;
    
    //each group
    std::vector<GPUGMLMPop_results_Group_GPU<FPTYPE> *> Groups;
    
public:
    
    //Primary constructor takes in the GMLMPop setup
    GPUGMLMPop_results_GPU(const GPUGMLMPop_structure_args <FPTYPE> * GMLMPopstructure, const size_t max_trials_, const int dev_, std::shared_ptr<GPUGL_msg> msg_);
        
    //destructor
    ~GPUGMLMPop_results_GPU();
    
    //brings results from GPU to page-locked host memory
    void gatherResults(const GPUGMLMPop_parameters_GPU<FPTYPE> * params, const GPUGMLMPop_computeOptions<FPTYPE> * opts, const cudaStream_t stream_main, const std::vector<cudaStream_t> stream_Groups);
    //adds results in page-locked host memory to user-supplied object for returning
    //  If reset is false, adds. If true, sets to 0 before adding.
    void addToHost(const GPUGMLMPop_parameters_GPU<FPTYPE> * params, GPUGMLMPop_results<FPTYPE>* results_dest, const GPUGMLMPop_computeOptions<FPTYPE> * opts, const GPUGMLMPop_dataset_GPU<FPTYPE> * dataset, const bool reset = false);

    
    //dimensions
    inline size_t dim_J() const { //number of coefficient tensor groups
        return Groups.size();
    }
    inline size_t dim_P() const { // number of neurons
        return dW->getSize(0);
    }
    inline size_t dim_B() const { // number of linear coefs
        return dB->getSize(0);
    }
    inline size_t max_trials() const { //number of trials (in entire GMLMPop)
        return trialLL->getSize_max(0);
    }
    inline cudaError_t set_dim_R(const int groupNum, const int dim_R_new, const cudaStream_t stream) {
        return Groups[groupNum]->set_dim_R(dim_R_new, stream);
    }
};

template <class FPTYPE>
class GPUGMLMPop_results_Group_GPU : GPUGL_base {
private:
    friend class GPUGMLMPop_computeBlock<FPTYPE>; 
    friend class GPUGMLMPop_dataset_Group_GPU<FPTYPE>; 
    
    //derivative of the group's parameters
    GPUData<FPTYPE> * dV;
    
    std::vector<GPUData<FPTYPE> *> dT; 
    std::vector<GPUData<FPTYPE> *> dF; 
    
    std::vector<bool> dF_assigned;
    
    
    const GPUGMLMPop_results_GPU<FPTYPE> * parent;
    
public:
    
    //Primary constructor takes in the GMLMPop setup
    GPUGMLMPop_results_Group_GPU(const GPUGMLMPop_structure_Group_args<FPTYPE> * GMLMPopGroupStructure, const GPUGMLMPop_results_GPU<FPTYPE> * parent);
        
    //destructor
    ~GPUGMLMPop_results_Group_GPU();
    
    //brings results from GPU to page-locked host memory
    void gatherResults(const GPUGMLMPop_parameters_Group_GPU<FPTYPE> * params, const GPUGMLMPop_group_computeOptions * opts, const cudaStream_t stream);
    //adds results in page-locked host memory to user-supplied object for returning
    //  If reset is false, adds. If true, sets to 0 before adding.
    void addToHost(const GPUGMLMPop_parameters_Group_GPU<FPTYPE> * params, GPUGMLMPop_group_results<FPTYPE>* results_dest, const GPUGMLMPop_group_computeOptions * opts, const GPUGMLMPop_dataset_GPU<FPTYPE> * dataset, const bool reset);

    
    //dimensions
    inline size_t dim_P() const  {
        return parent->dim_P();
    }
    inline size_t dim_S() const {
        return dT.size();
    }
    inline size_t dim_D() const {
        return dF.size();
    }
    inline size_t dim_T(int idx) const  {
        return dT[idx]->getSize(0);
    }
    inline size_t dim_R() const  {
        return dV->getSize(1);
    }
    inline size_t dim_R_max() const  {
        return dV->getSize_max(1);
    }
    inline cudaError_t set_dim_R(const int dim_R_new, const cudaStream_t stream) {
        if(dim_R_new < 0 || dim_R_new > dim_R_max()) {
            return cudaErrorInvalidValue;
        }
        else {
            cudaError_t ce = dV->resize(stream, -1, dim_R_new, -1);
            
            for(int ss = 0; ss < dim_S() && ce == cudaSuccess; ss++) {
                ce = dT[ss]->resize(stream, -1, dim_R_new, -1);
            }
            for(int dd = 0; dd < dim_D() && ce == cudaSuccess; dd++) {
                ce = dF[dd]->resize(stream, -1, dim_R_new, -1);
            }
            return ce;
        }
    }
};

/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
 * DATASET on GPU 
 * 
 */
template <class FPTYPE>
class GPUGMLMPop_dataset_GPU : GPUGL_base {
private:
    friend class GPUGMLMPop_dataset_Group_GPU<FPTYPE>;
    friend class GPUGMLMPop_results_GPU<FPTYPE>; 
    friend class GPUGMLMPop_parameters_GPU<FPTYPE>; 
    friend class GPUGMLMPop_results_Group_GPU<FPTYPE>; 
    friend class GPUGMLMPop_computeBlock<FPTYPE>; 
    friend class GPUGMLMPop_dataset_Group_GPU<FPTYPE>;

    FPTYPE log_dt; // log of bin size
    
    size_t max_trial_length; // max trial length for pre-allocation
    
    size_t max_trials_for_sparse_run;
    
    std::vector<bool> isInDataset_trial;  // size max_trials - if each trial is in this block
    
    GPUData<size_t> * dim_N = NULL;  //X trial lengths (size dim_M x 1)
    
    size_t dim_N_temp; // number of bins in this block for current sparse run (running only some trials)
    
    //spike counts (all trials)
    GPUData<FPTYPE> * Y = NULL; // size   dim_N_total x dim_P
    
    GPUData<FPTYPE> * normalizingConstants_trial = NULL; //X normalizing constant for each trial (dim_M x dim_P)
    
    //linear terms (split by neuron)
    GPUData<FPTYPE> * X_lin = NULL;  // X_lin is dim_N_total x dim_B x dim_P (third dim can be 1 so that all neurons have same linear term)
    
    //indices
    GPUData<unsigned int> * ridx_t_all = NULL;    //X idx of each trial into complete data arrays (like Y)
    
    GPUData<unsigned int> * ridx_st_sall = NULL;       // idx of each trial (indexed by current run on non-zero weighted trials) into sparse index
    GPUData<unsigned int> * ridx_sa_all = NULL;        // idx of sparse run indices into complete data arrays
    GPUData<unsigned int> * ridx_a_all = NULL;        // dummy array for more transparent full/sparse runs
    GPUData<unsigned int> * ridx_a_all_c = NULL;
    GPUData<unsigned int> * ridx_t_all_c = NULL;
    
    GPUData<unsigned int> * id_t_trial = NULL;  //X total trial number of each trial (total is across all GPUs, indexes into trial_weights and trialLL)   size is dim_M x 1
    
    GPUData<unsigned int> * id_a_trialM;  //X local trial index of each observation  size is dim_N_total x 1
                        // the M says that these numbers are 0:(dim_M-1). This is a reminder that it's different numbers than id_t_trial (without the M)
    
    //compute space
    GPUData<FPTYPE> * lambda = NULL; // dim_N_total x dim_P x dim_J, contribution to the log rate for each group
    
    GPUData<FPTYPE> *  LL = NULL; // size   dim_N_total x dim_P
    GPUData<FPTYPE> * dLL = NULL; // size   dim_N_total x dim_P
    
    GPUData<FPTYPE> *  dW_trial = NULL; // size   dim_M x dim_P
    
    GPUData<FPTYPE> * X_lin_temp = NULL; // temporary X_lin space for sparse computations, size  max_trial_length*max_trials_for_sparse_run x dim_B x dim_P
    
    //groups
    std::vector<GPUGMLMPop_dataset_Group_GPU<FPTYPE> *> Groups;
    
public:
    
    //Primary constructor takes in all the block data and GMLMPop setup
    GPUGMLMPop_dataset_GPU(const GPUGMLMPop_structure_args<FPTYPE> * GMLMPopstructure, const GPUGMLMPop_GPU_block_args <FPTYPE> * block, const size_t max_trials_,  const cudaStream_t stream, const std::vector<cusparseHandle_t> & cusparseHandle_Groups, std::shared_ptr<GPUGL_msg> msg_);
        
    //destructor
    ~GPUGMLMPop_dataset_GPU();
    
    inline bool isSparseRun(const GPUGMLMPop_parameters_GPU<FPTYPE> * params) {
        return params->getNumberOfNonzeroWeights() <= max_trials_for_sparse_run;
    }


    inline cudaError_t waitForGroups_LL(cudaStream_t stream) {
        cudaError_t ce = cudaSuccess;
        for(auto jj : Groups) {
            ce = cudaStreamWaitEvent(stream, jj->LL_event, 0);
            if(ce != cudaSuccess) {
                break;
            }
        }
        return ce;
    }
    
    //dimensions
    inline size_t dim_J() const { //number of coefficient tensor groups
        return Groups.size();
    }
    inline size_t dim_P() const { // number of neurons
        return Y->getSize(1);
    }
    inline size_t dim_B() const { // number of linear coefs
        return X_lin->getSize(1);
    }
    inline size_t dim_M() const { // number of trials
        return dim_N->getSize(0);
    }
    inline size_t dim_N_total() const { // number of bins in this block (sum dim_N)
    	return Y->getSize(0);
    }
    inline size_t max_trials() const {
        return isInDataset_trial.size();
    }
};

//The tensor groups contain both the data and the compute operations
//   GPUGMLMPop_dataset_Group_GPU is an abstract class
template <class FPTYPE>
class GPUGMLMPop_dataset_Group_GPU : protected GPUGL_base {
protected:
    friend class GPUGMLMPop_computeBlock<FPTYPE>; 
    friend class GPUGMLMPop_dataset_GPU<FPTYPE>; 
    
    const GPUGMLMPop_dataset_GPU<FPTYPE> * parent;
    
    const int groupNum;
    
    size_t dim_A;          // number of events for this group
    
    //regressors
    std::vector<GPUData<FPTYPE> *> X; // the regressors for each facotr group (dim_D x 1)
                 // if isShared[dd]==false -  X[dd] is  parent->dim_N_total x dim_F[dd] x (dim_A or 1), if third dim is 1 and dim_A > 0, uses same regressors for each event
                 // if isShared[dd]==true  -  X[dd] is  dim_X[ss] x dim_F[dd]
    
    std::vector<GPUData<int> *> iX; //indices into the rows of X[dd] for any isShared[dd]==true (dim_D x 1)
                 // if isShared[dd]==false -  iX_shared[dd] is  empty
                 // if isShared[dd]==true  -  iX_shared[dd] is  parent->dim_N_total x dim_A
    
    GPUData<bool> * isShared = NULL; // if the regressors for each factor group are shared (global) or local (dim_D x 1)
    GPUData<bool> * isSharedIdentity = NULL; // if the shared regressors for each factor group are the identity matrix (dim_D x 1)
    
    
    //compute space
    //   regressors times parameters
    std::vector<GPUData<FPTYPE> *>  XF; // each X times T (dim_D of them)
                 // if isShared[dd]==false -  XF[dd] is  parent->dim_N_total x dim_R_max x (dim_A or 1)
                 // if isShared[dd]==true  -  XF[dd] is  dim_X[dd] x dim_R_max

    
    //  compute space for derivatives
    GPUData<FPTYPE> * lambda_v = NULL; // dim_N_total x dim_R_max
    
    std::vector<GPUData<FPTYPE> *> lambda_d; // (dim_N_total x dim_R_max) x (dim_A or 1)
    
    GPUData<FPTYPE> * phi_d = NULL; // max({dim_N_total, X[dd].getSize(0) : isShared[dd]}) x dim_R_max
    
    //for sparse runs
    std::vector<GPUData<FPTYPE> *> X_temp;
    GPUData<FPTYPE> * buffer;
    
    
    // sparse matrices for shared regressor derivatives
    std::vector<GPUData<int> *> spi_rows;
    std::vector<GPUData<int> *> spi_cols;
    
    std::vector<GPUData<FPTYPE> *>  spi_data;
    
    std::vector<cusparseSpMatDescr_t*> spi_S;
    std::vector<cusparseDnVecDescr_t*> spi_phi_d;
    std::vector<cusparseDnVecDescr_t*> spi_lambda_d;
    std::vector<GPUData<char> *>  spi_buffer;
    std::vector<size_t> spi_buffer_size;
    
    cudaEvent_t LL_event;
public:
    //constructor
    GPUGMLMPop_dataset_Group_GPU(const int groupNum_, const GPUGMLMPop_structure_Group_args<FPTYPE> * GMLMPopGroupStructure, const std::vector<GPUGMLMPop_trial_args <FPTYPE> *> trials, const GPUGMLMPop_dataset_GPU<FPTYPE> * parent_, const cudaStream_t stream, const cusparseHandle_t & cusparseHandle);
    
    //destructor
    ~GPUGMLMPop_dataset_Group_GPU();
    
    
    void multiplyCoefficients(const bool isSparseRun, const GPUGMLMPop_parameters_Group_GPU<FPTYPE> * params, const cudaStream_t stream, const cublasHandle_t cublasHandle, cudaEvent_t & paramsLoaded);
    void getGroupRate(const bool isSparseRun, const GPUGMLMPop_parameters_Group_GPU<FPTYPE> * params, const GPUGMLMPop_group_computeOptions * opts, const cudaStream_t stream, const cublasHandle_t cublasHandle);
    void computeDerivatives(GPUGMLMPop_results_Group_GPU<FPTYPE> * results, const bool isSparseRun, GPUGMLMPop_parameters_Group_GPU<FPTYPE> * params, const GPUGMLMPop_group_computeOptions * opts, const cudaStream_t stream, const cublasHandle_t cublasHandle, const cusparseHandle_t cusparseHandle, cudaEvent_t & main_LL_event);

    //dimensions
    inline size_t dim_P() const  {
        return parent->dim_P();
    }
    inline size_t dim_X(int ff) const  {
        return X[ff]->getSize(0);
    }
    inline size_t dim_F(int idx) const  {
        return X[idx]->getSize(1);
    }
    inline size_t dim_D() const { // number of factors for tensor decomposition (if == dim_S, than is full CP, otherwise parts are full tensors
        return X.size();
    }
    inline size_t dim_R() const  { // current rank
        return XF[0]->getSize(1);
    }
    inline size_t dim_R_max() const  { // max allocated rank
        return XF[0]->getSize_max(1);
    }
    inline cudaError_t set_dim_R(const int dim_R_new, const cudaStream_t stream) { // set rank
        cudaError_t ce = (dim_R_new >=0 && dim_R_new <= dim_R_max()) ? cudaSuccess : cudaErrorInvalidValue;
        if(dim_R_new != dim_R()) {
            ce = lambda_v->resize(stream, -1, dim_R_new, -1);
            if(ce == cudaSuccess) {
                ce = phi_d->resize(stream, -1, dim_R_new, -1);
            }
            for(int dd = 0; dd < XF.size() && ce == cudaSuccess; dd++) {
                ce = XF[dd]->resize(stream, -1, dim_R_new, -1);
            }
            for(int dd = 0; dd < lambda_d.size() && ce == cudaSuccess; dd++) {
                ce = lambda_d[dd]->resize(stream, -1, dim_R_new, -1);
            }
        }
        return ce;
    }
};
        



}; //namespace
#endif