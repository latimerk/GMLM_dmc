/*
 * kcGMLM_dataStructures.hpp
 * Holds all the data - parameters, results, regressors, computation space
 * for a GMLM (on one GPU).
 *
 * Package GMLM_dmc for dimensionality reduction of neural data.
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
#ifndef GMLM_GMLM_DATASTRUCTURES_H
#define GMLM_GMLM_DATASTRUCTURES_H

#include "kcBase.hpp"
#include "kcGMLM_common.hpp"
namespace kCUDA {
    
/*The main datastructures for the GMLM on the GPU
 *      parameters
 *      results
 *      dataset (all data + pre-allocated compute space)
 *    
 *      I make overly generous use of friend classes to streamline some computation code.
 */
    
template <class FPTYPE> class GPUGMLM_parameters_Group_GPU;
template <class FPTYPE> class GPUGMLM_parameters_GPU;
template <class FPTYPE> class GPUGMLM_results_Group_GPU;
template <class FPTYPE> class GPUGMLM_results_GPU;
template <class FPTYPE> class GPUGMLM_dataset_Group_GPU;
template <class FPTYPE> class GPUGMLM_dataset_GPU;
template <class FPTYPE> class GPUGMLMPop_dataset_Group_GPU;
template <class FPTYPE> class GPUGMLMPop_dataset_GPU;

template <class FPTYPE> 
class GPUGMLM_computeBlock_base : public GPUGL_base {
    public:
        virtual bool loadParams(const GPUGMLM_params<FPTYPE> * params_host, const GPUGMLM_computeOptions<FPTYPE> * opts = NULL) { return false; };
        virtual void computeRateParts(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun) {};
        virtual void computeLogLike(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun) {};
        virtual void computeDerivatives(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun) {};
        virtual void gatherResults(const GPUGMLM_computeOptions<FPTYPE> * opts) {};
        virtual bool addResultsToHost(GPUGMLM_results<FPTYPE>* results_dest, const GPUGMLM_computeOptions<FPTYPE> * opts, const bool reset = false) { return false; };
        virtual void syncStreams() {};
        virtual ~GPUGMLM_computeBlock_base() {};
};
};

#include "kcGMLM_computeBlock.hpp"
#include "kcGMLMPop_computeBlock.hpp"


namespace kCUDA {


/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
 * PARAMETERS on GPU 
 */

template <class FPTYPE>
class GPUGMLM_parameters_GPU : GPUGL_base {
private:
    friend class GPUGMLM_parameters_Group_GPU<FPTYPE>; //to give groups access to the main vars
    friend class GPUGMLM_computeBlock_base<FPTYPE>; 
    friend class GPUGMLM_computeBlock<FPTYPE>; 
    friend class GPUGMLMPop_computeBlock<FPTYPE>; 
    friend class GPUGMLM_results_GPU<FPTYPE>;
    
    //each neuron's baseline rate
    GPUData<FPTYPE> * W; //vectory size dim_P
    //each neuron's linear terms
    GPUData<FPTYPE> * B;
    
    GPUData<FPTYPE> * trial_weights;                  // can be NULL or trial_weights_temp (NULL means all weights == 1)
    GPUData<FPTYPE> * trial_weights_0;                // is (dim_M x 0) 
    GPUData<FPTYPE> * trial_weights_temp;             // vector size dim_M
    
    GPUData<unsigned int> * trial_included;           // can be NULL or trial_included_temp (NULL means not doing a sparse run)
    GPUData<unsigned int> * trial_included_temp;      // vector size trial_weights_nonzero_cnt (0 to max_trials)
                                                      // list of which trials are currently being computed (non-zero weights) - page-locked host + GPU memory 
    GPUData<unsigned int> * trial_included_0;         // is (dim_M x 0) 
    
    logLikeType logLikeSettings;
    GPUData<FPTYPE> * logLikeParams;

    cudaEvent_t	paramsLoaded_event; 	

    const bool isSimultaneousPopulation;
    
    //each group
    std::vector<GPUGMLM_parameters_Group_GPU<FPTYPE> *> Groups;
    
    
public:
    
    //Primary constructor takes in the GMLM setup
    GPUGMLM_parameters_GPU(const GPUGMLM_structure_args<FPTYPE> * GMLMstructure, const size_t dim_M_, const int dev_, std::shared_ptr<GPUGL_msg> msg_);
        
    //destructor
    ~GPUGMLM_parameters_GPU();
    
    //sends user-provided parameters to the GPU loading with the given streams
        //opts gives the user-provided trial weights. if opts is NULL, assumes all trial_weights are 1
    void copyToGPU(const GPUGMLM_params<FPTYPE> * gmlm_params, GPUGMLM_dataset_GPU<FPTYPE> * dataset,  const cudaStream_t stream, const std::vector<cudaStream_t> stream_Groups, const GPUGMLM_computeOptions<FPTYPE> * opts = NULL);
    void copyToGPU(const GPUGMLM_params<FPTYPE> * gmlm_params, GPUGMLMPop_dataset_GPU<FPTYPE> * dataset,  const cudaStream_t stream, const std::vector<cudaStream_t> stream_Groups, const GPUGMLM_computeOptions<FPTYPE> * opts = NULL);

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
class GPUGMLM_parameters_Group_GPU : GPUGL_base {
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
    
    const GPUGMLM_parameters_GPU<FPTYPE> * parent;
    
    friend class GPUGMLM_computeBlock<FPTYPE>; 
    friend class GPUGMLM_dataset_Group_GPU<FPTYPE>; 
    friend class GPUGMLMPop_computeBlock<FPTYPE>; 
    friend class GPUGMLMPop_dataset_Group_GPU<FPTYPE>;
    friend class GPUGMLM_results_Group_GPU<FPTYPE>;
    
public:
    
    //Primary constructor takes in the GMLM setup
    GPUGMLM_parameters_Group_GPU(const GPUGMLM_structure_Group_args<FPTYPE> * GMLMGroupStructure, const GPUGMLM_parameters_GPU<FPTYPE> * parent_);
        
    //destructor
    ~GPUGMLM_parameters_Group_GPU();
    
    void copyToGPU(const GPUGMLM_group_params<FPTYPE> * gmlm_group_params, const cudaStream_t stream, const GPUGMLM_group_computeOptions * opts);
    
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
class GPUGMLM_results_GPU : GPUGL_base {
private:
    friend class GPUGMLM_results_Group_GPU<FPTYPE>; //to give groups access to the main vars
    friend class GPUGMLM_computeBlock<FPTYPE>; 
    friend class GPUGMLMPop_computeBlock<FPTYPE>; 
    
    //each trials's log like
    GPUData<FPTYPE> * trialLL;
    
    //derivative of each neuron's baseline rate
    GPUData<FPTYPE> * dW;
    
    //derivative of each neuron's linear terms
    GPUData<FPTYPE> * dB;
    
    //each group
    std::vector<GPUGMLM_results_Group_GPU<FPTYPE> *> Groups;
    
    const bool isSimultaneousPopulation;

public:
    
    //Primary constructor takes in the GMLM setup
    GPUGMLM_results_GPU(const GPUGMLM_structure_args <FPTYPE> * GMLMstructure, const size_t max_trials_,  const int dev_, std::shared_ptr<GPUGL_msg> msg_);
        
    //destructor
    ~GPUGMLM_results_GPU();
    
    //brings results from GPU to page-locked host memory
    void gatherResults(const GPUGMLM_parameters_GPU<FPTYPE> * params, const GPUGMLM_computeOptions<FPTYPE> * opts, const cudaStream_t stream_main, const std::vector<cudaStream_t> stream_Groups);
    //adds results in page-locked host memory to user-supplied object for returning
    //  If reset is false, adds. If true, sets to 0 before adding.
    void addToHost(const GPUGMLM_parameters_GPU<FPTYPE> * params, GPUGMLM_results<FPTYPE>* results_dest, const GPUGMLM_computeOptions<FPTYPE> * opts, const std::vector<bool> * isInDataset_trial, const std::vector<size_t> * dim_N_neuron_temp, const bool reset = false);

    
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
    inline size_t max_trials() const { //number of trials (in entire GMLM)
        return trialLL->getSize_max(0);
    }
    inline cudaError_t set_dim_R(const int groupNum, const int dim_R_new, const cudaStream_t stream) {
        return Groups[groupNum]->set_dim_R(dim_R_new, stream);
    }
};

template <class FPTYPE>
class GPUGMLM_results_Group_GPU : GPUGL_base {
private:
    friend class GPUGMLM_computeBlock<FPTYPE>; 
    friend class GPUGMLM_dataset_Group_GPU<FPTYPE>; 
    friend class GPUGMLMPop_computeBlock<FPTYPE>; 
    friend class GPUGMLMPop_dataset_Group_GPU<FPTYPE>; 
    
    //derivative of the group's parameters
    GPUData<FPTYPE> * dV;
    
    std::vector<GPUData<FPTYPE> *> dT; 
    std::vector<GPUData<FPTYPE> *> dF; 
    
    std::vector<bool> dF_assigned;
    
    
    const GPUGMLM_results_GPU<FPTYPE> * parent;
    
public:
    
    //Primary constructor takes in the GMLM setup
    GPUGMLM_results_Group_GPU(const GPUGMLM_structure_Group_args<FPTYPE> * GMLMGroupStructure, const GPUGMLM_results_GPU<FPTYPE> * parent);
        
    //destructor
    ~GPUGMLM_results_Group_GPU();
    
    //brings results from GPU to page-locked host memory
    void gatherResults(const GPUGMLM_parameters_Group_GPU<FPTYPE> * params, const GPUGMLM_group_computeOptions * opts, const cudaStream_t stream);
    //adds results in page-locked host memory to user-supplied object for returning
    //  If reset is false, adds. If true, sets to 0 before adding.
    void addToHost(const GPUGMLM_parameters_Group_GPU<FPTYPE> * params, GPUGMLM_group_results<FPTYPE>* results_dest, const GPUGMLM_group_computeOptions * opts, const std::vector<size_t> * dim_N_neuron_temp, bool isSimultaneousPopulation, const bool reset);

    
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

        



}; //namespace
#endif