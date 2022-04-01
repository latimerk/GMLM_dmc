/*
 * kcGMLM_mex_computeLL_async.cu
 * Mex function to call the log likelihood function in a GMLM object. Does not copy results back (use the gather function to get results), but returns to CPU for asynchronous operations.
 *  Takes 4-5 arguments: ptr (in long int form) to GMLM object
 *                       boolean (is double) for if the object is double precision (or single)
 *                       params struct (from GMLM class)
 *                       results struct (from GMLM class) - empty results 
 *                                matrices mean do not compute certain parts (like derivatives)
 *                       trial weights (optional) - a vector of weights for each trial
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
#include "kcGMLM_mex_shared.hpp"
#include "kcGMLM.hpp"
#include <cmath>
#include <string>

// using namespace std::chrono; 
  
//creates a c++ param object for running the GMLM log likelihood from a matlab structure.
//This requires that the matlab matrices match the precision of the GMLM -> it will not cast from double to single!
//This performs no data copying: the pointers to the matlab matrices are taken.
//(not safe to pass back to matlab from reuse: if those pointers are cleared by matlab, seg faults could happen!)
template <class FPTYPE>
class matlab_GPUGMLM_params : public kCUDA::GPUGMLM_params<FPTYPE> {
private:
    
public:
    //constructor
    matlab_GPUGMLM_params(const matlab::data::StructArray & GMLM_params, const std::shared_ptr<matlab::engine::MATLABEngine> & matlabPtr ) { 
        //gets the W input and its size
        const matlab::data::TypedArray<const FPTYPE> W_mat = GMLM_params[0]["W"];
        this->W =  new GLData_matlab<FPTYPE>(W_mat);
        
        //gets the linear weights
        const matlab::data::TypedArray<const FPTYPE> B_mat = GMLM_params[0]["B"];
        this->B = new GLData_matlab<FPTYPE>(B_mat);
        if(this->W->getSize(0) != this->B->getSize(1) && !(this->B->empty())) {
            matlab::data::ArrayFactory factory;
            matlabPtr->feval(u"error", 0,std::vector<matlab::data::Array>({ factory.createScalar("inconsistent dimensions in param struct: W rows does not match B cols (number of neurons)") }));
        }
        
        //gets the groups struct and its size (the number of groups)
        const matlab::data::StructArray Groups_mat = GMLM_params[0]["Groups"];
        size_t dim_J  = Groups_mat.getNumberOfElements();
        this->Groups.resize(dim_J);
        for(int jj = 0; jj < dim_J; jj++) {
            //for each group: creates a group param object
            //this->Groups[jj] = new kCUDA::GPUGMLM_group_params<FPTYPE>();
            this->Groups[jj] = new kCUDA::GPUGMLM_group_params<FPTYPE>();
            
            //get's the V input (neuron loadings)
            // V must be dim_P x dim_R
            const matlab::data::TypedArray<const FPTYPE> V_mat = Groups_mat[jj]["V"];
            if(this->W->getSize(0) != V_mat.getDimensions()[0]) {
                matlab::data::ArrayFactory factory;
                matlabPtr->feval(u"error", 0,std::vector<matlab::data::Array>({ factory.createScalar("inconsistent dimensions in param struct: V rows does not match W") }));
            }
            this->Groups[jj]->V = new GLData_matlab<FPTYPE>(V_mat);
            
            //gets the T inputs
            const matlab::data::CellArray T_mats = Groups_mat[jj]["T"];
            size_t dim_S = T_mats.getNumberOfElements(); //number of T's: order of tensor (minus the neuron dimension)
            this->Groups[jj]->T.resize(dim_S);
            for(int ss = 0; ss < dim_S; ss++) {
                //for each dimension, get's the T and check's size (T{ss} is dim_T[ss] x dim_R
                const matlab::data::TypedArray<const FPTYPE> T_mat = T_mats[ss];
                if(T_mat.getDimensions()[1] != this->Groups[jj]->V->getSize(1)) {
                    matlab::data::ArrayFactory factory;
                    matlabPtr->feval(u"error", 0,std::vector<matlab::data::Array>({ factory.createScalar("inconsistent dimensions in param struct: T rank does not match V rank") }));
                }
                this->Groups[jj]->T[ss]     = new GLData_matlab<FPTYPE>(T_mat);
            }
        }
    }
    //destructor
    ~matlab_GPUGMLM_params() {
        //clear the groups
        delete this->W;
        delete this->B;
        for(int jj = 0; jj < this->Groups.size(); jj++) {
            for(int ss = 0; ss < this->Groups[jj]->T.size(); ss++) {
                delete this->Groups[jj]->T[ss];
            }
            delete this->Groups[jj]->V;
            delete this->Groups[jj];
        }
    }
};

//creates an compute options object (i.e., which derivatives to compute) from the a matlab results struct
//If the result's field is empty, the compute option is false for that derivative. Otherwise, it's true.
template <class FPTYPE>
class matlab_GPUGMLM_computeOptions : public kCUDA::GPUGMLM_computeOptions<FPTYPE> {
public:
    matlab_GPUGMLM_computeOptions(const matlab::data::StructArray & GMLM_opts, const matlab::data::TypedArray<FPTYPE> * trial_weights_mat = NULL) {

        const matlab::data::TypedArray<bool> trialLL = GMLM_opts[0]["trialLL"];
        this->compute_trialLL =  trialLL[0];
        
        //checks for an assigned dW vector
        const matlab::data::TypedArray<bool> dW = GMLM_opts[0]["dW"];
        this->compute_dW =  dW[0];
        
        //checks for an assigned dB vector
        const matlab::data::TypedArray<bool> dB = GMLM_opts[0]["dB"];
        this->compute_dB =  dB[0];
        
        //sets up trial weights
        this->trial_weights = new GLData_matlab<FPTYPE>(trial_weights_mat);
        
        //for each group
        const matlab::data::StructArray Groups_mat = GMLM_opts[0]["Groups"];
        size_t dim_J  = Groups_mat.getNumberOfElements();
        this->Groups.resize(dim_J);
        for(int jj = 0; jj < dim_J; jj++) {
            this->Groups[jj] = new kCUDA::GPUGMLM_group_computeOptions;
            
            //checks for a dV matrix
            const matlab::data::TypedArray<bool> dV = Groups_mat[jj]["dV"];
            this->Groups[jj]->compute_dV =  dV[0];
            
            //for the remaining dimensions, looks at all T
            const matlab::data::TypedArray<bool> dT = Groups_mat[jj]["dT"];
            
            size_t dim_S = dT.getNumberOfElements();
            this->Groups[jj]->compute_dT.resize(dim_S);
            for(int ss = 0; ss < dim_S; ss++) {
                //checks for a dT matrix
                this->Groups[jj]->compute_dT[ss] =  dT[ss];
            }
        }
    }
    //destructor
    ~matlab_GPUGMLM_computeOptions() {
        //clears all the groups
        delete this->trial_weights;
        for(int jj = 0; jj < this->Groups.size(); jj++) {
            delete this->Groups[jj];
        }
    }
};


class MexFunction : public matlab::mex::Function {
private:
    // Pointer to MATLAB engine to call fprintf
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    matlab::data::ArrayFactory factory;
    
    // Create an output stream
    std::ostringstream stream;
    
    GPUGL_msg_mex * msg;
    
private:
    

    template <class FPTYPE>
    void runComputation(const uint64_t GMLM_ptr, const matlab::data::StructArray & GMLM_params, const matlab::data::StructArray & GMLM_opts, const matlab::data::TypedArray<FPTYPE> * trial_weights) {
        if(GMLM_ptr == 0) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("gpu pointer is not initialized!") }));
            return;
        }
        
        kCUDA::GPUGMLM<FPTYPE>         * gmlm_obj = reinterpret_cast<kCUDA::GPUGMLM<FPTYPE> *>(GMLM_ptr); //forcefully recasts the uint64 value
        //gets the model parameters
        std::shared_ptr<kCUDA::GPUGMLM_params<FPTYPE>> params = std::make_shared<matlab_GPUGMLM_params<FPTYPE>>(GMLM_params, matlabPtr);  
        //gets the compute options based on which fields are available in restults
        std::shared_ptr<kCUDA::GPUGMLM_computeOptions<FPTYPE>> opts = std::make_shared<matlab_GPUGMLM_computeOptions<FPTYPE>>(GMLM_opts, trial_weights);  
        
        //runs the log likelihood computation. After, the results will be in the matlab arrays as results holds the pointers to those arrays.
        gmlm_obj->computeLogLikelihood_async(params, opts);
    }
    
    
//main func
public:
    
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        if(inputs.size() != 4  && inputs.size() != 5) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("mex function requires 4-5 arguments!") }));
        }
    
        //checks that arg 0 is a pointer value to the GMLM C++ object
        const matlab::data::TypedArray<uint64_t> gpu_ptr_a = inputs[0];
        if(gpu_ptr_a.getNumberOfElements() != 1) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("pointer argument must only contain 1 element!") }));
        }
        
        //get datatype field from GMLM object
        const matlab::data::TypedArray<bool> isDouble = inputs[1];
        if(isDouble.getNumberOfElements() != 1) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("cannot determine gpu data type!") }));
        }
        
        const matlab::data::StructArray GMLM_params  = inputs[2];
        const matlab::data::StructArray GMLM_opts    = inputs[3];
        
        
        if(gpu_ptr_a.getNumberOfElements() < 1 || gpu_ptr_a[0] == 0) {
            //no gmlm object to clear
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("gpu pointer is not initialized!") }));
        }
        else if(isDouble[0]) {
            if(inputs.size() >= 5 && !inputs[4].isEmpty()) {
                const matlab::data::TypedArray<double> trial_weights_mat = inputs[4];
                runComputation<double>(gpu_ptr_a[0], GMLM_params, GMLM_opts, &trial_weights_mat);
            }
            else {
                runComputation<double>(gpu_ptr_a[0], GMLM_params, GMLM_opts, NULL);
            }
        }
        else {
            if(inputs.size() >= 5 && !inputs[4].isEmpty()) {
                const matlab::data::TypedArray<float> trial_weights_mat = inputs[4];
                runComputation<float>(gpu_ptr_a[0], GMLM_params, GMLM_opts, &trial_weights_mat);
            }
            else {
                runComputation<float>(gpu_ptr_a[0], GMLM_params, GMLM_opts, NULL);
            }
        }
    }
};
