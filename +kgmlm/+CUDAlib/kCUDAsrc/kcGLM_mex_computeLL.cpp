/*
 * kcGLM_mex_computeLL.cu
 * Mex function to call the log likelihood function in a GLM object.
 *  Takes 4-5 arguments: ptr (in long int form) to GLM object
 *                       boolean (is double) for if the object is double precision (or single)
 *                       params struct (from kcGLM class)
 *                       results struct (from kcGLM class) - empty results 
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
#include "kcGLM.hpp"
#include <cmath>
#include <string>

// using namespace std::chrono; 
  
//creates a c++ param object for running the GLM log likelihood from a matlab structure.
//This requires that the matlab matrices match the precision of the GLM -> it will not cast from double to single!
//This performs no data copying: the pointers to the matlab matrices are taken.
//(not safe to pass back to matlab from reuse: if those pointers are cleared by matlab, seg faults could happen!)
template <class FPTYPE>
class matlab_GPUGLM_params : public kCUDA::GPUGLM_params<FPTYPE> {
public:
    //constructor
    matlab_GPUGLM_params(const matlab::data::StructArray & GLM_params) {
        //gets the W input and its size
        const matlab::data::TypedArray<FPTYPE> K_mat = GLM_params[0]["K"];
        this->K = new GLData_matlab<FPTYPE>(K_mat);
    }
    //destructor
    ~matlab_GPUGLM_params() {
        delete this->K;
    }
};

//creates an compute options object (i.e., which derivatives to compute) from the a matlab results struct
//If the result's field is empty, the compute option is false for that derivative. Otherwise, it's true.
template <class FPTYPE>
class matlab_GPUGLM_computeOptions : public kCUDA::GPUGLM_computeOptions<FPTYPE> {
public:
    matlab_GPUGLM_computeOptions(const matlab::data::StructArray & GLM_results, const std::vector<FPTYPE> & trial_weights) {

        //checks for an assigned dK vector
        this->compute_dK =  !(GLM_results[0]["dK"].isEmpty());
        //checks for an assigned d2K vector
        this->compute_d2K =  !(GLM_results[0]["d2K"].isEmpty());
        
        //checks for an assigned trialLL vector
        this->compute_trialLL =  !(GLM_results[0]["trialLL"].isEmpty());
        
        //sets up trial weights
        this->trial_weights = std::vector<FPTYPE>(trial_weights);
    }
    //destructor
    ~matlab_GPUGLM_computeOptions() {
    }
};

//creates a results object to send into the GLM computations. It takes in a matlab struct and uses the results fields given in that struct.
//This requires that the matlab matrices match the precision of the GLM -> it will not cast from double to single!
//This creates no new space: it gets the matlab pointers and does a pass-by-values assignment.
//This bypasses the MATLAB C++ API in mata::data for assignments, because going through there can cause significant overhead.
//(not safe to pass back to matlab from reuse: if those pointers are cleared by matlab, seg faults could happen!)
template <class FPTYPE>
class matlab_GPUGLM_results : public kCUDA::GPUGLM_results<FPTYPE> {
public:
    matlab_GPUGLM_results(const matlab::data::StructArray & GLM_results, const kCUDA::GPUGLM_computeOptions<FPTYPE>    * opts) {
        //assigns the pointers to the dK and d2K results 
        if(opts->compute_dK) {
            const matlab::data::TypedArray<FPTYPE> dK_mat = GLM_results[0]["dK"];
            this->dK =  new GLData_matlab<FPTYPE>(dK_mat);
        }
        else {
            this->dK = new kCUDA::GLData<FPTYPE>(NULL, 0);
        }
        
        if(opts->compute_d2K) {
            const matlab::data::TypedArray<FPTYPE> d2K_mat = GLM_results[0]["d2K"];
            this->d2K =  new GLData_matlab<FPTYPE>(d2K_mat);
        }
        else {
            this->d2K = new kCUDA::GLData<FPTYPE>(NULL, 0);
        }
        
        //assigns the pointers to the trial-wise log likelihood results 
        if(opts->compute_trialLL) {
            const matlab::data::TypedArray<FPTYPE> trialLL_mat = GLM_results[0]["trialLL"];
            this->trialLL =  new GLData_matlab<FPTYPE>(trialLL_mat);
        }
        else {
            this->trialLL = new kCUDA::GLData<FPTYPE>(NULL, 0);
        }
    }
    //destructor
    ~matlab_GPUGLM_results() {
        delete this->trialLL;
        delete this->dK;
        delete this->d2K;
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
    void runComputation(const uint64_t GLM_ptr, const matlab::data::StructArray & GLM_params, const matlab::data::StructArray & GLM_results, const std::vector<FPTYPE> & trial_weights) {
        if(GLM_ptr == 0) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("gpu pointer is not initialized!") }));
            return;
        }
        kCUDA::GPUGLM<FPTYPE>         * glm_obj = reinterpret_cast<kCUDA::GPUGLM<FPTYPE> *>(GLM_ptr); //forcefully recasts the uint64 value
        //gets the model parameters
        kCUDA::GPUGLM_params<FPTYPE>  * params = new matlab_GPUGLM_params<FPTYPE>(GLM_params);
        //gets the compute options based on which fields are available in restults
        kCUDA::GPUGLM_computeOptions<FPTYPE>    * opts    = new matlab_GPUGLM_computeOptions<FPTYPE>(GLM_results, trial_weights);
        //gets the model results
        kCUDA::GPUGLM_results<FPTYPE> * results = new matlab_GPUGLM_results<FPTYPE>(GLM_results,opts);
        
        //runs the log likelihood computation. After, the results will be in the matlab arrays as results holds the pointers to those arrays.
        glm_obj->computeLogLikelihood(params,opts,results);
        
        delete params;
        delete opts;
        delete results;
    }
    
    
//main func
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        if(inputs.size() != 4  && inputs.size() != 5) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("mex function requires 4-5 arguments!") }));
        }
    
        //checks that arg 0 is a pointer value to the GLM C++ object
        const matlab::data::TypedArray<uint64_t> gpu_ptr_a = inputs[0];
        if(gpu_ptr_a.getNumberOfElements() != 1) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("pointer argument must only contain 1 element!") }));
        }
        
        //get datatype field from GLM object
        const matlab::data::TypedArray<bool> isDouble = inputs[1];
        if(isDouble.getNumberOfElements() != 1) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("cannot determine gpu data type!") }));
        }
        
        const matlab::data::StructArray GLM_params  = inputs[2];
        const matlab::data::StructArray GLM_results = inputs[3];
        
        
        if(gpu_ptr_a.getNumberOfElements() < 1 || gpu_ptr_a[0] == 0) {
            //no glm object to clear
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("gpu pointer is not initialized!") }));
        }
        else if(isDouble[0]) {
            std::vector<double> trial_weights;
            if(inputs.size() >= 5 && !inputs[4].isEmpty()) {
                const matlab::data::TypedArray<double> & trial_weights_mat =  inputs[4];
                trial_weights.resize(trial_weights_mat.getNumberOfElements());
                //MATLAB's API makes things really realy slow: I get around it via this sloppy forceful operation
                double* ww = const_cast<double*>((trial_weights_mat.cbegin()).operator->());
                for(int ii = 0; ii < trial_weights.size(); ii++) {
                    trial_weights[ii] = ww[ii];
                }
            }
            runComputation<double>(gpu_ptr_a[0], GLM_params, GLM_results, trial_weights);
        }
        else {
            std::vector<float> trial_weights;
            if(inputs.size() >= 5 && !inputs[4].isEmpty()) {
                const matlab::data::TypedArray<float> & trial_weights_mat =  inputs[4];
                trial_weights.resize(trial_weights_mat.getNumberOfElements());
                //MATLAB's API makes things really realy slow: I get around it via this sloppy forceful operation
                float* ww = const_cast<float*>((trial_weights_mat.cbegin()).operator->());
                for(int ii = 0; ii < trial_weights.size(); ii++) {
                    trial_weights[ii] = ww[ii];
                }
            }
            runComputation<float>(gpu_ptr_a[0], GLM_params, GLM_results, trial_weights);
        }
    }
};
