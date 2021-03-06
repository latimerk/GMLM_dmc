/*
 * kcGMLM_mex_computeLL_gather.cu. Gathers results of LL computation from an async call.
 * Mex function to call the log likelihood function in a GMLM object.
 *  Takes 3 arguments: ptr (in long int form) to GMLM object
 *                       boolean (is double) for if the object is double precision (or single)
 *                       results struct (from GMLM class) - empty results 
 *                                matrices mean do not compute certain parts (like derivatives)
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
  




//creates a results object to send into the GMLM computations. It takes in a matlab struct and uses the results fields given in that struct.
//This requires that the matlab matrices match the precision of the GMLM -> it will not cast from double to single!
//This creates no new space: it gets the matlab pointers and does a pass-by-values assignment.
//This bypasses the MATLAB C++ API in mata::data for assignments, because going through there can cause significant overhead.
//(not safe to pass back to matlab from reuse: if those pointers are cleared by matlab, seg faults could happen!)
template <class FPTYPE>
class matlab_GPUGMLM_results : public kCUDA::GPUGMLM_results<FPTYPE> {
private:
    
public:
    matlab_GPUGMLM_results(const matlab::data::StructArray & GMLM_results,  std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr) {
        //assigns the pointers to the dW and d2W results 
        if(!(GMLM_results[0]["dW"].isEmpty())) {
            const matlab::data::TypedArray<FPTYPE> dW_mat = GMLM_results[0]["dW"];
            this->dW =  new GLData_matlab<FPTYPE>(dW_mat);
        }
        else{
            this->dW =  new GLData_matlab<FPTYPE>();
        }
        
        if(!(GMLM_results[0]["dB"].isEmpty())) {
            const matlab::data::TypedArray<FPTYPE> dB_mat = GMLM_results[0]["dB"];
            this->dB = new GLData_matlab<FPTYPE>(dB_mat);
        }
        else {
            this->dB = new GLData_matlab<FPTYPE>();
        }
        
        //assigns the pointers to the trial-wise log likelihood results 
        if(!(GMLM_results[0]["trialLL"].isEmpty())) {
            const matlab::data::TypedArray<FPTYPE> trialLL_mat = GMLM_results[0]["trialLL"];
            this->trialLL =  new GLData_matlab<FPTYPE>(trialLL_mat);
        }
        else {
            this->trialLL =  new GLData_matlab<FPTYPE>();
        }
        this->isSimultaneousPopulation = this->trialLL->getSize(1) > 1; // if results are expecting multiple neurons per trial
        
        //gets the tensor groups
        const matlab::data::StructArray Groups_mat = GMLM_results[0]["Groups"];
        size_t dim_J  = Groups_mat.getNumberOfElements();
        
        this->Groups.resize(dim_J);
        for(int jj = 0; jj < dim_J; jj++) {
            //creates a groups object
            this->Groups[jj] = new kCUDA::GPUGMLM_group_results<FPTYPE>;
            
            //assigns the pointers to the dV results for the group
            if(!(Groups_mat[jj]["dV"].isEmpty())) {
                const matlab::data::TypedArray<FPTYPE> dV_mat = Groups_mat[jj]["dV"];
                this->Groups[jj]->dV  = new GLData_matlab<FPTYPE>(dV_mat);
            }
            else {
                this->Groups[jj]->dV  = new GLData_matlab<FPTYPE>();
            }

            //looks at the results for the derivatives of the components in the remaining dimensions
            const matlab::data::CellArray dT_mats  = Groups_mat[jj]["dT"];
            size_t dim_S = dT_mats.getNumberOfElements();
            this->Groups[jj]->dT.resize(dim_S);
            for(int ss = 0; ss < dim_S; ss++) {
                //for each dimension, gets the pointers to the dT results
                
                //checks for a dT matrix
                if(!(dT_mats[ss].isEmpty())) {
                    const matlab::data::TypedArray<FPTYPE> dT_mat  = dT_mats[ss];
                    this->Groups[jj]->dT[ss]  = new GLData_matlab<FPTYPE>(dT_mat);
                }
                else {
                    this->Groups[jj]->dT[ss]  = new GLData_matlab<FPTYPE>();
                }
            }
        }
    }
    //destructor
    ~matlab_GPUGMLM_results() {
        //clears all the groups
        delete this->dW;
        delete this->dB;
        delete this->trialLL;
        for(int jj = 0; jj < this->Groups.size(); jj++) {
            delete this->Groups[jj]->dV;
            for(int ss = 0; ss < this->Groups[jj]->dT.size(); ss++) {
                delete this->Groups[jj]->dT[ss];
            }
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
    void runComputation(const uint64_t GMLM_ptr, const matlab::data::StructArray & GMLM_results) {
        if(GMLM_ptr == 0) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("gpu pointer is not initialized!") }));
            return;
        }
        
        kCUDA::GPUGMLM<FPTYPE>         * gmlm_obj = reinterpret_cast<kCUDA::GPUGMLM<FPTYPE> *>(GMLM_ptr); //forcefully recasts the uint64 value
        //gets the model results
        kCUDA::GPUGMLM_results<FPTYPE> * results = new matlab_GPUGMLM_results<FPTYPE>(GMLM_results,  matlabPtr);
        
        //runs the log likelihood computation. After, the results will be in the matlab arrays as results holds the pointers to those arrays.
        gmlm_obj->computeLogLikelihood_gather(results, false);
        
        delete results;
    }
    
    
//main func
public:
    
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        if(inputs.size() != 3) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("mex function requires 3 arguments!") }));
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
        
        const matlab::data::StructArray GMLM_results = inputs[2];
        
        
        if(gpu_ptr_a.getNumberOfElements() < 1 || gpu_ptr_a[0] == 0) {
            //no gmlm object to clear
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("gpu pointer is not initialized!") }));
        }
        else if(isDouble[0]) {
            runComputation<double>(gpu_ptr_a[0], GMLM_results);
        }
        else {
            runComputation<float>(gpu_ptr_a[0],  GMLM_results);
        }
    }
};