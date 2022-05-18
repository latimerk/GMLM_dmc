/*
 * kcGMLMPython_glm.hpp
 * Structures for linking my C++/CUDA GLM code to Python via pybind11.
 *   
 *  References
 *   Kenneth Latimer & David Freeedman (2021). Low-dimensional encoding of 
 *   decisions in parietal cortex reflects long-term training history.
 *   bioRxiv
 *
 *  Copyright (c) 2022 Kenneth Latimer
 *
 *   This software is distributed under the GNU General Public
 *   License (version 3 or later); please refer to the file
 *   License.txt, included with the software, for details.
 */
#ifndef GMLM_PYTHON_GLM_H
#define GMLM_PYTHON_GLM_H

#include "kcSharedPython.hpp"
#include "kcGLM.hpp"

namespace kCUDA { 



// handlers for parameters
template <class FPTYPE>
class GPUGLM_params_python : public GPUGLM_params<FPTYPE> {
    public:
        GPUGLM_params_python() {
            numCovariates = 0;
            this->K = NULL;
        }
        GPUGLM_params_python(const unsigned int numCovariates_) {
            if(numCovariates_ < 1) {
                throw py::value_error("Number of covariates must be positive.");
            }
            numCovariates = numCovariates_;
            K_numpy = py::array_t<FPTYPE, py::array::f_style>(numCovariates);
            this->K = new GLData_numpy<FPTYPE>(K_numpy);
        }
        ~GPUGLM_params_python() {
            //py::print("params destructor");
            delete this->K;
        }

        void setK(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> K_new) {
            if(K_new.size() != numCovariates || K_new.strides(0)/sizeof(FPTYPE) != 1) {
                throw py::value_error("Parameter vector invalid! Must be a vector of length dim_K.");
            }

            K_numpy = K_new;
            delete this->K;
            this->K = new GLData_numpy<FPTYPE>(K_numpy);
        }
    private:
        py::array_t<FPTYPE, py::array::f_style> K_numpy;
        unsigned int numCovariates = 0;
};

// holds results in numpy + local form
template <class FPTYPE>
class GPUGLM_results_python : public GPUGLM_results<FPTYPE> {
    public:
        GPUGLM_results_python() {
            numCovariates = 0;
            this->dK = NULL;
            this->d2K = NULL;
            this->trialLL = NULL;
            numTrials     = 0;
        }
        GPUGLM_results_python(const unsigned int numCovariates_) {
            if(numCovariates_ < 1) {
                throw py::value_error("Number of covariates must be positive.");
            }
            numCovariates = numCovariates_;
            dK_numpy  = py::array_t<FPTYPE, py::array::f_style>(numCovariates);
            this->dK  = new GLData_numpy<FPTYPE>(dK_numpy);
            d2K_numpy = py::array_t<FPTYPE, py::array::f_style>({numCovariates, numCovariates});
            this->d2K = new GLData_numpy<FPTYPE>(d2K_numpy);

            numTrials     = 0;
            this->trialLL = NULL;
        }
        ~GPUGLM_results_python() {
            //std::ostringstream output_stream;
            //py::print("results destructor ");

            delete this->dK;
            delete this->d2K;
            delete this->trialLL;
            this->dK = NULL;
            this->d2K = NULL;
            this->trialLL = NULL;
        }

        void setupResults(unsigned int numTrials_) {
            if(numTrials_ < 1) {
                throw py::value_error("Need at least one trial!");
            }
            if(numTrials != numTrials_) {
                delete this->trialLL;
                numTrials = numTrials_;
                trialLL_numpy = py::array_t<FPTYPE, py::array::f_style>(numTrials);
                this->trialLL = new GLData_numpy<FPTYPE>(trialLL_numpy);
            }
        }

        py::array_t<FPTYPE, py::array::f_style> getTrialLL() {
            return trialLL_numpy;
        }
        py::array_t<FPTYPE, py::array::f_style> getDK() {
            return dK_numpy;
        }
        py::array_t<FPTYPE, py::array::f_style> getD2K() {
            return d2K_numpy;
        }
    private:
        unsigned int numTrials     = 0;
        unsigned int numCovariates = 0;
        py::array_t<FPTYPE, py::array::f_style> dK_numpy;
        py::array_t<FPTYPE, py::array::f_style> d2K_numpy;
        py::array_t<FPTYPE, py::array::f_style> trialLL_numpy;
};

// options to send to kcGLM
template <class FPTYPE>
class GPUGLM_computeOptions_python : public GPUGLM_computeOptions<FPTYPE> {
    public:
        GPUGLM_computeOptions_python() {
            reset();
        }
        void reset() {
            this->compute_trialLL = false;
            this->compute_dK      = false;
            this->compute_d2K     = false;
            this->trial_weights.clear(); // make sure no trial weights
        }
};

// holds a trial
template <class FPTYPE>
class kcGLM_trial : public GPUGLM_trial_args<FPTYPE> {
    public:
        kcGLM_trial(unsigned int trialNum, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> X_, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> Y_) {
            if(X_.shape(0) == Y_.size() && X_.ndim() == 2 && Y_.strides(0)/sizeof(FPTYPE) == 1 && Y_.ndim() <= 2) {
                X_numpy = X_;
                Y_numpy = Y_;
                this->trial_idx = trialNum;
                this->X = new GLData_numpy<FPTYPE>(X_numpy);
                this->Y = new GLData_numpy<FPTYPE>(Y_numpy);
            }
            else {
                throw py::value_error("Trial X and Y inconistent size!");
            }
        }
        void print() {
            std::ostringstream output_stream;
            output_stream << "X ";
            for(int ii = 0; ii < this->X->getSize(0) && ii < 5; ii++) {
                if(ii > 0) {
                    output_stream << ", ";
                }
                FPTYPE dd = (*(this->X))[ii];
                output_stream << dd;
            }
            output_stream << "\n";
            py::print(output_stream.str());
        }
        ~kcGLM_trial() {
            //py::print("trial destructor");
            delete this->X;
            delete this->Y;
        };

    private:
        py::array_t<FPTYPE, py::array::f_style> X_numpy;
        py::array_t<FPTYPE, py::array::f_style> Y_numpy;

};

//holds a block of trials
template <class FPTYPE>
class kcGLM_trialBlock : public GPUGLM_GPU_block_args<FPTYPE> {
    public:
        kcGLM_trialBlock(int dev_num_, int max_trials_for_sparse_run_) {
            this->dev_num = dev_num_;
            this->max_trials_for_sparse_run = max_trials_for_sparse_run_;
        }
        kcGLM_trialBlock(int dev_num_) : kcGLM_trialBlock(dev_num_, 32) { }

        int addTrial(std::shared_ptr<kcGLM_trial<FPTYPE>> trial) {
            trials_shared.push_back(trial);
            this->trials.push_back(trial.get()); // gross, but my simple API needed the raw pointers and I don't want to change that 
            return this->trials.size()-1; // trial number within block
        }

        ~kcGLM_trialBlock() {
            //py::print("trial block destructor");
            trials_shared.clear(); // don't think this is necessary, but still checking
        }
    private:
        // Do I need to keep a copy of the trials here so ptrs aren't deleted by garbage collection?
        std::vector<std::shared_ptr<kcGLM_trial<FPTYPE>>> trials_shared;
};


// the main interface class
template <class FPTYPE>
class kcGLM_python {
    public:
        kcGLM_python(unsigned int numCovariates, logLikeType ll_type, FPTYPE binSize);
        bool isOnGPU();
        void freeGPU();
        int addBlock(std::shared_ptr<kcGLM_trialBlock<FPTYPE>> block);
        void toGPU();
        // compute log likelihood: returns trial-wise LL
        py::array_t<FPTYPE, py::array::f_style> computeLogLikelihood(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> K);
        // compute log likelihood gradient
        py::array_t<FPTYPE, py::array::f_style> computeLogLikelihood_grad(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> K);
        // compute log likelihood hessian
        py::array_t<FPTYPE, py::array::f_style> computeLogLikelihood_hess(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> K);
        
        // return log likelihood with last given beta
        /* py::array_t<FPTYPE, py::array::f_style> computeLogLikelihood() {

        }*/

        ~kcGLM_python();
    private:
        GPUGLM<FPTYPE> * kcglm = NULL;
        GPUGLM_structure_args<FPTYPE> * structure = NULL;
        std::vector<GPUGLM_GPU_block_args<FPTYPE> *> blocks;
        std::vector<std::shared_ptr<kcGLM_trialBlock<FPTYPE>>> blocks_shared;
        
        std::shared_ptr<GPUGL_msg_python> msg;

        GPUGLM_params_python<FPTYPE> * params = NULL;
        GPUGLM_computeOptions_python<FPTYPE> * opts = NULL;
        GPUGLM_results_python<FPTYPE> * results = NULL;


        // calls the log likelihood function with parameters given the current opts setup
        void runComputation(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> K);

}; //GLM class



}; //namespace

#endif