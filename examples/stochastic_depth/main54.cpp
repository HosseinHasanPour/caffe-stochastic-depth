#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <map>
#include <assert.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/sgd_solvers.hpp"



using namespace caffe;
using namespace std;

int main(int argc, char** argv)
{
    Caffe::set_mode(Caffe::GPU);

    string param_file = "examples/stochastic_depth/solver54.prototxt";
    SolverParameter param;
    ReadSolverParamsFromTextFileOrDie(param_file, &param);
    Solver<float>* solver = SolverRegistry<float>::CreateSolver(param);
    shared_ptr<Net<float> > net = solver->net();
    vector<shared_ptr<Layer<float> > > layers = net->layers();


    vector<int>* layers_chosen = new vector<int>();
    net->ChooseLayers_StochDep(layers_chosen);

    cout << "params_lr: " << net->params_lr().size() << "\t";
    cout << "has_params_lr: " << net->has_params_lr().size()<< "\t";
    cout << "params_weight_decay: " << net->params_weight_decay().size()<< "\t";
    cout << "has_params_decay: " << net->has_params_decay().size() << endl;
//
//    for (int l = 0; l < net->params_lr().size(); l++) {
//        cout << net->params_lr()[l] << "\t";
//        cout << net->has_params_lr()[l] << "\t";
//        cout << net->params_weight_decay()[l] << "\t";
//        cout << net->has_params_decay()[l] << endl;
//    }

//	for (int i = 0; i < layers_chosen->size(); i++) {
//        int layer_id = (*layers_chosen)[i];
//        int mapvecsize = 0;
//        typedef typename map<int, vector<Blob<float>* >* >::const_iterator iter;
//        iter pair;
//        if (net->layer_num_to_learnable_params().count(layer_id) > 0) {
//            pair = net->layer_num_to_learnable_params().find(layer_id);
//            mapvecsize = (int)pair->second->size();
//        }
//		cout << (*layers_chosen)[i] << ": " << layers[layer_id]->type() << "\t" <<layers[layer_id]->blobs().size() << "\t mapvecsize: " << mapvecsize << endl;
//        for (int k = 0; k < layers[layer_id]->blobs().size(); k++){
//            cout << layers[layer_id]->blobs()[k] << " " << (*pair->second)[k] << "\t";
//        }
//        cout << endl;
//	}
//
    cout << "layers; " << net->layers().size() << endl;
    cout << "params: " << net->params().size() << endl;
//    cout << "learnable params: " << net->learnable_params().size() << endl;
//    for (int i = 0; j < net->learnable_params().size(); j++) {
//        cout << (*layers_chosen)[i] << ": " << layers[(*layers_chosen)[i]]->type() << "\t" <<layers[(*layers_chosen)[i]]->blobs().size() << endl;
//    }

    solver->Solve_StochDep();
}


//--------------------------------------- NET -------------------------------------------------------------------------



template <typename Dtype>
void Net<Dtype>::SetLearnableParams_StochDep(vector<int>* layers_chosen) {
    learnable_params_ids_stochdept_.resize(0);
    for (int i = 0; i < (*layers_chosen).size(); i++) {
        int layer_id = (*layers_chosen)[i];
        typedef typename map<int, vector<int>* >::const_iterator iter;
        iter pair;
        if (layer_num_to_learnable_params_.count(layer_id) > 0) {
            pair = layer_num_to_learnable_params_idxs.find(layer_id);
            vector<int> idx_vec =  *pair->second;
            for ( int j = 0; j < idx_vec.size() ; j++){
                learnable_params_ids_stochdept_.push_back(idx_vec[j]);
            }
        }
    }
}

template <typename Dtype>
void Net<Dtype>::standardResLayer(int & elts, int & idx, vector<int>* layers_chosen, double ran, double prob) {
    if (ran < prob){ // include res block
        for (int i = 0; i < 10; i++){
            layerHelper_StochDep(elts, idx, layers_chosen, 1, 1, 0, true);
        }
    }
    else{  // skip res block
        layerHelper_StochDep(elts, idx, layers_chosen, 10, 1, 10, false);
//        cout << "skipping standard block: " << elts << endl;
    }
}

template <typename Dtype>
void Net<Dtype>::transitionResLayer(int & elts, int& idx, vector<int>* layers_chosen, double ran, double prob){
    if (ran < prob) { //include res block
        for (int i = 0; i < 13; i++) {
            layerHelper_StochDep(elts, idx, layers_chosen, 1, 1, 0, true);
        }
    }
    else { // skip res block
        layerHelper_StochDep(elts, idx, layers_chosen, 2, 1, 2, false);
        layerHelper_StochDep(elts, idx, layers_chosen, 1, 1, 0, true);
        layerHelper_StochDep(elts, idx, layers_chosen, 1, 1, 0, true);
        layerHelper_StochDep(elts, idx, layers_chosen, 9, 1, 9, false);
//		cout << "skipping transition block: " << elts << endl;
    }
}

template <typename Dtype>
void Net<Dtype>::layerHelper_StochDep(int & elts, int& idx, vector<int>* layers_chosen, int elt_incr, int idx_incr, int bottom_incr, bool use_top) {
    bottom_vecs_stochdept_[idx] = bottom_vecs_[elts];

    if (use_top) {
        top_vecs_stochdept_[idx] = top_vecs_[elts + bottom_incr];
    }
    else {
        top_vecs_stochdept_[idx] = bottom_vecs_[elts + bottom_incr];
    }

    (*layers_chosen)[idx] = elts;

//    // prints
//
//    shared_ptr<Layer<Dtype> > curr_layer = layers_[elts];
//    vector<Blob<Dtype>*> og_bottom = bottom_vecs_[elts];
//    vector<Blob<Dtype>*> og_top = top_vecs_[elts];
//    cout << "og layer:\t" << curr_layer->type() << " " <<  elts << "\tbottom size: " << og_bottom.size() << "\ttop size: " << og_top.size() <<  endl;
//    vector<Blob<Dtype>*> curr_bottom = bottom_vecs_stochdept_[idx];
//    vector<Blob<Dtype>*> curr_top = top_vecs_stochdept_[idx];
//    cout << "my layer:\t" << curr_layer->type() << " " <<  elts << "\tbottom size: " << curr_bottom.size() << "\ttop size: " << curr_top.size() <<  endl;
//
//    // end prints

    elts += elt_incr;
    idx += idx_incr;
}


template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo_StochDep(vector<int>* layers_chosen) {
    Dtype loss = 0;
    int layer_idx;
    for (int i = 0; i < layers_chosen->size(); i++) {
        layer_idx = (*layers_chosen)[i];
        shared_ptr<Layer<Dtype> > curr_layer = layers_[layer_idx];

        vector<Blob<Dtype>*> curr_bottom = bottom_vecs_stochdept_[i];
        vector<Blob<Dtype>*> curr_top = top_vecs_stochdept_[i];

        Dtype layer_loss = curr_layer->Forward(curr_bottom, curr_top);
        loss += layer_loss;
        if (debug_info_) { ForwardDebugInfo(layer_idx); }
    }
    return loss;
}

template<typename Dtype>
void Net<Dtype>::printvecblobs(vector<vector<Blob<Dtype>*> > vec, int &idx) {
for (int i = 0; i < vec[idx].size(); i++) {
Blob<Dtype>* blo= vec[idx][i];
//cout << blo->shape(0) << " " << blo->shape(1) << " " << blo->shape(2) << " " <<  blo->shape(3)  << endl;
cout << blo << endl;
}
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo_StochDep(vector<int>* layers_chosen) {
    int layer_idx;
    for (int i = layers_chosen->size() - 1; i >= 0; i--) {
        layer_idx = (*layers_chosen)[i];
        if (layer_need_backward_[layer_idx]) {
            layers_[layer_idx]->Backward(top_vecs_stochdept_[i], bottom_need_backward_[layer_idx], bottom_vecs_stochdept_[i]);
            if (debug_info_) { BackwardDebugInfo(layer_idx); }
        }
    }
}

template<typename Dtype>
Dtype Net<Dtype>::ForwardBackward_StochDep(vector<int>* layers_chosen) {
    Dtype loss;
    Forward_StochDep(layers_chosen, &loss);
    Backward_StochDep(layers_chosen);
    return loss;
}

template<typename Dtype>
void Net<Dtype>::ChooseLayers_StochDep(vector<int>* layers_chosen){
    bottom_vecs_stochdept_.resize(this->layers().size());
    top_vecs_stochdept_.resize(this->layers().size());
    layers_chosen->resize(this->layers().size());
    int elts = 0;
    int idx = 0;
    for (int i = 0; i < 4; i++){
        layerHelper_StochDep(elts, idx, layers_chosen, 1, 1, 0, true);
    }

    srand((unsigned)time(NULL));
    double block_num  = 0;
    double prob;
    double ran;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            ran = (double) rand()/RAND_MAX;
            prob = 1 - 0.5*(block_num)/13;
            standardResLayer(elts, idx, layers_chosen, ran, prob);
            block_num += 1.0;
        }
        ran = (double) rand()/RAND_MAX;
        prob = 1 - 0.5*(block_num)/13;
        transitionResLayer(elts, idx, layers_chosen, ran, prob);
        block_num += 1.0;
    }
    for (int j = 0; j < 4; j++) {
        ran = (double) rand()/RAND_MAX;
        prob = 1 - 0.5*(block_num)/13;
        standardResLayer(elts, idx, layers_chosen, ran, prob);
        block_num += 1.0;
    }

    for (int i = 0; i < 4; i++) {
        layerHelper_StochDep(elts, idx, layers_chosen, 1, 1, 0, true);
    }
    bottom_vecs_stochdept_.resize(idx);
    top_vecs_stochdept_.resize(idx);
    layers_chosen->resize(idx);
}

template <typename Dtype>
void Net<Dtype>::Backward_StochDep( vector<int>* layers_chosen) {
    BackwardFromTo_StochDep(layers_chosen);
    if (debug_info_) {
        Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
        for (int i = 0; i < learnable_params_.size(); ++i) {
            asum_data += learnable_params_[i]->asum_data();
            asum_diff += learnable_params_[i]->asum_diff();
            sumsq_data += learnable_params_[i]->sumsq_data();
            sumsq_diff += learnable_params_[i]->sumsq_diff();
        }
        const Dtype l2norm_data = std::sqrt(sumsq_data);
        const Dtype l2norm_diff = std::sqrt(sumsq_diff);
        LOG(ERROR) << "    [Backward] All net params (data, diff): "
        << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
        << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
    }
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward_StochDep(vector<int>* layers_chosen, Dtype* loss) {
if (loss != NULL) {
*loss = ForwardFromTo_StochDep(layers_chosen);
} else {
ForwardFromTo_StochDep(layers_chosen);
}
return net_output_blobs_;
}


template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward_StochDep_Test(Dtype* loss) {
if (loss != NULL) {
*loss = ForwardFromTo_StochDep_Test(0, layers_.size() - 1);
} else {
ForwardFromTo_StochDep_Test(0, layers_.size() - 1);
}
return net_output_blobs_;
}


template <typename Dtype>
const Dtype Net<Dtype>::ForwardFromTo_StochDep_Test(int start, int end) {
    CHECK_GE(start, 0);
    CHECK_LT(end, layers_.size());
    Dtype loss = 0;
    for (int i = start; i <= end; ++i) {
//         LOG(ERROR) << "Forwarding " << layer_names_[i];
//    cout << layers_[i]->type() << i << "\t bottom size: " <<  bottom_vecs_[i].size() << endl;
        Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
        loss += layer_loss;
        vector<Blob<Dtype>*> top_vec = top_vecs_[i];
        double prob = test_scaling_stochdept_[i];
        if (prob < 1.0) {
            for (int j = 0; j < top_vec.size(); j++) {
                Blob<Dtype> *top_blob = top_vec[j];
//                cout <<"i: " << i << "\t j: " << j << '\t' << layers_[i]->type() << "\t prob: " << prob << endl;
                top_blob->scale_data(prob);
            }
        }
        if (debug_info_) { ForwardDebugInfo(i); }
    }
    return loss;
}





//----------------------------------------- SOLVER ---------------------------------------------------------------------

template <typename Dtype>
void Solver<Dtype>::Step_StochDep(int iters, vector<int>* layers_chosen) {
    const int start_iter = iter_;
    const int stop_iter = iter_ + iters;
    int average_loss = this->param_.average_loss();
    losses_.clear();
    smoothed_loss_ = 0;

    while (iter_ < stop_iter) {
        // zero-init the params
        net_->ClearParamDiffs();
        if (param_.test_interval() && iter_ % param_.test_interval() == 0
            && (iter_ > 0 || param_.test_initialization())
            && Caffe::root_solver()) {
            TestAll_StochDep();
            if (requested_early_exit_) {
                // Break out of the while loop because stop was requested while testing.
                break;
            }
        }

        for (int i = 0; i < callbacks_.size(); ++i) {
            callbacks_[i]->on_start();
        }
        const bool display = param_.display() && iter_ % param_.display() == 0;
        net_->set_debug_info(display && param_.debug_info());
        // accumulate the loss and gradient
        Dtype loss = 0;

        for (int i = 0; i < param_.iter_size(); ++i) {
            net_->ChooseLayers_StochDep(layers_chosen);
            net_->SetLearnableParams_StochDep(layers_chosen);
//            cout << "learnable_params_stochdep: " << net_->learnable_params_ids_stochdept().size() <<"\t layers chosen: " << (*layers_chosen).size() << endl;
            loss += net_->ForwardBackward_StochDep(layers_chosen);
        }
        loss /= param_.iter_size();
        // average the loss across iterations for smoothed reporting
        UpdateSmoothedLoss(loss, start_iter, average_loss);
        if (display) {
            LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
            << ", loss = " << smoothed_loss_;
            const vector<Blob<Dtype>*>& result = net_->output_blobs();
            int score_index = 0;
            for (int j = 0; j < result.size(); ++j) {
                const Dtype* result_vec = result[j]->cpu_data();
                const string& output_name =
                        net_->blob_names()[net_->output_blob_indices()[j]];
                const Dtype loss_weight =
                        net_->blob_loss_weights()[net_->output_blob_indices()[j]];
                for (int k = 0; k < result[j]->count(); ++k) {
                    ostringstream loss_msg_stream;
                    if (loss_weight) {
                        loss_msg_stream << " (* " << loss_weight
                        << " = " << loss_weight * result_vec[k] << " loss)";
                    }
                    LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
                    << score_index++ << ": " << output_name << " = "
                    << result_vec[k] << loss_msg_stream.str();
                }
            }
        }
        for (int i = 0; i < callbacks_.size(); ++i) {
            callbacks_[i]->on_gradients_ready();
        }
        ApplyUpdate_StochDep();

        // Increment the internal iter_ counter -- its value should always indicate
        // the number of times the weights have been updated.
        ++iter_;

        SolverAction::Enum request = GetRequestedAction();

        // Save a snapshot if needed.
        if ((param_.snapshot()
             && iter_ % param_.snapshot() == 0
             && Caffe::root_solver()) ||
            (request == SolverAction::SNAPSHOT)) {
            Snapshot();
        }
        if (SolverAction::STOP == request) {
            requested_early_exit_ = true;
            // Break out of training loop.
            break;
        }
    }
}


template <typename Dtype>
void Solver<Dtype>::Solve_StochDep(const char* resume_file) {
    CHECK(Caffe::root_solver());
    LOG(INFO) << "Solving " << net_->name();
    LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

    // Initialize to false every time we start solving.
    requested_early_exit_ = false;

    if (resume_file) {
        LOG(INFO) << "Restoring previous solver status from " << resume_file;
        Restore(resume_file);
    }

    // For a network that is trained by the solver, no bottom or top vecs
    // should be given, and we will just provide dummy vecs.
    int start_iter = iter_;
    vector<int>* layers_chosen = new vector<int>();
    Step_StochDep(param_.max_iter() - iter_, layers_chosen);
    // If we haven't already, save a snapshot after optimization, unless
    // overridden by setting snapshot_after_train := false
    if (param_.snapshot_after_train()
        && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
        Snapshot();
    }
    if (requested_early_exit_) {
        LOG(INFO) << "Optimization stopped early.";
        return;
    }
    // After the optimization is done, run an additional train and test pass to
    // display the train and test loss/outputs if appropriate (based on the
    // display and test_interval settings, respectively).  Unlike in the rest of
    // training, for the train net we only run a forward pass as we've already
    // updated the parameters "max_iter" times -- this final pass is only done to
    // display the loss, which is computed in the forward pass.
    if (param_.display() && iter_ % param_.display() == 0) {
        int average_loss = this->param_.average_loss();
        Dtype loss;
        net_->Forward_StochDep_Test(&loss);

        UpdateSmoothedLoss(loss, start_iter, average_loss);

        LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
    }
    if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
        TestAll_StochDep();
    }
    LOG(INFO) << "Optimization Done.";
}


template <typename Dtype>
void Solver<Dtype>::TestAll_StochDep() {
    for (int test_net_id = 0; test_net_id < test_nets_.size() && !requested_early_exit_; ++test_net_id) {
        Test_StochDep(test_net_id);
    }
}


template <typename Dtype>
void Solver<Dtype>::Test_StochDep(const int test_net_id) {
    CHECK(Caffe::root_solver());
    LOG(INFO) << "Iteration " << iter_
    << ", Testing net (#" << test_net_id << ")";
    CHECK_NOTNULL(test_nets_[test_net_id].get())->
            ShareTrainedLayersWith(net_.get());
    vector<Dtype> test_score;
    vector<int> test_score_output_id;
    const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
    Dtype loss = 0;
    for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
        SolverAction::Enum request = GetRequestedAction();
        // Check to see if stoppage of testing/training has been requested.
        while (request != SolverAction::NONE) {
            if (SolverAction::SNAPSHOT == request) {
                Snapshot();
            } else if (SolverAction::STOP == request) {
                requested_early_exit_ = true;
            }
            request = GetRequestedAction();
        }
        if (requested_early_exit_) {
            // break out of test loop.
            break;
        }

        Dtype iter_loss;
        const vector<Blob<Dtype>*>& result =
                                          test_net->Forward_StochDep_Test(&iter_loss);
        if (param_.test_compute_loss()) {
            loss += iter_loss;
        }
        if (i == 0) {
            for (int j = 0; j < result.size(); ++j) {
                const Dtype* result_vec = result[j]->cpu_data();
                for (int k = 0; k < result[j]->count(); ++k) {
                    test_score.push_back(result_vec[k]);
                    test_score_output_id.push_back(j);
                }
            }
        } else {
            int idx = 0;
            for (int j = 0; j < result.size(); ++j) {
                const Dtype* result_vec = result[j]->cpu_data();
                for (int k = 0; k < result[j]->count(); ++k) {
                    test_score[idx++] += result_vec[k];
                }
            }
        }
    }
    if (requested_early_exit_) {
        LOG(INFO)     << "Test interrupted.";
        return;
    }
    if (param_.test_compute_loss()) {
        loss /= param_.test_iter(test_net_id);
        LOG(INFO) << "Test loss: " << loss;
    }
    for (int i = 0; i < test_score.size(); ++i) {
        const int output_blob_index =
                test_net->output_blob_indices()[test_score_output_id[i]];
        const string& output_name = test_net->blob_names()[output_blob_index];
        const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
        ostringstream loss_msg_stream;
        const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
        if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
            << " = " << loss_weight * mean_score << " loss)";
        }
        LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
        << mean_score << loss_msg_stream.str();
    }
}



//------------------------------------------------ SGD SOLVER ----------------------------------------------------------


