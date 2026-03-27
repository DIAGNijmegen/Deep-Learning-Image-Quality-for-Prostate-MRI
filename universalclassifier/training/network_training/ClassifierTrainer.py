# Copied and edited from https://github.com/MIC-DKFZ/nnUNet/blob/8e0ad8ebe0b24165419d087d4451b4631f5b37f2/nnunet/training/network_training/nnUNetTrainer.py#L432

from collections import OrderedDict

from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.network_training.network_trainer import NetworkTrainer
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from universalclassifier.training.data_augmentation.data_augmentation import get_moreDA_augmentation
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    default_3D_augmentation_params

from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast

from sklearn.model_selection import KFold
from nnunet.training.learning_rate.poly_lr import poly_lr

from universalclassifier.network_architecture.i3d.i3dpt import I3D
from universalclassifier.training.dataloading.dataset_loading import load_dataset
from universalclassifier.training.dataloading.data_loading import DataLoader3D

from multiprocessing import Pool
from nnunet.configuration import default_num_threads

from typing import Tuple, List
from nnunet.utilities.random_stuff import no_op

from universalclassifier.inference.export import save_output

import universalclassifier


class ClassifierTrainer(NetworkTrainer):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, stage=None, unpack_data=True,
                 deterministic=True, fp16=False):
        super().__init__(deterministic, fp16)
        self.unpack_data = unpack_data
        self.init_args = (plans_file, fold, output_folder, dataset_directory, unpack_data,
                          deterministic, fp16)
        # set through arguments from init
        self.stage = stage
        self.experiment_name = self.__class__.__name__
        self.plans_file = plans_file
        self.output_folder = output_folder
        self.dataset_directory = dataset_directory
        self.output_folder_base = self.output_folder
        self.fold = fold

        self.plans = None

        # if we are running inference only then the self.dataset_directory is set (due to checkpoint loading) but it
        # irrelevant
        if self.dataset_directory is not None and isdir(self.dataset_directory):
            self.gt_niftis_folder = join(self.dataset_directory, "gt_segmentations")
        else:
            self.gt_niftis_folder = None

        self.folder_with_preprocessed_data = None

        # set in self.initialize()

        self.dl_tr = self.dl_val = None
        self.num_input_channels = self.num_classes = self.num_classification_classes = self.batch_size = self.threeD = \
            self.intensity_properties = self.normalization_schemes = self.image_size = None  # loaded automatically from plans_file
        self.data_aug_params = self.transpose_forward = self.transpose_backward = None

        self.loss = CrossEntropyLoss()

        self.classes = self.do_dummy_2D_aug = self.use_mask_for_norm = None

        self.update_fold(fold)

        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 30
        self.weight_decay = 3e-5

        self.max_num_epochs = 100  # 1000
        self.initial_lr = 1e-2

        self.pin_memory = True

    def update_fold(self, fold):
        """
        used to swap between folds for inference (ensemble of models from cross-validation)
        DO NOT USE DURING TRAINING AS THIS WILL NOT UPDATE THE DATASET SPLIT AND THE DATA AUGMENTATION GENERATORS
        :param fold:
        :return:
        """
        if fold is not None:
            if isinstance(fold, str):
                assert fold == "all", "if self.fold is a string then it must be \'all\'"
                if self.output_folder.endswith("%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "%s" % str(fold))
            else:
                if self.output_folder.endswith("fold_%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "fold_%s" % str(fold))
            self.fold = fold

    def initialize(self, training=True, force_load_plans=False):
        """
        create self.output_folder
        modify self.output_folder if you are doing cross-validation (one folder per fold)
        set self.tr_gen and self.val_gen
        call self.initialize_network and self.initialize_optimizer_and_scheduler (important!)
        finally set self.was_initialized to True
        :param training:
        :return:
        """
        maybe_mkdir_p(self.output_folder)

        if force_load_plans or (self.plans is None):
            self.load_plans_file()

        self.process_plans(self.plans)
        self.setup_DA_params()

        self.loss = MultipleOutputLoss2(self.loss, weight_factors=None)

        if training:
            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)

            self.dl_tr, self.dl_val = self.get_basic_generators()
            if self.unpack_data:
                self.print_to_log_file("unpacking dataset")
                unpack_dataset(self.folder_with_preprocessed_data)
                self.print_to_log_file("done")
            else:
                self.print_to_log_file(
                    "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                    "will wait all winter for your model to finish!")

            self.tr_gen, self.val_gen = get_moreDA_augmentation(
                self.dl_tr, self.dl_val,
                self.data_aug_params[
                    'image_size_for_spatialtransform'],
                self.data_aug_params,
                pin_memory=self.pin_memory
            )
            self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                   also_print_to_console=False)
            self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                   also_print_to_console=False)
        else:
            pass

        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        self.was_initialized = True

    def process_plans(self, plans):
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.batch_size = stage_plans['batch_size']
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']

        self.intensity_properties = plans['dataset_properties']['intensityproperties']
        self.normalization_schemes = plans['normalization_schemes']
        self.num_input_channels = plans['num_modalities'] + 1  # +1 for roi/segmentation channel
        self.num_classes = plans['num_classes'] + 1  # classes in roi/segmentation map
        self.classes = plans['all_classes']  # classes in roi/segmentation map
        self.num_classification_classes = plans['num_classification_classes']
        self.use_mask_for_norm = plans['use_mask_for_norm']

        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                  "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                  "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']

        self.image_size = stage_plans['image_size']  ## added

        if len(self.image_size) == 2:
            self.threeD = False
        elif len(self.image_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid image size in plans file: %s" % str(self.patch_size))

    def setup_DA_params(self):
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.image_size) / min(self.image_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['image_size_for_spatialtransform'] = self.image_size

        # added:
        self.data_aug_params['num_seg_classes'] = self.num_classes

    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data, self.dataset_directory, "training")

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.image_size, self.batch_size, memmap_mode='r')
            dl_val = DataLoader3D(self.dataset_val, self.image_size, self.batch_size, memmap_mode='r')
        else:
            raise RuntimeError("2D dataloader not implemented.")
        return dl_tr, dl_val

    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans = load_pickle(self.plans_file)

    def initialize_network(self):
        if not self.threeD:
            raise RuntimeError("2D network not implemented")

        self.network = I3D(self.num_input_channels, self.num_classification_classes)

        if torch.cuda.is_available():
            self.network.cuda()

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def validate(self, overwrite: bool = True, validation_folder_name: str = 'validation_raw'):
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            if self.folder_with_preprocessed_data is None:
                self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                          "_stage%d" % self.stage)
            self.load_dataset()
            self.do_split()

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'overwrite': overwrite, 'validation_folder_name': validation_folder_name}
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        # save predictions if needed
        for k in self.dataset_val.keys():
            save_fname = join(output_folder, k + ".npz")
            if overwrite or not isfile(save_fname):
                item = np.load(self.dataset[k]['data_file'])
                data = item['data']

                data = self.rescale_segmentation_channel(data)

                result = {}
                result['pred'], result['logits'] = self.predict_preprocessed_data_return_pred_and_logits(data[None],
                                                                                                         self.fp16)
                result['out'] = [softmax(x) for x in result['logits']]
                result.update(self.dataset[k])
                np.savez(save_fname, **result)

        # load predictions
        results = []
        for k in self.dataset_val.keys():
            save_fname = join(output_folder, k + ".npz")
            result = np.load(save_fname, allow_pickle=True)
            results += [result]
        # convert from list of dicts to dict of np arrays:
        results = {k: np.asarray([dic[k] for dic in results]) for k in results[0]}
        results['classification_labels'] = results['classification_labels'][0]

        # compute performance and save it
        for label_it, label in enumerate(results['classification_labels']):  # for each label
            values = results['classification_labels'][label_it]['values']
            assert set(values.keys()) == set(str(it) for it in range(len(values)))
            for value in range(len(values)):  # for each value that label can have
                if len(values) == 2 and value == 0:
                    continue  # no need to compute the performance twice for binary labels
                # compute metrics
                task_targets = results['target'][:, label_it] == value
                task_preds = results['pred'][:, label_it] == value
                assert results['out'].shape[2] == 1
                task_outs = [o[value] for o in results['out'][:, label_it, 0]]

                results["acc"] = accuracy_score(task_targets, task_preds)
                results["auc"] = roc_auc_score(task_targets, task_outs)
                results["roc_curve"] = {k: v for k, v in zip(["fpr", "tpr", "thresholds"], roc_curve(task_targets, task_outs))}
                # print metrics
                title = f"{label['name']}: {values[str(value)]}"
                self.print_to_log_file(f"Metrics for {title}")
                for metric in ["acc", "auc"]:
                    self.print_to_log_file(f"\t{metric}: {results[metric]}")
                results_file = join(output_folder, "results.npz")
                np.savez(results_file, **results)
                print(f"Saved predictions and performance in {results_file}")

        self.network.train(current_mode)

    def predict_preprocessed_data_return_pred_and_logits(self, data: np.ndarray,
                                                         mixed_precision: bool) -> Tuple[List, List]:
        valid = list((I3D,))
        assert isinstance(self.network, tuple(valid))

        current_mode = self.network.training
        self.network.eval()

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        data = maybe_to_torch(data)
        if torch.cuda.is_available():
            data = to_cuda(data)

        with context():
            with torch.no_grad():
                ret = [output.cpu().numpy() for output in self.network(data)]
        self.network.train(current_mode)
        return [np.argmax(x) for x in ret], ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability
        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)


        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1
        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)
        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # TODO: maybe add this back in?, but then we would need another threshold since BCE is not going to be 0
        """ 
        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.reinitialize()
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        """
        return continue_training

    def save_debug_information(self):
        # saving some debug information
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct['plans']
        del dct['intensity_properties']
        del dct['dataset']
        del dct['dataset_tr']
        del dct['dataset_val']
        save_json(dct, join(self.output_folder, "debug.json"))

        import nnunet.utilities.shutil_sol as shutil_sol
        shutil_sol.copyfile(self.plans_file, join(self.output_folder_base, "plans.pkl"))

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr
        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        self.save_debug_information()

        super(ClassifierTrainer, self).run_training()

    def plot_network_architecture(self):
        self.print_to_log_file(type(self.network))

    def save_checkpoint(self, fname, save_optimizer=True):
        super(ClassifierTrainer, self).save_checkpoint(fname, save_optimizer)
        info = OrderedDict()
        info['init'] = self.init_args
        info['name'] = self.__class__.__name__
        info['class'] = str(self.__class__)
        info['plans'] = self.plans

        write_pickle(info, fname + ".pkl")

    def run_online_evaluation(self, output, target):
        pass  # TODO: implement this function (optional)

    def finish_online_evaluation(self):
        pass  # TODO: implement this function (optional)

    def preprocess_patient(self, input_files, seg_file):
        """
                Used to predict new unseen data. Not used for the preprocessing of the training/test data
                :param input_files:
                :param seg_file:
                :return:
                """
        from nnunet.training.model_restore import recursive_find_python_class
        preprocessor_name = self.plans.get('preprocessor_name')
        if preprocessor_name is None:
            if self.threeD:
                preprocessor_name = "UniversalClassifierPreprocessor"
            else:
                raise RuntimeError("2D preprocessor not implemented")

        self.print_to_log_file("using preprocessor", preprocessor_name)
        preprocessor_class = recursive_find_python_class([join(universalclassifier.__path__[0], "preprocessing")],
                                                         preprocessor_name,
                                                         current_module="universalclassifier.preprocessing")

        assert preprocessor_class is not None, "Could not find preprocessor %s in nnunet.preprocessing" % \
                                               preprocessor_name
        preprocessor = preprocessor_class(self.normalization_schemes, self.use_mask_for_norm,
                                          self.transpose_forward, self.intensity_properties)

        d, s, properties = preprocessor.preprocess_test_case(input_files,
                                                             self.plans['plans_per_stage'][self.stage][
                                                                 'current_spacing'],
                                                             self.plans['plans_per_stage'][self.stage][
                                                                 'image_size'],
                                                             seg_file)
        return d, s, properties

    def preprocess_predict_nifti(self, input_files: List[str], seg_file: str, output_file: str = None) -> None:
        """
        Use this to predict new data
        :param input_files:
        :param seg_file:
        :param output_file:
        :return:
        """
        self.print_to_log_file(f"Processing {input_files, seg_file}:")
        self.print_to_log_file("preprocessing...")
        d, s, properties = self.preprocess_patient(input_files, seg_file)
        data = self.combine_data_and_seg(d, s)
        self.print_to_log_file("predicting...")
        categorical_output, pred = self.predict_preprocessed_data_return_pred_and_logits(data[None], self.fp16)[
            1]  # generates logits output
        self.print_to_log_file("exporting prediction...")
        save_output(categorical_output, pred, output_file, properties)
        self.print_to_log_file("done")

    def combine_data_and_seg(self, data, seg):
        data = np.vstack((data, seg)).astype(np.float32)
        data = self.rescale_segmentation_channel(data)
        return data

    def rescale_segmentation_channel(self, data):
        data[-1][data[-1] == -1] = 0
        data[-1] = data[-1] / self.num_classes  # this is number of classes in segmentation omap
        return data
