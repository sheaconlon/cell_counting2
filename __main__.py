# =====
# Import built-in libraries.
# =====
import os
import subprocess
import shutil

# ====
# Import from the package.
# ====
from cell_counting.utilities import prompt

# =====
# Get instructions from the user.
# =====


def make_strset_converter(valid):
    def strset_converter(resp):
        if resp in valid:
            return True, resp
        else:
            return False, "{0:s} is not one of the allowed responses " \
                "{1:s}".format(str(resp), str(valid))
    return strset_converter


def make_path_converter(must_exist):
    def path_converter(path):
        if not os.path.isdir(path):
            if must_exist:
                return False, "Directory {0:s} does not exist".format(path)
            else:
                os.makedirs(path, exist_ok=True)
        return True, path
    return path_converter


def make_int_converter(validator):
    def int_converter(resp):
        try:
            num = int(resp)
        except ValueError:
            return False, "{0:s} is not a valid int".format(resp)
        if not validator(num):
            return False, "{0:d} is not a valid choice (see " \
                          "instructions)".format(num)
        return True, num
    return int_converter


def side_length_validator(num):
    return num > 0 and num % 2 == 1


def make_float_converter(validator):
    def float_converter(resp):
        try:
            num = float(resp)
        except ValueError:
            return False, "{0:s} is not a valid float".format(resp)
        if not validator(num):
            return False, "{0:f} is not a valid choice (see " \
                          "instructions)".format(num)
        return True, num
    return float_converter


model_type = prompt("Do you want to use (D)CNNs or (S)VMs?",
                    make_strset_converter({'D', 'S'}))

mode = prompt("Do you want to do training, validation, and testing to generate "
              "a (N)ew model or do you want to use a (P)re-existing model "
              "that you have saved?",
              make_strset_converter({'N', 'P'}))

if mode == "N":
    save_dir = prompt("Provide a path to a directory in which to save "
                      "results and intermediate files.\nIt will be created if "
                      "nonexistent.", make_path_converter(False))
    os.makedirs(save_dir, exist_ok=True)
else:
    save_dir = prompt("Provide a path to a directory to load the model "
                      "from and save additional results and intermediate "
                      "files to.", make_path_converter(True))

if mode == "N":
    if model_type == "D":
        pass
        # image_mask_dir = prompt("Please enter a path to a directory "
        #                         "containing image-mask data. See README.md "
        #                         "for formatting details",
        #                         make_path_converter(True))
        # side_length = prompt("Please enter a side length (px) for the "
        #                           "patches that the DCNN will classify. This "
        #                           "is recommended to be the size of the "
        #                           "largest colony found in the entire dataset."
        #                           "It must be odd",
        #                           make_int_converter(side_length_validator))
        # max_patches = prompt("Please enter the maximum number of patches to "
        #                      "extract. These patches will be used for "
        #                      "training, validation, and testing. Must be at "
        #                      "least 100",
        #                      make_int_converter(lambda num: num >= 100))
        # num_augs = prompt("Please enter the number of augmented copies that "
        #                   "should be produced for each image. Must be at "
        #                   "least 1",
        #                   make_int_converter(lambda num: num >= 1))
        # num_scales = prompt("Please enter the number of scaled copies that "
        #                     "should be produced for each image. Must be at "
        #                     "least 1",
        #                     make_int_converter(lambda num: num >= 1))
        # metric_examples = prompt("Please enter the number of patches that "
        #                          "should be tested after each round of "
        #                          "training to assess accuracy",
        #                          make_int_converter(lambda num: num >= 1))
        # train_duration = prompt("Please enter the number of minutes that the "
        #                         "DCNN should be trained for. Note that "
        #                         "the time taken for periodic accuracy "
        #                         "assessments will not be included in this, "
        #                         "so the entire process might take "
        #                         "significantly longer",
        #                         make_int_converter(lambda num: num >= 1))
        # metric_interval = prompt("Please enter the number of minutes that "
        #                          "should elapse between the periodic metric "
        #                          "evaluations",
        #     make_int_converter(lambda num: 1 <= num < train_duration))
        # train_steps = prompt("Please enter the number of training steps to "
        #                      "schedule at once. 5 might be appropriate for a "
        #                      "personal computer. 50 might be appropriate for a "
        #                      "high-memory machine with a GPU",
        #                      make_int_converter(lambda num: num >= 1))
        # batch_size = prompt("Please enter the maximum number of patches that "
        #                     "should be held in memory at once. 400 might "
        #                     "be appropriate for a personal computer. 4000 "
        #                     "might be appropriate for a high-memory machine",
        #                     make_int_converter(lambda num: num >= 1))

    else:
        well_count_dir = prompt("Provide a path to a directory "
                                "containing pinned plate images and "
                                "their well counts.\nSee README.md for "
                                "details about the directory structure and "
                                "file formatting expectations.",
                                make_path_converter(True))
        num_augs = prompt("Provide the number of augmentations that should "
                          "be produced for each well image, or -1 for "
                          "no augmentation.\nMust be either -1 or at least "
                          "1. -1 is recommended.",
                          make_int_converter(lambda num: num == -1 or num >= 1))
        colony_size = prompt("Provide an estimated side length for the "
                             "colonies, in pixels.",
                           make_int_converter(lambda num: num >= 1))
        scale_factor = prompt("Provide a factor that all well images should be "
                              "scaled by.\n1 is recommended.",
                              make_float_converter(lambda num: num > 0))
        valid_p = prompt("Provide the proportion of wells that should be set "
                         "aside as validation data.\nMust produce at least 1 "
                         "well. 0.2 is recommended.",
                         make_float_converter(lambda num: num > 0))
        test_p = prompt("Provide the proportion of wells that should be set "
                         "aside as test data.\nMust produce at least 1 "
                         "well. 0.2 is recommended.",
                         make_float_converter(lambda num: num > 0))
        trials = prompt("Provide the number of hyperparameter settings "
                        "that should be trialed.\nEach trial will require "
                        "training a new SVM. 4 is recommended for a personal "
                        "machine and 32 is ideal. Must be at least 4.",
                        make_int_converter(lambda num: num >= 4))
        processes = prompt("Provide the number of SVMs that should be trained "
                           "concurrently in processes.\n1 is recommended for "
                           "a personal machine and 4 is recommended for a "
                           "high-memory machine. Must be at least 1.",
                           make_int_converter(lambda num: num >= 1))
        path = os.path.join(os.path.dirname(__file__), "manuscript",
                            "preprocess_pinned.py")
        subprocess.run(["python3.6", path,
                        "-outdir", os.path.join(save_dir,
                                                "preprocess_pinned"),
                        "-numaugs", str(num_augs),
                        "-patchsize", str(colony_size),
                        "-validp", str(valid_p),
                        "-testp", str(test_p),
                        "-sizefactor", str(scale_factor),
                        "-datapath", well_count_dir])
        path = os.path.join(os.path.dirname(__file__), "manuscript",
                            "train_validate_svm.py")
        subprocess.run(["python3.6", path,
                        "-pinned", os.path.join(save_dir,
                                                "preprocess_pinned"),
                        "-out", os.path.join(save_dir,
                                                "train_validate_svm"),
                        "-cvals", str(trials // 4),
                        "-processes", str(processes)])
        path = os.path.join(os.path.dirname(__file__), "manuscript",
                            "test_svm.py")
        subprocess.run(["python3.6", path,
                        "-pinned", os.path.join(save_dir,
                                                "preprocess_pinned"),
                        "-tvsvm", os.path.join(save_dir,
                                             "train_validate_svm"),
                        "-out", os.path.join(save_dir, "test_svm")])
else:
    if model_type == "S":
        well_count_dir = prompt("Provide a path to a directory "
                                "containing pinned plate images and "
                                "their well counts. The well counts can "
                                "be garbage, as they will not be used.\nSee "
                                "README.md for details about the directory "
                                "structure and file formatting expectations.",
                                make_path_converter(True))
        colony_size = prompt("Provide an estimated side length for the "
                             "colonies, in pixels.",
                           make_int_converter(lambda num: num >= 1))
        scale_factor = prompt("Provide a factor that all well images should be "
                              "scaled by.\n1 is recommended.",
                              make_float_converter(lambda num: num > 0))
        path = os.path.join(os.path.dirname(__file__), "manuscript",
                            "preprocess_pinned.py")
        shutil.rmtree(os.path.join(save_dir, "preprocess_pinned_tmp"),
                      ignore_errors=True)
        subprocess.run(["python3.6", path,
                        "-outdir", os.path.join(save_dir,
                                                "preprocess_pinned_tmp"),
                        "-numaugs", str(-1),
                        "-patchsize", str(colony_size),
                        "-validp", str(0),
                        "-testp", str(1),
                        "-sizefactor", str(scale_factor),
                        "-datapath", well_count_dir])
        path = os.path.join(os.path.dirname(__file__), "manuscript",
                            "test_svm.py")
        shutil.rmtree(os.path.join(save_dir, "test_svm_tmp"),
                      ignore_errors=True)
        subprocess.run(["python3.6", path,
                        "-pinned", os.path.join(save_dir,
                                                "preprocess_pinned_tmp"),
                        "-tvsvm", os.path.join(save_dir,
                                             "train_validate_svm"),
                        "-out", os.path.join(save_dir, "test_svm_tmp")])