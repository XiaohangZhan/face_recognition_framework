import numpy as np
import os

path = 'output/committee_ms1m/base/'
exps = ['base-nasnetasmall-lr0.1', 'base-irv2-lr0.1', 'base-densenet121-lr0.1', 'base-resnet101-lr0.1', 'base-resnet50-lr0.3']
output_exp = 'ensemble-top{}'.format(len(exps))
if not os.path.isdir(path + output_exp):
    os.makedirs(path + output_exp)

megaface_prob = np.concatenate([np.fromfile(path + exp + '/megaface_new-prob_last.bin', dtype=np.float32).reshape(-1,256) for exp in exps], axis=1)
megaface_distractor = np.concatenate([np.fromfile(path + exp + '/megaface_new-distractor_last.bin', dtype=np.float32).reshape(-1,256) for exp in exps], axis=1)
ijba = np.concatenate([np.fromfile(path + exp + '/ijba_last.bin', dtype=np.float32).reshape(-1,256) for exp in exps], axis=1)

megaface_prob.tofile(path + output_exp + '/megaface_new-prob.bin')
megaface_distractor.tofile(path + output_exp + '/megaface_new-distractor.bin')
ijba.tofile(path + output_exp + '/ijba.bin')
