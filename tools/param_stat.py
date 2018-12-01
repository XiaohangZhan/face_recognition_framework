import torch
import sys

def param_num(state):
    num = 0
    for key in state.keys():
        sz = tuple(state[key].size())
        nn = 1
        for s in sz:
            nn *= s
        print('{}: {}    {}'.format(key, sz, nn))
        num += nn
    print('total num: {}'.format(num))

if __name__ == "__main__":
    fn = sys.argv[1]
    state = torch.load(fn)['state_dict']
    param_num(state)
