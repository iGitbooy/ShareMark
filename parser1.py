import argparse

class Parser:
    
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()
       
    def set_arguments(self):

        self.parser.add_argument('--gpu', type=str, default='0', help='gpu ids to use e.g. 0,1,2,...')
        self.parser.add_argument('--work-type', type=str, default='train', help='work-types i.e. gen_data or train')
        self.parser.add_argument('--model', type=str, default='fedmatch', help='model i.e. fedmatch')

        self.parser.add_argument('--fedmethod', type=str, default='fedshare', help='federated aggregation method')

        self.parser.add_argument('--task', type=str, default='ls-biid-diri-c10-lab50-d1-share0.03-only-expert',
                                 help='task i.e. '
                                      'ls-biid-diri-c10-d1.0, '
                                      'ls-biid-diri-c10-lab50-d1.0-share0.01-only'
                                      )

        self.parser.add_argument('--frac-clients', default='0.05', type=float, help='fraction of clients per round')
        self.parser.add_argument('--seed', type=int, default='1', help='seed for experiment')
        
    def parse(self):
        args, unparsed  = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args