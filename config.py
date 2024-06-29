def set_config(args):


    args.gpu_mem = 4 # Gbyte (adjust this as needed)
    args.dataset_path = r''#'/path/to/data/'  # for datasets

    args.output_path =  r'/log' #'/path/to/outputs/' # for logs, weights, etc.
    
    args.archi = 'resnet9'
    args.dataset_id_to_name = {0: 'cifar_10', 1: 'fashion-mnist'}

    args.diri = False
    args.share = False
    args.only = False
    args.isFed = False

    if 'only' in args.task:
        args.only = True

    if 'diri' in args.task:
        args.diri = True
        args.alpha = float(args.task.split('-')[5][1:])
        args.fac = 0.1

    if 'share'in args.task:
        args.share = True
        args.share_rate = float(args.task.split('-')[6][5:])

    # scenarios
    if 'lc' in args.task:
        args.scenario = 'labels-at-client'
        args.num_labels_per_class = 5
        args.num_epochs_client = 1 
        args.batch_size_client = 10 # for labeled set
        args.num_epochs_server = 0
        args.batch_size_server = 0
        args.num_epochs_server_pretrain = 0
        args.lr_factor = 3
        args.lr_patience = 5
        args.lr_min = 1e-20
    elif 'ls' in args.task:
        args.scenario = 'labels-at-server'
        args.num_labels_per_class = 100
        if 'lab' in args.task:
            args.num_labels_per_class = int(args.task.split('-')[4][3:])
        args.num_epochs_client = 1 
        args.batch_size_client = 100
        args.batch_size_server = 100
        args.num_epochs_server = 1
        args.num_epochs_server_pretrain = 1
        args.lr_factor = 3
        args.lr_patience = 20
        args.lr_min = 1e-20

    # tasks
    if 'biid' in args.task or 'bimb'in args.task: 
        args.sync = False
        args.num_tasks = 1
        args.num_clients = 100
        if 'ls' in args.task:
            args.num_rounds = 150
        else:
            args.num_rounds = 1000
    
    # datasets
    if 'c10' in args.task:
        args.dataset_id = 0
        args.num_classes = 10
        args.num_test = 2000
        args.num_valid = 2000
        args.batch_size_test = 100
        args.dataset_path = r''

    if 'mnist' in args.task:
        #args.frac_clients = 0.4
        args.dataset_id = 1
        args.num_classes = 10
        args.num_test = 2000
        args.num_valid = 2000
        args.batch_size_test = 100
        args.dataset_path = r''

    # base networks
    if args.archi in ['resnet9']:
        args.lr = 1e-3
        args.wd = 1e-4

    # hyper-parameters
    if args.model in ['fedmatch']:
        args.num_helpers = 2
        args.confidence = 0.75
        args.psi_factor = 0.2
        args.h_interval = 10

        if args.scenario == 'labels-at-client':
            args.lambda_s = 10 # supervised learning
            args.lambda_i = 1e-2 # inter-client consistency
            args.lambda_a = 1e-2 # agreement-based pseudo labeling
            args.lambda_l2 = 10
            args.lambda_l1 = 1e-4
            args.l1_thres = 1e-6 * 5
            args.delta_thres = 1e-5 * 5
                
        elif args.scenario == 'labels-at-server':
            args.lambda_s = 10 # supervised learning
            args.lambda_i = 1e-2 # inter-client consistency
            args.lambda_a = 1e-2 # agreement-based pseudo labeling
            args.lambda_l2 = 10
            args.lambda_l1 = 1e-5
            args.l1_thres = 1e-5
            args.delta_thres = 1e-5 

    return args


