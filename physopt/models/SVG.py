import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import time
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import svg.utils as utils
import itertools
import progressbar
import numpy as np
import pickle
from physopt.utils import PhysOptObjective


def run(
        datasets = ['collide2_new'],
        seed = 0,
        data_root = '/mnt/fs4/mrowca/neurips/images/rigid',
        model_dir = '/mnt/fs4/mrowca/hyperopt/svg/default/0/model',
        feature_file = '/mnt/fs4/mrowca/hyperopt/svg/default/0/model/features/default/feat.pkl',
        model = 'vgg',
        freeze_encoder_weights = False,
        write_feat = '',
        max_run_time = 86400 * 100, # 100 days
        ):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--log_dir', default='logs/lp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--name', default='', help='identifier for directory')
    parser.add_argument('--data_root', default='/mnt/fs4/mrowca/neurips/images/rigid',
            help='root directory for data')
    parser.add_argument('--data_subsets', default=None, type=str, help='data subsets under root')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--epoch_size', type=int, default=3000, help='epoch size') #600, 3600
    parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--channels', default=3, type=int)
    parser.add_argument('--dataset', default='tdw', help='dataset to train with')
    parser.add_argument('--n_past', type=int, default=4, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=6, help='number of frames to predict during training')
    parser.add_argument('--n_eval', type=int, default=10, help='number of frames to predict during eval')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--model', default='vgg',
            help='model type (dcgan | vgg | vgg_pretrained | deit_pretrained)')
    parser.add_argument('--data_threads', type=int, default=0, help='number of data loading threads')
    parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--write_feat', default='', type=str, help='writes out features of train or test set')
    parser.add_argument('--freeze_encoder_weights', action='store_true', help='Freezes weights of ImageNet pretrained VGG')



    opt, unknown = parser.parse_known_args()
    print("Unkown arguments", unknown)

    # Set our params
    opt.data_subsets = ','.join(datasets)
    opt.data_root = data_root
    opt.seed = seed
    opt.model = model
    opt.freeze_encoder_weights = freeze_encoder_weights
    opt.write_feat = write_feat
    opt.batch_size = 2 if opt.write_feat else opt.batch_size
    opt.model_dir = model_dir
    opt.log_dir = os.path.join(opt.model_dir, 'logs')

    if opt.model in ['deit_pretrained', 'clip_pretrained']:
        opt.image_width = 224
        opt.batch_size = 2

    write_feat = opt.write_feat
    data_subsets = opt.data_subsets
    data_root = opt.data_root
    if os.path.exists('%s/model.pth' % opt.model_dir):
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % opt.model_dir)
        optimizer = opt.optimizer
        model_dir = opt.model_dir
        opt = saved_model['opt']
        opt.optimizer = optimizer
        opt.model_dir = model_dir
        opt.log_dir = '%s/continued' % opt.log_dir
        opt.write_feat = write_feat
        if write_feat:
            opt.data_subsets = data_subsets
            opt.data_root = data_root
            opt.n_future = 20 #45
            opt.n_eval = 24 #49
        load_model = True
    else:
        os.makedirs(opt.model_dir, exist_ok=True)
        load_model = False
        #name = 'model=%s%dx%d-rnn_size=%d-predictor-posterior-prior-rnn_layers=%d-%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f%s' % (opt.model, opt.image_width, opt.image_width, opt.rnn_size, opt.predictor_rnn_layers, opt.posterior_rnn_layers, opt.prior_rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim, opt.z_dim, opt.last_frame_skip, opt.beta, opt.name)
        #if opt.dataset == 'smmnist':
        #    opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
        #else:
        #    opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

    os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
    os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    dtype = torch.cuda.FloatTensor


# ---------------- load the models  ----------------

    print(opt)

# ---------------- optimizers ----------------
    if opt.optimizer == 'adam':
        opt.optimizer = optim.Adam
    elif opt.optimizer == 'rmsprop':
        opt.optimizer = optim.RMSprop
    elif opt.optimizer == 'sgd':
        opt.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % opt.optimizer)


    import svg_models.lstm as lstm_models
    if load_model:
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
        prior = saved_model['prior']
    else:
        frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
        posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
        prior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size)
        frame_predictor.apply(utils.init_weights)
        posterior.apply(utils.init_weights)
        prior.apply(utils.init_weights)

    if opt.model == 'dcgan':
        if opt.image_width == 64:
            import svg_models.dcgan_64 as model 
        elif opt.image_width == 128:
            import svg_models.dcgan_128 as model  
    elif opt.model == 'vgg':
        if opt.image_width == 64:
            import svg_models.vgg_64 as model
        elif opt.image_width == 128:
            import svg_models.vgg_128 as model
    elif opt.model == 'vgg_pretrained':
        if opt.image_width == 64:
            import svg_models.vgg_pretrained_64 as model
        elif opt.image_width == 128:
            raise NotImplementedError('Not implemented!')
    elif opt.model == 'deit_pretrained':
        if opt.image_width == 224:
            import svg_models.deit_pretrained_224 as model
        else:
            raise NotImplementedError('Not Implemented!')
    elif opt.model == 'clip_pretrained':
        if opt.image_width == 224:
            import svg_models.clip_pretrained_224 as model
        else:
            raise NotImplementedError('Not Implemented!')
    else:
        raise ValueError('Unknown model: %s' % opt.model)

    if load_model:
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = model.encoder(opt.g_dim, opt.channels)
        decoder = model.decoder(opt.g_dim, opt.channels)
        if 'clip' in opt.model:
            encoder.c5.apply(utils.init_weights)
        else:
            encoder.apply(utils.init_weights)
        decoder.apply(utils.init_weights)

    frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    if hasattr(opt, "freeze_encoder_weights") and opt.freeze_encoder_weights:
        if opt.model == 'vgg_pretrained' and opt.image_width == 64:
            encoder.c1.requires_grad_(False)
            encoder.c2.requires_grad_(False)
            encoder.c3.requires_grad_(False)
            encoder.c4.requires_grad_(False)
            print("Encoder weights frozen.")
        elif opt.model == 'deit_pretrained' and opt.image_width == 224:
            encoder.c1.requires_grad_(False)
            encoder.c2.requires_grad_(False)
            encoder.c3.requires_grad_(False)
            encoder.c4.requires_grad_(False)
            print("Encoder weights frozen.")
        elif opt.model == 'clip_pretrained' and opt.image_width == 224:
            encoder.c1.requires_grad_(False)
            print("Encoder weights frozen.")
        else:
            raise ValueError("This encoder cannot be frozen")

# --------- loss functions ------------------------------------
    mse_criterion = nn.MSELoss()
    def kl_criterion(mu1, logvar1, mu2, logvar2):
        # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) = 
        #   log( sqrt(
        # 
        sigma1 = logvar1.mul(0.5).exp() 
        sigma2 = logvar2.mul(0.5).exp() 
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        return kld.sum() / opt.batch_size

# --------- transfer to gpu ------------------------------------
    frame_predictor.cuda()
    posterior.cuda()
    prior.cuda()
    encoder.cuda()
    decoder.cuda()
    mse_criterion.cuda()

# --------- load a dataset ------------------------------------
    train_data, test_data = utils.load_dataset(opt)

    train_loader = DataLoader(train_data,
                              num_workers=opt.data_threads,
                              batch_size=opt.batch_size,
                              shuffle=False if opt.write_feat else True,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=opt.data_threads,
                             batch_size=opt.batch_size,
                             shuffle=False if opt.write_feat else True,
                             drop_last=True,
                             pin_memory=True)

    def get_training_batch():
        while True:
            for sequence in train_loader:
                batch = utils.normalize_data(opt, dtype, sequence)
                yield batch
    training_batch_generator = get_training_batch()

    def get_testing_batch():
        while True:
            for sequence in test_loader:
                batch = utils.normalize_data(opt, dtype, sequence)
                yield batch 
    testing_batch_generator = get_testing_batch()

# --------- write features ---------------------------------------

    def write_features(loader, feature_file):
        assert os.path.exists('%s/model.pth' % opt.model_dir), \
                'Model does not exist: %s/model.pth' % opt.model_dir
        # TODO: Only use one sample for now
        nsample = 1

        if not os.path.exists(os.path.dirname(feature_file)):
            os.makedirs(os.path.dirname(feature_file))

        outputs = []

        progress = progressbar.ProgressBar().start()
        counter = 0
        for seq_id, sequence in enumerate(loader):
            counter += 1
            progress.update(counter)

            images = sequence["images"]
            x = utils.normalize_data(opt, dtype, images)

            gen_seq = []
            encoded_states = []
            rollout_states = []
            gt_seq = [utils.make_images(x[i]) for i in range(len(x))]

            frame_predictor.hidden = frame_predictor.init_hidden()
            posterior.hidden = posterior.init_hidden()
            prior.hidden = prior.init_hidden()
            gen_seq.append(utils.make_images(x[0]))
            x_in = x[0]
            for i in range(1, opt.n_eval):
                h = encoder(x_in)
                if opt.last_frame_skip or i < opt.n_past:
                    h, skip = h
                else:
                    h, _ = h
                h = h

                if i == 1:
                    # First frame
                    rollout_states.append(h.cpu().numpy())

                if i < opt.n_past:
                    h_target = encoder(x[i])
                    h_target = h_target[0]
                    z_t, _, _ = posterior(h_target)
                    prior(h)
                    frame_predictor(torch.cat([h, z_t], 1))
                    x_in = x[i]
                    gen_seq.append(utils.make_images(x_in))
                    rollout_states.append(h_target.cpu().numpy())
                else:
                    z_t, _, _ = prior(h)
                    h = frame_predictor(torch.cat([h, z_t], 1))
                    x_in = decoder([h, skip])
                    gen_seq.append(utils.make_images(x_in))
                    rollout_states.append(h.cpu().numpy())

            for i in range(0, opt.n_eval):
                encoded_states.append(encoder(x[i])[0].cpu().numpy())

            outputs.append({
                #"predicted_images": np.stack(gen_seq, axis=1),
                #"true_images": np.stack(gt_seq, axis=1),
                #"raw_images": sequence["raw_images"].numpy(),
                "binary_labels": sequence["binary_labels"].numpy(),
                "reference_ids": sequence["reference_ids"].numpy(),
                "human_prob": sequence["human_prob"].numpy(),
                "encoded_states": np.stack(encoded_states, axis=1),
                "rollout_states": np.stack(rollout_states, axis=1),
                })

            # Write every 1000 sequences to not lose data
            if seq_id % 1000 == 0 and seq_id > 0:
                with open(feature_file, 'wb') as f:
                    pickle.dump(outputs, f)
                print('Results stored in %s' % feature_file)

        progress.finish()

        # Final write
        with open(feature_file, 'wb') as f:
            pickle.dump(outputs, f)
        print('Results stored in %s' % feature_file)
        return


    with torch.no_grad():
        if opt.write_feat == 'train':
            write_features(train_loader, feature_file)
            return
        elif opt.write_feat in ['test', 'human']:
            write_features(test_loader, feature_file)
            return

# --------- plotting funtions ------------------------------------
    def plot(x, epoch):
        nsample = 20 
        gen_seq = [[] for i in range(nsample)]
        gt_seq = [x[i] for i in range(len(x))]

        for s in range(nsample):
            frame_predictor.hidden = frame_predictor.init_hidden()
            posterior.hidden = posterior.init_hidden()
            prior.hidden = prior.init_hidden()
            gen_seq[s].append(x[0])
            x_in = x[0]
            for i in range(1, opt.n_eval):
                h = encoder(x_in)
                if opt.last_frame_skip or i < opt.n_past:	
                    h, skip = h
                else:
                    h, _ = h
                h = h
                if i < opt.n_past:
                    h_target = encoder(x[i])
                    h_target = h_target[0]
                    z_t, _, _ = posterior(h_target)
                    prior(h)
                    frame_predictor(torch.cat([h, z_t], 1))
                    x_in = x[i]
                    gen_seq[s].append(x_in)
                else:
                    z_t, _, _ = prior(h)
                    h = frame_predictor(torch.cat([h, z_t], 1))
                    x_in = decoder([h, skip])
                    gen_seq[s].append(x_in)

        to_plot = []
        gifs = [ [] for t in range(opt.n_eval) ]
        nrow = min(opt.batch_size, 10)
        for i in range(nrow):
            # ground truth sequence
            row = [] 
            for t in range(opt.n_eval):
                row.append(gt_seq[t][i])
            to_plot.append(row)

            # best sequence
            min_mse = 1e7
            for s in range(nsample):
                mse = 0
                for t in range(opt.n_eval):
                    mse +=  torch.sum( (gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu())**2 )
                if mse < min_mse:
                    min_mse = mse
                    min_idx = s

            s_list = [min_idx, 
                      np.random.randint(nsample), 
                      np.random.randint(nsample), 
                      np.random.randint(nsample), 
                      np.random.randint(nsample)]
            for ss in range(len(s_list)):
                s = s_list[ss]
                row = []
                for t in range(opt.n_eval):
                    row.append(gen_seq[s][t][i]) 
                to_plot.append(row)
            for t in range(opt.n_eval):
                row = []
                row.append(gt_seq[t][i])
                for ss in range(len(s_list)):
                    s = s_list[ss]
                    row.append(gen_seq[s][t][i])
                gifs[t].append(row)

        fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch) 
        utils.save_tensors_image(fname, to_plot)

        fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch) 
        utils.save_gif(fname, gifs)
        print("Saved at %s" % fname)


    def plot_rec(x, epoch):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        gen_seq = []
        gen_seq.append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_past+opt.n_future):
            h = encoder(x[i-1])
            h_target = encoder(x[i])
            if opt.last_frame_skip or i < opt.n_past:	
                h, skip = h
            else:
                h, _ = h
            h_target, _ = h_target
            h = h
            h_target = h_target
            z_t, _, _= posterior(h_target)
            if i < opt.n_past:
                frame_predictor(torch.cat([h, z_t], 1)) 
                gen_seq.append(x[i])
            else:
                h_pred = frame_predictor(torch.cat([h, z_t], 1))
                x_pred = decoder([h_pred, skip])
                gen_seq.append(x_pred)
       
        to_plot = []
        nrow = min(opt.batch_size, 10)
        for i in range(nrow):
            row = []
            for t in range(opt.n_past+opt.n_future):
                row.append(gen_seq[t][i]) 
            to_plot.append(row)
        fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch) 
        utils.save_tensors_image(fname, to_plot)
        print("Saved at %s" % fname)

    if opt.write_feat == 'plot':
        # Test plotting
        print("Plotting mode")
        with torch.no_grad():
            progress = progressbar.ProgressBar().start()
            counter = 10000
            for sequence in test_loader:
                counter += 1
                progress.update(counter)
                images = sequence["images"]
                x = utils.normalize_data(opt, dtype, images)
                plot(x, counter)
                plot_rec(x, counter)
            return

# --------- training funtions ------------------------------------
    def seconds_to_time(seconds_time):
        hours = seconds_time // 3600
        minutes = (seconds_time - (hours * 3600)) // 60
        seconds = seconds_time - (minutes * 60)
        return hours, minutes, seconds


    def exceeded_time(start_time, max_run_time):
        run_time = time.time() - start_time
        if run_time > max_run_time:
            print("Maximum run time (%d h %d m %d s) exceeded after %d h %d m %d s" % \
                    (*seconds_to_time(max_run_time),
                        *seconds_to_time(run_time)))
            return True
        else:
            return False

    def train(x):
        frame_predictor.zero_grad()
        posterior.zero_grad()
        prior.zero_grad()
        encoder.zero_grad()
        decoder.zero_grad()

        # initialize the hidden state.
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()

        mse = 0
        kld = 0
        for i in range(1, opt.n_past+opt.n_future):
            h = encoder(x[i-1])
            h_target = encoder(x[i])[0]
            if opt.last_frame_skip or i < opt.n_past:	
                h, skip = h
            else:
                h = h[0]
            z_t, mu, logvar = posterior(h_target)
            _, mu_p, logvar_p = prior(h)
            h_pred = frame_predictor(torch.cat([h, z_t], 1))
            x_pred = decoder([h_pred, skip])
            mse += mse_criterion(x_pred, x[i])
            kld += kl_criterion(mu, logvar, mu_p, logvar_p)

        loss = mse + kld*opt.beta
        loss.backward()

        frame_predictor_optimizer.step()
        posterior_optimizer.step()
        prior_optimizer.step()
        encoder_optimizer.step()
        decoder_optimizer.step()


        return mse.data.cpu().numpy()/(opt.n_past+opt.n_future), kld.data.cpu().numpy()/(opt.n_future+opt.n_past)

# --------- training loop ------------------------------------
    # train start time
    start_time = time.time()
    best_loss = 1e6
    for epoch in range(opt.niter):
        frame_predictor.train()
        posterior.train()
        prior.train()
        encoder.train()
        decoder.train()
        epoch_mse = 0
        epoch_kld = 0
        progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
        for i in range(opt.epoch_size):
            progress.update(i+1)
            x = next(training_batch_generator)

            # train frame_predictor 
            mse, kld = train(x)
            epoch_mse += mse
            epoch_kld += kld


        progress.finish()
        utils.clear_progressbar()

        print_message = '[%02d] mse loss: %.5f | kld loss: %.5f (%d)' % (epoch, epoch_mse/opt.epoch_size, epoch_kld/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size)
        print(print_message)

        with open('%s/log.txt' % opt.log_dir, 'a+') as f:
            f.write(print_message + '\n')

        # plot some stuff
        frame_predictor.eval()
        #encoder.eval()
        #decoder.eval()
        posterior.eval()
        prior.eval()
        
        with torch.no_grad():
            x = next(testing_batch_generator)
            plot(x, epoch)
            plot_rec(x, epoch)

        # save the model
        torch.save({
            'encoder': encoder,
            'decoder': decoder,
            'frame_predictor': frame_predictor,
            'posterior': posterior,
            'prior': prior,
            'opt': opt},
            '%s/latest_model.pth' % opt.model_dir)

        # save the best model
        current_loss = (epoch_mse + epoch_kld) / opt.epoch_size
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save({
                'encoder': encoder,
                'decoder': decoder,
                'frame_predictor': frame_predictor,
                'posterior': posterior,
                'prior': prior,
                'opt': opt},
                '%s/model.pth' % opt.model_dir)
            print('best loss: %f' % best_loss)

        if epoch % 10 == 0:
            print('log dir: %s' % opt.log_dir)

        if exceeded_time(start_time, max_run_time):
            break



class Objective(PhysOptObjective):
    def __init__(self,
            exp_key,
            seed,
            train_data,
            feat_data,
            output_dir,
            extract_feat,
            debug,
            max_run_time,
            model,
            freeze_encoder_weights):
        super().__init__(exp_key, seed, train_data, feat_data, output_dir,
                extract_feat, debug, max_run_time)
        self.model = model
        self.freeze_encoder_weights = freeze_encoder_weights


    def __call__(self, *args, **kwargs):
        results = super().__call__()
        if self.extract_feat:
            write_feat = 'train'
            if 'human' in self.feat_data['name']:
                write_feat = 'human'
            if 'test' in self.feat_data['name']:
                write_feat = 'test'
            #write_feat = 'plot'
            run(datasets=self.feat_data['data'], seed=self.seed, data_root='',
                    model_dir=self.model_dir, feature_file=self.feature_file,
                    write_feat=write_feat, model=self.model,
                    freeze_encoder_weights=self.freeze_encoder_weights)
        else:
            run(datasets=self.train_data['data'], seed=self.seed, data_root='',
                    model_dir=self.model_dir, feature_file=self.feature_file,
                    write_feat='', model=self.model,
                    freeze_encoder_weights=self.freeze_encoder_weights,
                    max_run_time=self.max_run_time)

        results['loss'] = 0.0
        results['model'] = self.model
        results['freeze_encoder_weights'] = self.freeze_encoder_weights
        return results


class VGGObjective(Objective):
    def __init__(self, *args, **kwargs):
        super(VGGObjective, self).__init__(*args, **kwargs,
                model='vgg', freeze_encoder_weights=False)


class VGGPretrainedObjective(Objective):
    def __init__(self, *args, **kwargs):
        super(VGGPretrainedObjective, self).__init__(*args, **kwargs,
                model='vgg_pretrained', freeze_encoder_weights=False)


class VGGPretrainedFrozenObjective(Objective):
    def __init__(self, *args, **kwargs):
        super(VGGPretrainedFrozenObjective, self).__init__(*args, **kwargs,
                model='vgg_pretrained', freeze_encoder_weights=True)


class DEITPretrainedObjective(Objective):
    def __init__(self, *args, **kwargs):
        super(DEITPretrainedObjective, self).__init__(*args, **kwargs,
                model='deit_pretrained', freeze_encoder_weights=False)


class DEITPretrainedFrozenObjective(Objective):
    def __init__(self, *args, **kwargs):
        super(DEITPretrainedFrozenObjective, self).__init__(*args, **kwargs,
                model='deit_pretrained', freeze_encoder_weights=True)


class CLIPPretrainedObjective(Objective):
    def __init__(self, *args, **kwargs):
        super(CLIPPretrainedObjective, self).__init__(*args, **kwargs,
                model='clip_pretrained', freeze_encoder_weights=False)


class CLIPPretrainedFrozenObjective(Objective):
    def __init__(self, *args, **kwargs):
        super(CLIPPretrainedFrozenObjective, self).__init__(*args, **kwargs,
                model='clip_pretrained', freeze_encoder_weights=True)



if __name__ == '__main__':
    run()
