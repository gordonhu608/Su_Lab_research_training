from utils import *

def plot_vae_training_plot(train_losses, test_losses, title, fname):
    elbo_train, recon_train, kl_train = train_losses[:, 0], train_losses[:, 1], train_losses[:, 2]
    elbo_test, recon_test, kl_test = test_losses[:, 0], test_losses[:, 1], test_losses[:, 2]
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, elbo_train, label='-elbo_train')
    plt.plot(x_train, recon_train, label='recon_loss_train')
    plt.plot(x_train, kl_train, label='kl_loss_train')
    plt.plot(x_test, elbo_test, label='-elbo_test')
    plt.plot(x_test, recon_test, label='recon_loss_test')
    plt.plot(x_test, kl_test, label='kl_loss_test')

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    savefig(fname)
    
def vae_save_results(dset_id, fn):
    assert dset_id in [1, 2]
    if dset_id == 1:
        train_data, test_data = load_pickled_data('svhn.pkl')
    else:
        train_data, test_data = load_pickled_data('cifar10.pkl')

    train_losses, test_losses, samples, reconstructions, interpolations = fn(train_data, test_data, dset_id)
    samples, reconstructions, interpolations = samples.astype('float32'), reconstructions.astype('float32'), interpolations.astype('float32')
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    plot_vae_training_plot(train_losses, test_losses, f'VAE Dataset {dset_id} Train Plot',
                           f'results/VAE_dset{dset_id}_train_plot.png')
    show_samples(samples, title=f'VAE Dataset {dset_id} Samples',
                 fname=f'results/VAE_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'VAE Dataset {dset_id} Reconstructions',
                 fname=f'results/VAE_dset{dset_id}_reconstructions.png')
    show_samples(interpolations, title=f'VAE Dataset {dset_id} Interpolations',
                 fname=f'results/VAE_dset{dset_id}_interpolations.png')