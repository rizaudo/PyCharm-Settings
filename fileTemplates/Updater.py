import chainer
from chainer import Variable
import chainer.functions as F
import chainer.functions as L
from chainer import training
import numpy as np


class MyUpdater(training.StandardUpdater):
    """
    Base Template for WGAN based GANs.

    Generator must had make_input_z attribute
    Args:
        *args: Variable length argument list.
        **kwargs: Arbitary keyword arguments.

    Raises:
        AttributeError: if generator haven't method make_input_z
    """
    def __init__(self, *args, **kwargs):
        super(MyUpdater, self).__init__()
        self.generator, self.critic = kwargs.pop('models')
        self.num_critic = kwargs.pop('num_critic')
        # λ
        self.lam = kwargs.pop('lambda')
        if not callable(getattr(self.generator, "make_input_z", None)):
            raise AttributeError('generator must have method make_input_z')

    def _grab_batch(self):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        batch = self.converter(batch, self.device)
        return batchsize, batch

    def update_core(self):
        gen_optimizer = self.get_optimizer('generator')
        critic_optimizer = self.get_optimizer('critic')
        xp = self.generator.xp
        for i in range(self.num_critic):
            batchsize, batch = self._grab_batch()

            # Data Init Zone
            real_data = Variable(batch) / 255.
            z = Variable(xp.asarray(self.generator.make_input_z(batchsize)))

            # Generator
            gen_data = self.generator(z)

            # Critic(Discriminator)
            critic_real = self.critic(real_data)
            critic_fake = self.critic(gen_data)

            # Loss Function WRITE HERE~~~~~

            # Simple WGAN sample
            # loss_gan = F.average(critic_fake - critic_real)
            if i == 0:
                # Generator Loss function Write Here
                # Simple WGAN
                # loss_gen = F.average(-critic_fake)
                pass

