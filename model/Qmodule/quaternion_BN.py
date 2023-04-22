import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
# from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import torch
from typing import Optional, Any


def sqrt_init(shape, dtype=None):
    value = (1 / torch.sqrt(torch.tensor(4.0))) * torch.ones(shape)
    return value

def moving_average_update(x, value, momentum):
    """Compute the exponential moving average of a value.
      The moving average 'x' is updated with 'value' following:
      ```
      x = x * momentum + value * (1 - momentum)
      ```
      For example:
      >>> x = tf.Variable(0.0)
      >>> momentum=0.9
      >>> moving_average_update(x, value = 2.0, momentum=momentum).numpy()
      >>> x.numpy()
      0.2
      The result will be biased towards the initial value of the variable.
      If the variable was initialized to zero, you can divide by
      `1 - momentum ** num_updates` to debias it (Section 3 of
      [Kingma et al., 2015](https://arxiv.org/abs/1412.6980)):
      >>> num_updates = 1.0
      >>> x_zdb = x/(1 - momentum**num_updates)
      >>> x_zdb.numpy()
      2.0
      Args:
          x: A Variable, the moving average.
          value: A tensor with the same shape as `x`, the new value to be
            averaged in.
          momentum: The moving average momentum.
      Returns:
          The updated variable.
      """
    if isinstance(momentum, torch.Tensor):
        momentum = momentum.to(x)
    else:
        momentum = torch.tensor(momentum)
        momentum = momentum.to(x)

    if isinstance(value, torch.Tensor):
        value = value.to(x)
    else:
        value = torch.tensor(value)
        value = value.to(x)
    x=x * momentum + value * (1 - momentum)
    return x



def quaternion_standardization(input_centred,
                               Vrr, Vri, Vrj, Vrk, Vii,
                               Vij, Vik, Vjj, Vjk, Vkk,
                               layernorm=False, axis=1):
    size = input_centred.size()
    ndim = len(size)
    input_dim = input_centred.size()[axis] // 4
    variances_broadcast = [1] * ndim
    variances_broadcast[axis] = input_dim
    # if layernorm:
    #     variances_broadcast[0] = torch.shape(input_centred)[0]

    # Chokesky decomposition of 4x4 symmetric matrix
    Wrr = torch.sqrt(Vrr)
    Wri = (1.0 / Wrr) * (Vri)
    Wii = torch.sqrt((Vii - (Wri*Wri)))
    Wrj = (1.0 / Wrr) * (Vrj)
    Wij = (1.0 / Wii) * (Vij - (Wri*Wrj))
    Wjj = torch.sqrt((Vjj - (Wij*Wij + Wrj*Wrj)))
    Wrk = (1.0 / Wrr) * (Vrk)
    Wik = (1.0 / Wii) * (Vik - (Wri*Wrk))
    Wjk = (1.0 / Wjj) * (Vjk - (Wij*Wik + Wrj*Wrk))
    Wkk = torch.sqrt((Vkk - (Wjk*Wjk + Wik*Wik + Wrk*Wrk)))

    # Normalization. We multiply, x_normalized = W.x.
    # The returned result will be a quaternion standardized input
    # where the r, i, j, and k parts are obtained as follows:
    # x_r_normed = Wrr * x_r_cent + Wri * x_i_cent + Wrj * x_j_cent + Wrk * x_k_cent
    # x_i_normed = Wri * x_r_cent + Wii * x_i_cent + Wij * x_j_cent + Wik * x_k_cent
    # x_j_normed = Wrj * x_r_cent + Wij * x_i_cent + Wjj * x_j_cent + Wjk * x_k_cent
    # x_k_normed = Wrk * x_r_cent + Wik * x_i_cent + Wjk * x_j_cent + Wkk * x_k_cent
    # print("------------------bro----------------")
    # print(Wrr.shape)
    # print(variances_broadcast)
    broadcast_Wrr = torch.reshape(Wrr, variances_broadcast)
    broadcast_Wri = torch.reshape(Wri, variances_broadcast)
    broadcast_Wrj = torch.reshape(Wrj, variances_broadcast)
    broadcast_Wrk = torch.reshape(Wrk, variances_broadcast)
    broadcast_Wii = torch.reshape(Wii, variances_broadcast)
    broadcast_Wij = torch.reshape(Wij, variances_broadcast)
    broadcast_Wik = torch.reshape(Wik, variances_broadcast)
    broadcast_Wjj = torch.reshape(Wjj, variances_broadcast)
    broadcast_Wjk = torch.reshape(Wjk, variances_broadcast)
    broadcast_Wkk = torch.reshape(Wkk, variances_broadcast)

    cat_W_1 = torch.cat([broadcast_Wrr, broadcast_Wri, broadcast_Wrj, broadcast_Wrk], axis=axis)
    cat_W_2 = torch.cat([broadcast_Wri, broadcast_Wii, broadcast_Wij, broadcast_Wik], axis=axis)
    cat_W_3 = torch.cat([broadcast_Wrj, broadcast_Wij, broadcast_Wjj, broadcast_Wjk], axis=axis)
    cat_W_4 = torch.cat([broadcast_Wrk, broadcast_Wik, broadcast_Wjk, broadcast_Wkk], axis=axis)

    # if (axis == 1 and ndim != 3) or ndim == 2:
    #     centred_r = input_centred[:, :input_dim]
    #     centred_i = input_centred[:, input_dim:input_dim*2]
    #     centred_j = input_centred[:, input_dim*2:input_dim*3]
    #     centred_k = input_centred[:, input_dim*3:]
    # elif ndim == 3:
    #     centred_r = input_centred[:, :, :input_dim]
    #     centred_i = input_centred[:, :, input_dim:input_dim*2]
    #     centred_j = input_centred[:, :, input_dim*2:input_dim*3]
    #     centred_k = input_centred[:, :, input_dim*3:]
    if  ndim == 4:
        # centred_r = input_centred[:, :, :, :input_dim]
        # centred_i = input_centred[:, :, :, input_dim:input_dim*2]
        # centred_j = input_centred[:, :, :, input_dim*2:input_dim*3]
        # centred_k = input_centred[:, :, :, input_dim*3:]
        centred_r = input_centred[:, :input_dim, :, :]
        centred_i = input_centred[:, input_dim:input_dim * 2, :, :]
        centred_j = input_centred[:, input_dim * 2:input_dim * 3, :, :]
        centred_k = input_centred[:, input_dim * 3:, :, :]
    # elif axis == -1 and ndim == 5:
    #     centred_r = input_centred[:, :, :, :, :input_dim]
    #     centred_i = input_centred[:, :, :, :, input_dim:input_dim*2]
    #     centred_j = input_centred[:, :, :, :, input_dim*2:input_dim*3]
    #     centred_k = input_centred[:, :, :, :, input_dim*3:]
    else:
        raise ValueError(
            'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
            'axis: ' + str(axis) + '; ndim: ' + str(ndim) + '.'
        )

    input1 = torch.cat([centred_r, centred_r, centred_r, centred_r], axis=axis)
    input2 = torch.cat([centred_i, centred_i, centred_i, centred_i], axis=axis)
    input3 = torch.cat([centred_j, centred_j, centred_j, centred_j], axis=axis)
    input4 = torch.cat([centred_k, centred_k, centred_k, centred_k], axis=axis)

    output =  cat_W_1 * input1 + \
              cat_W_2 * input2 + \
              cat_W_3 * input3 + \
              cat_W_4 * input4

    #   Wrr * x_r_cent | Wri * x_r_cent | Wrj * x_r_cent | Wrk * x_r_cent
    # + Wri * x_i_cent | Wii * x_i_cent | Wij * x_i_cent | Wik * x_i_cent
    # + Wrj * x_j_cent | Wij * x_j_cent | Wjj * x_j_cent | Wjk * x_j_cent
    # + Wrk * x_k_cent | Wik * x_k_cent | Wjk * x_k_cent | Wkk * x_k_cent
    # -----------------------------------------------
    # = output
    # print("run Q BN")

    return output



def QuaternionBN(input_centred,
                 Vrr, Vri, Vrj, Vrk, Vii,
                 Vij, Vik, Vjj, Vjk, Vkk,
                 bias,
                 gamma_rr, gamma_ri, gamma_rj, gamma_rk, gamma_ii,
                 gamma_ij, gamma_ik, gamma_jj, gamma_jk, gamma_kk,
                 affine=True,
                 center=True, layernorm=False, axis=1):
    size = input_centred.size()
    ndim = len(size)
    input_dim = input_centred.size()[axis] // 4
    if affine:
        gamma_broadcast_shape = [1] * ndim
        gamma_broadcast_shape[axis] = input_dim
        # print('------------------------grmma_shjape-------------------------')
        # print(gamma_broadcast_shape)
        # print(gamma_ij.size())
    if center:
        broadcast_beta_shape = [1] * ndim
        broadcast_beta_shape[axis] = input_dim * 4

    if affine:
        standardized_output = quaternion_standardization(
            input_centred,
            Vrr, Vri, Vrj, Vrk, Vii,
            Vij, Vik, Vjj, Vjk, Vkk,
            layernorm,
            axis=axis
        )
        # print('------------------------stand---------------------------')
        # print(standardized_output)
        # Now we perform the scaling and shifting of the normalized x using
        # the scaling parameter
        #           [  gamma_rr gamma_ri gamma_rj gamma_rk  ]
        #   Gamma = [  gamma_ri gamma_ii gamma_ij gamma_ik  ]
        #           [  gamma_rj gamma_ij gamma_jj gamma_jk  ]
        #           [  gamma_rk gamma_ik gamma_jk gamma_kk  ]
        # and the shifting parameter
        #    Beta = [beta_r beta_i beta_j beta_k].T
        # where:
        # x_r_BN = gamma_rr * x_r + gamma_ri * x_i + gamma_rj * x_j + gamma_rk * x_k + beta_r
        # x_i_BN = gamma_ri * x_r + gamma_ii * x_i + gamma_ij * x_j + gamma_ik * x_k + beta_i
        # x_j_BN = gamma_rj * x_r + gamma_ij * x_i + gamma_jj * x_j + gamma_jk * x_k + beta_j
        # x_k_BN = gamma_rk * x_r + gamma_ik * x_i + gamma_jk * x_j + gamma_kk * x_k + beta_k

        broadcast_gamma_rr = torch.reshape(gamma_rr, gamma_broadcast_shape)
        broadcast_gamma_ri = torch.reshape(gamma_ri, gamma_broadcast_shape)
        broadcast_gamma_rj = torch.reshape(gamma_rj, gamma_broadcast_shape)
        broadcast_gamma_rk = torch.reshape(gamma_rk, gamma_broadcast_shape)
        broadcast_gamma_ii = torch.reshape(gamma_ii, gamma_broadcast_shape)
        broadcast_gamma_ij = torch.reshape(gamma_ij, gamma_broadcast_shape)
        broadcast_gamma_ik = torch.reshape(gamma_ik, gamma_broadcast_shape)
        broadcast_gamma_jj = torch.reshape(gamma_jj, gamma_broadcast_shape)
        broadcast_gamma_jk = torch.reshape(gamma_jk, gamma_broadcast_shape)
        broadcast_gamma_kk = torch.reshape(gamma_kk, gamma_broadcast_shape)

        cat_gamma_1 = torch.cat([broadcast_gamma_rr,
                                     broadcast_gamma_ri,
                                     broadcast_gamma_rj,
                                     broadcast_gamma_rk], axis=axis)
        cat_gamma_2 = torch.cat([broadcast_gamma_ri,
                                     broadcast_gamma_ii,
                                     broadcast_gamma_ij,
                                     broadcast_gamma_ik], axis=axis)
        cat_gamma_3 = torch.cat([broadcast_gamma_rj,
                                     broadcast_gamma_ij,
                                     broadcast_gamma_jj,
                                     broadcast_gamma_jk], axis=axis)
        cat_gamma_4 = torch.cat([broadcast_gamma_rk,
                                     broadcast_gamma_ik,
                                     broadcast_gamma_jk,
                                     broadcast_gamma_kk], axis=axis)
        # print('-------------------------cat_g1-------------------')
        # print(cat_gamma_1)
        # print('------------------------cat_g2-------------------------')
        # print(cat_gamma_2)
        # print('------------------------cat_g3-------------------------')
        # print(cat_gamma_3)
        # print('------------------------cat_g4-------------------------')
        # print(cat_gamma_4)
        # if (axis == 1 and ndim != 3) or ndim == 2:
        #     centred_r = standardized_output[:, :input_dim]
        #     centred_i = standardized_output[:, input_dim:input_dim * 2]
        #     centred_j = standardized_output[:, input_dim * 2:input_dim * 3]
        #     centred_k = standardized_output[:, input_dim * 3:]
        # elif ndim == 3:
        #     centred_r = standardized_output[:, :, :input_dim]
        #     centred_i = standardized_output[:, :, input_dim:input_dim * 2]
        #     centred_j = standardized_output[:, :, input_dim * 2:input_dim * 3]
        #     centred_k = standardized_output[:, :, input_dim * 3:]
        if   ndim == 4:
            centred_r = standardized_output[:, :input_dim, :,: ]
            centred_i = standardized_output[:, input_dim:input_dim * 2, :,: ]
            centred_j = standardized_output[:, input_dim * 2:input_dim * 3, :,: ]
            centred_k = standardized_output[:,  input_dim * 3:, :,:]
        # elif axis == -1 and ndim == 5:
        #     centred_r = standardized_output[:, :, :, :, :input_dim]
        #     centred_i = standardized_output[:, :, :, :, input_dim:input_dim * 2]
        #     centred_j = standardized_output[:, :, :, :, input_dim * 2:input_dim * 3]
        #     centred_k = standardized_output[:, :, :, :, input_dim * 3:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: ' + str(axis) + '; ndim: ' + str(ndim) + '.'
            )

        input1 = torch.cat([centred_r, centred_r, centred_r, centred_r], axis=axis)
        input2 = torch.cat([centred_i, centred_i, centred_i, centred_i], axis=axis)
        input3 = torch.cat([centred_j, centred_j, centred_j, centred_j], axis=axis)
        input4 = torch.cat([centred_k, centred_k, centred_k, centred_k], axis=axis)
        # print('------------------------input1-------------------------')
        # print(input1)
        # print('------------------------input2-------------------------')
        # print(input2)
        # print('------------------------input3-------------------------')
        # print(input3)
        # print('------------------------input4-------------------------')
        # print(input4)
        # print("center:")
        # print(center)
        if center:
            broadcast_beta = torch.reshape(bias, broadcast_beta_shape)
            # print('--------------------------------output---------------------')
            output = cat_gamma_1 * input1 + \
                     cat_gamma_2 * input2 + \
                     cat_gamma_3 * input3 + \
                     cat_gamma_4 * input4 + \
                     broadcast_beta
            # print('--------------------------------output---------------------')
            # print(output)
            return output
        else:
            return cat_gamma_1 * input1 + \
                   cat_gamma_2 * input2 + \
                   cat_gamma_3 * input3 + \
                   cat_gamma_4 * input4
    else:
        if center:
            broadcast_beta = torch.reshape(bias, broadcast_beta_shape)
            return input_centred + broadcast_beta
        else:
            return input_centred


class _BatchNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True,epsilon=1e-4,center=True,axis=1):
        super(_BatchNorm, self).__init__()
        self.epsilon = epsilon
        self.center = center
        self.affine =affine
        self.eps = eps
        self.momentum = momentum
        self.axis =axis
        self.track_running_stats = track_running_stats

        if self.affine:
            self.gamma_rr = Parameter(sqrt_init(int(num_features / 4)))
            self.gamma_ii = Parameter(sqrt_init(int(num_features / 4)))
            self.gamma_jj = Parameter(sqrt_init(int(num_features / 4)))
            self.gamma_kk = Parameter(sqrt_init(int(num_features / 4)))
            self.gamma_ri = Parameter(torch.zeros(int(num_features / 4)))
            self.gamma_rj = Parameter(torch.zeros(int(num_features / 4)))
            self.gamma_rk = Parameter(torch.zeros(int(num_features / 4)))
            self.gamma_ij = Parameter(torch.zeros(int(num_features / 4)))
            self.gamma_ik = Parameter(torch.zeros(int(num_features / 4)))
            self.gamma_jk = Parameter(torch.zeros(int(num_features / 4)))

            self.moving_Vrr = sqrt_init(int(num_features / 4))
            self.moving_Vii = sqrt_init(int(num_features / 4))
            self.moving_Vjj = sqrt_init(int(num_features / 4))
            self.moving_Vkk = sqrt_init(int(num_features / 4))
            self.moving_Vri = torch.zeros(int(num_features / 4))
            self.moving_Vrj = torch.zeros(int(num_features / 4))
            self.moving_Vrk = torch.zeros(int(num_features / 4))
            self.moving_Vij = torch.zeros(int(num_features / 4))
            self.moving_Vik = torch.zeros(int(num_features / 4))
            self.moving_Vjk = torch.zeros(int(num_features / 4))

        else:
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_jj = None
            self.gamma_kk = None
            self.gamma_ri = None
            self.gamma_rj = None
            self.gamma_rk = None
            self.gamma_ij = None
            self.gamma_ik = None
            self.gamma_jk = None

            self.moving_Vrr = None
            self.moving_Vii = None
            self.moving_Vjj = None
            self.moving_Vkk = None
            self.moving_Vri = None
            self.moving_Vrj = None
            self.moving_Vrk = None
            self.moving_Vij = None
            self.moving_Vik = None
            self.moving_Vjk = None
        # print('-----------------------gammm-rr----------------------')
        # print(self.gamma_rr)
        # print('-----------------------gammm-ri----------------------')
        # print(self.gamma_ri)
        # print('-----------------------gammm-rj----------------------')
        # print(self.gamma_rj)
        # print('-----------------------gammm-rk----------------------')
        # print(self.gamma_rk)
        if self.center:
            self.bias = Parameter(torch.zeros(num_features))
            self.moving_mean = Parameter(torch.empty(num_features))
            # self.bias = Parameter(torch.zeros(num_features * 4))
            # self.moving_mean = Parameter(torch.empty(num_features * 4))
        else:
            self.bias = None
            self.moving_mean = None

    def forward(self, input: Tensor) -> Tensor:
        # self._check_input_dim(input)
        input_shape = input.size()
        ndim = len(input_shape)
        reduction_axes = list(range(ndim))
        del reduction_axes[self.axis]  # 删除通道维度
        input_dim = input_shape[self.axis] // 4
        mu = torch.mean(input, dim=reduction_axes)
        broadcast_mu_shape = [1] * len(input_shape)
        broadcast_mu_shape[self.axis] = input_shape[self.axis]
        broadcast_mu = torch.reshape(mu, broadcast_mu_shape)
        # print('---------------------mu-----------------------')
        # print(broadcast_mu)
        if self.center:
            input_centred = input - broadcast_mu
        else:
            input_centred = input
        centred_squared = input_centred ** 2

        # if  ndim == 2:
        #     centred_squared_r = centred_squared[:, :input_dim]
        #     centred_squared_i = centred_squared[:, input_dim:input_dim * 2]
        #     centred_squared_j = centred_squared[:, input_dim * 2:input_dim * 3]
        #     centred_squared_k = centred_squared[:, input_dim * 3:]
        #     centred_r = input_centred[:, :input_dim]
        #     centred_i = input_centred[:, input_dim:input_dim * 2]
        #     centred_j = input_centred[:, input_dim * 2:input_dim * 3]
        #     centred_k = input_centred[:, input_dim * 3:]
        # elif ndim == 3:
        #     centred_squared_r = centred_squared[:, :, :input_dim]
        #     centred_squared_i = centred_squared[:, :, input_dim:input_dim * 2]
        #     centred_squared_j = centred_squared[:, :, input_dim * 2:input_dim * 3]
        #     centred_squared_k = centred_squared[:, :, input_dim * 3:]
        #     centred_r = input_centred[:, :, :input_dim]
        #     centred_i = input_centred[:, :, input_dim:input_dim * 2]
        #     centred_j = input_centred[:, :, input_dim * 2:input_dim * 3]
        #     centred_k = input_centred[:, :, input_dim * 3:]
        if  ndim == 4:
            centred_squared_r = centred_squared[:, :input_dim, :, :]
            centred_squared_i = centred_squared[:, input_dim:input_dim * 2, :, :]
            centred_squared_j = centred_squared[:, input_dim * 2:input_dim * 3, :, :]
            centred_squared_k = centred_squared[:, input_dim * 3:, :,:]
            centred_r = input_centred[:, :input_dim, :,]
            centred_i = input_centred[:, input_dim:input_dim * 2, :, : ]
            centred_j = input_centred[:, input_dim * 2:input_dim * 3, :, :]
            centred_k = input_centred[:, input_dim * 3:, :, : ]
        # elif ndim == 5:
        #     centred_squared_r = centred_squared[:, :, :, :, :input_dim]
        #     centred_squared_i = centred_squared[:, :, :, :, input_dim:input_dim * 2]
        #     centred_squared_j = centred_squared[:, :, :, :, input_dim * 2:input_dim * 3]
        #     centred_squared_k = centred_squared[:, :, :, :, input_dim * 3:]
        #     centred_r = input_centred[:, :, :, :, :input_dim]
        #     centred_i = input_centred[:, :, :, :, input_dim:input_dim * 2]
        #     centred_j = input_centred[:, :, :, :, input_dim * 2:input_dim * 3]
        #     centred_k = input_centred[:, :, :, :, input_dim * 3:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: '+'; ndim: ' + str(ndim) + '.'
            )
        # print(centred_squared_i.size())
        if self.affine:
            # 这里input_center = x-E(x)
            # 计算协方差Vrr 协方差Cov(x,y)=E[(x-E(x)(y-E(y))]
            Vrr = torch.mean(centred_squared_r,axis=reduction_axes) + self.epsilon
            Vii = torch.mean(centred_squared_i,axis=reduction_axes) + self.epsilon
            Vjj = torch.mean(centred_squared_j,axis=reduction_axes) + self.epsilon
            Vkk = torch.mean(centred_squared_k,axis=reduction_axes) + self.epsilon
            Vri = torch.mean(centred_r * centred_i,axis=reduction_axes,)
            Vrj = torch.mean(centred_r * centred_j,axis=reduction_axes,)
            Vrk = torch.mean(centred_r * centred_k,axis=reduction_axes,)
            Vij = torch.mean(centred_i * centred_j,axis=reduction_axes,)
            Vik = torch.mean(centred_i * centred_k,axis=reduction_axes,)
            Vjk = torch.mean(centred_j * centred_k,axis=reduction_axes,)
            # print('-------------------------vrr------------------------')
            # print(Vrr)
            # print('-------------------------vii------------------------')
            # print(Vii)
            # print('-------------------------vjj------------------------')
            # print(Vjj)
            # print('-------------------------vkk------------------------')
            # print(Vkk)
            # print('-------------------------vri------------------------')
            # print(Vri)
            # print('-------------------------vrj------------------------')
            # print(Vrj)
            # print('-------------------------vrk------------------------')
            # print(Vrk)
            # print('-------------------------vij------------------------')
            # print(Vij)
            # print('-------------------------vik------------------------')
            # print(Vik)
            # print('-------------------------vjk------------------------')
            # print(Vjk)
        elif self.center:
            Vrr = None
            Vii = None
            Vjj = None
            Vkk = None
            Vri = None
            Vrj = None
            Vrk = None
            Vij = None
            Vik = None
            Vjk = None

        input_bn = QuaternionBN(
            input_centred,
            Vrr, Vri, Vrj, Vrk, Vii,
            Vij, Vik, Vjj, Vjk, Vkk,
            self.bias,
            self.gamma_rr, self.gamma_ri,
            self.gamma_rj, self.gamma_rk,
            self.gamma_ii, self.gamma_ij,
            self.gamma_ik, self.gamma_jj,
            self.gamma_jk, self.gamma_kk,
            self.affine, self.center,
            axis=self.axis
        )
        # print('---------------------input_bn-------------------------------')
        # print(input_bn)
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        # if self.training and self.track_running_stats:
        #     # TODO: if statement only here to tell the jit to skip emitting this when it is None
        #     if self.num_batches_tracked is not None:  # type: ignore
        #         self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
        #         if self.momentum is None:  # use cumulative moving average
        #             exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        #         else:  # use exponential moving average
        #             exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        if bn_training in {0,False}:
            return input_bn
        else:
            update_list = []
            if self.center:
                update_list.append(moving_average_update(self.moving_mean, mu, self.momentum))
            if self.affine:
                update_list.append(moving_average_update(self.moving_Vrr, Vrr, self.momentum))
                update_list.append(moving_average_update(self.moving_Vii, Vii, self.momentum))
                update_list.append(moving_average_update(self.moving_Vjj, Vjj, self.momentum))
                update_list.append(moving_average_update(self.moving_Vkk, Vkk, self.momentum))
                update_list.append(moving_average_update(self.moving_Vri, Vri, self.momentum))
                update_list.append(moving_average_update(self.moving_Vrj, Vrj, self.momentum))
                update_list.append(moving_average_update(self.moving_Vrk, Vrk, self.momentum))
                update_list.append(moving_average_update(self.moving_Vij, Vij, self.momentum))
                update_list.append(moving_average_update(self.moving_Vik, Vik, self.momentum))
                update_list.append(moving_average_update(self.moving_Vjk, Vjk, self.momentum))
            # self.add_update(update_list, input)
            def normalize_inference():
                if self.center:
                    inference_centred = input - torch.reshape(self.moving_mean, broadcast_mu_shape)
                else:
                    inference_centred = input
                return QuaternionBN(
                    inference_centred,
                    self.moving_Vrr, self.moving_Vri,
                    self.moving_Vrj, self.moving_Vrk,
                    self.moving_Vii, self.moving_Vij,
                    self.moving_Vik, self.moving_Vjj,
                    self.moving_Vjk, self.moving_Vkk,
                    self.bias,
                    self.gamma_rr, self.gamma_ri,
                    self.gamma_rj, self.gamma_rk,
                    self.gamma_ii, self.gamma_ij,
                    self.gamma_ik, self.gamma_jj,
                    self.gamma_jk, self.gamma_kk,
                    self.scale, self.center, axis=self.axis
                )
        if bn_training:
            return  input_bn
        else:
            return normalize_inference()

class BatchNorm1d(_BatchNorm):
    r"""Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .
    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.
    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.
    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.
    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    Examples::
        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

class BatchNorm2d(_BatchNorm):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .
    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.
    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.
    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.
    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples::
        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class BatchNorm3d(_BatchNorm):
    r"""Applies Batch Normalization over a 5D input (a mini-batch of 3D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .
    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.
    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.
    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.
    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric Batch Normalization
    or Spatio-temporal Batch Normalization.
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)
    Examples::
        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
