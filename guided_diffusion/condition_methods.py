from abc import ABC, abstractmethod
import torch
import os
import re
import numpy as np
import matplotlib.pyplot as plt

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, measurement_f, regular_scale,fig_num=0, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            prediction = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - prediction
            norm_x = torch.linalg.norm(x_prev)
            norm_a = torch.linalg.norm(difference)
            norm_a_tik = norm_a + regular_scale * norm_x
            norm_grad = torch.autograd.grad(outputs=norm_a, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm_a
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.decline = kwargs.get('decline', 1.0)
        self.allnum = 0
        self.regular_scale = kwargs.get('regular_scale', 0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, measurement_f, lam, fig_num=0, **kwargs):
        norm_grad, norm_a = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement_f=measurement_f,measurement=measurement, regular_scale=self.regular_scale,fig_num=fig_num, **kwargs)
        x_new = x_t - norm_grad * self.scale * (1+lam**2)/2

        self.allnum += 1
        if self.allnum == 100:
            self.allnum *= 0
            self.scale *= self.decline
        return x_new, norm_a
