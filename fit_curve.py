import matplotlib.pyplot as plt
import logging
import numpy as np
import os
from scipy.optimize import curve_fit


class CurveFitter():
    def __init__(self, acc_by_pvt, acc_by_cos):
        self.acc_by_pvt = acc_by_pvt
        self.acc_by_cos = acc_by_cos

    def exp(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def powl(self, x, a, b, c, k):
        return a * np.power(b * x + c, -k)

    def fit(self):
        self.popt_pvt_exp, pcov = curve_fit(self.exp, self.xdata, self.y_pvt)
        self.popt_pvt_powl, pcov = curve_fit(self.powl, self.xdata, self.y_pvt)
        self.popt_cos_exp, pcov = curve_fit(self.exp,
                                            self.xdata[:len(self.y_cos)],
                                            self.y_cos)
        self.popt_cos_powl, pcov = curve_fit(self.powl,
                                             self.xdata[:len(self.y_cos)],
                                             self.y_cos)
        logging.info('Paramerters of best exp curve to pvt: {}'.format(
            self.popt_pvt_exp))
        logging.info('Paramerters of best exp curve to cos: {}'.format(
            self.popt_cos_exp))
        logging.info('Paramerters of best powl curve to pvt: {}'.format(
            self.popt_pvt_powl))
        logging.info('Paramerters of best powl curve to cos: {}'.format(
            self.popt_cos_powl))

    def logg_mse(self):
        logging.info(
            'Mean square errors '\
            '(exp approximation of pvt and cos, then powl approximation of both)')
        mse_l = [
                np.sum((self.y_pvt - self.exp(
                    self.xdata, *self.popt_pvt_exp))**2) / len(self.y_pvt),
                np.sum((self.y_cos - self.exp(
                    self.xdata[:len(self.y_cos)], *self.popt_cos_exp))**2) / len(self.y_cos),
                np.sum((self.y_pvt - self.powl(
                    self.xdata, *self.popt_pvt_powl))**2) / len(self.y_pvt),
                np.sum((self.y_cos - self.powl(
                    self.xdata[:len(self.y_cos)], *self.popt_cos_powl))**2) / len(self.y_cos)
        ]
        for mse in mse_l:
            logging.info('{:.4e}'.format(mse))

    def plot(self):
        plt.plot(self.xdata, self.y_pvt, label='pvt')
        plt.plot(self.xdata[:len(self.y_cos)], self.y_cos, label='cos')
        plt.plot(self.self.xdata, self.exp(self.xdata, *self.popt_pvt_exp), label='# of pivots, exp')
        plt.plot(self.self.xdata, self.powl(self.xdata, *self.popt_pvt_powl), label='# of pivots, powl')
        plt.plot(self.self.xdata, self.exp(self.xdata, *self.popt_cos_exp), label='cos, exp')
        plt.plot(self.self.xdata, self.powl(self.xdata, *self.popt_cos_powl), label='cos, powl')
        plt.legend() 
        plt.show()

    def savetxt(self)
        np.savetxt('best_fit_exp_pivot.dat', np.stack((self.self.xdata * scale, self.exp(self.xdata, *self.popt_pvt_exp)), axis=1))
        np.savetxt('best_fit_powl_pivot.dat', np.stack((self.self.xdata * scale, self.powl(self.xdata, *self.popt_pvt_powl)), axis=1))
        np.savetxt('best_fit_exp_cos.dat', np.stack((self.self.xdata * scale, self.exp(self.xdata, *self.popt_cos_exp)), axis=1))
        np.savetxt('best_fit_powl_cos.dat', np.stack((self.self.xdata * scale, self.powl(self.xdata, *self.popt_cos_powl)), axis=1)) 

    def main(self):
        scale = 1000
        self.xdata = self.acc_by_pvt[:,0] / scale
        self.y_pvt = self.acc_by_pvt[:,1]
        self.y_cos = self.acc_by_cos[:,1]
        len_cos = len(self.y_cos)
        len_pvt = len(self.y_pvt)
        assert len(len_cos < len_pvt)
        for accr in [self.y_pvt, self.y_cos]:
            logging.info('Overall accuracy (longer dict cut): {:.2%}'.format(
                sum(accr[:len(self.y_cos)])/len(self.y_cos)))
        self.fit()
        self.logg_mse()
        #self.plot()
        self.savetxt()


if __name__ == '__main__':
    fmt = "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.DEBUG)
    proj_dir =  os.path.expanduser('~/repo/paper/LREC16/triang_filter/dat')
    acc_by_pvt = np.loadtxt(os.path.join(proj_dir, 'fig-eval-edt.dat'))
    acc_by_cos = np.loadtxt(os.path.join(proj_dir, 'fig-eval-cos.dat'))
    CurveFitter(acc_by_pvt, acc_by_cos).main()
