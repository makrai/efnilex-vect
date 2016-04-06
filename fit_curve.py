import matplotlib.pyplot as plt
import logging
import numpy as np
import os
from scipy.optimize import curve_fit


class CurveFitter():
    def __init__(self):
        proj_dir =  os.path.expanduser('~/repo/paper/LREC16/triang_filter/dat')
        self.acc_by_pvt = np.loadtxt(os.path.join(proj_dir, 'fig-eval-edt.dat'))
        self.acc_by_cos = np.loadtxt(os.path.join(proj_dir, 'fig-eval-cos.dat'))

    def exp(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def powl(self, x, a, b, c, k):
        return a * np.power(b * x + c, -k)

    def main(self):
        scale = 1000
        xdata = self.acc_by_pvt[:,0] / scale
        y_pvt = self.acc_by_pvt[:,1]
        y_cos = self.acc_by_cos[:,1]
        for accr in [y_pvt, y_cos]:
            logging.info('{:.2%}'.format(sum(accr[:len(y_cos)])/len(y_cos)))
        # TODO p0 mindke1t family-hez
        popt_pvt_exp, pcov = curve_fit(self.exp, xdata, y_pvt)
        popt_pvt_powl, pcov = curve_fit(self.powl, xdata, y_pvt)
        popt_cos_exp, pcov = curve_fit(self.exp, xdata[:len(y_cos)], y_cos)
        popt_cos_powl, pcov = curve_fit(self.powl, xdata[:len(y_cos)], y_cos)
        logging.debug(popt_pvt_exp)
        logging.debug(popt_cos_exp)
        logging.debug(popt_cos_powl)
        logging.debug(popt_cos_powl)
        for mse in [
                np.sum((y_pvt - self.exp(xdata, *popt_pvt_exp))**2) / len(y_pvt),
                np.sum((y_cos - self.exp(xdata[:len(y_cos)], *popt_cos_exp))**2) / len(y_cos),
                np.sum((y_pvt - self.powl(xdata, *popt_pvt_powl))**2) / len(y_pvt),
                np.sum((y_cos - self.powl(xdata[:len(y_cos)], *popt_cos_powl))**2) / len(y_cos)
        ]:
            logging.info('{:.4e}'.format(mse))
        plt.plot(xdata, y_pvt, label='pvt')
        plt.plot(xdata[:len(y_cos)], y_cos, label='cos')
        plt.plot(xdata, self.exp(xdata, *popt_pvt_exp), label='# of pivots, exp')
        plt.plot(xdata, self.powl(xdata, *popt_pvt_powl), label='# of pivots, powl')
        plt.plot(xdata, self.exp(xdata, *popt_cos_exp), label='cos, exp')
        plt.plot(xdata, self.powl(xdata, *popt_cos_powl), label='cos, powl')
        plt.legend() 
        #plt.show()
        np.savetxt('best_fit_exp_pivot.dat', np.stack((xdata * scale, self.exp(xdata, *popt_pvt_exp)), axis=1))
        np.savetxt('best_fit_powl_pivot.dat', np.stack((xdata * scale, self.powl(xdata, *popt_pvt_powl)), axis=1))
        np.savetxt('best_fit_exp_cos.dat', np.stack((xdata * scale, self.exp(xdata, *popt_cos_exp)), axis=1))
        np.savetxt('best_fit_powl_cos.dat', np.stack((xdata * scale, self.powl(xdata, *popt_cos_powl)), axis=1))


if __name__ == '__main__':
    fmt = "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.DEBUG)
    CurveFitter().main()
